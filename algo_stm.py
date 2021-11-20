from ._types import Params
from ._types import LossClosure
from ._types import StepResult
from ._types import TwoFloats

import torch

from math import sqrt

from copy import deepcopy


class STM_Method(torch.optim.Optimizer):
    """ 
    Implement STM (Similar Triangle Method)
    checkout paper on arxiv there much interesting facts
    https://arxiv.org/pdf/2102.02921.pdf
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: external learning rate (default: 1e-3)
        
    """

    @staticmethod
    def _calculate_next_alpha_A(alpha: float, A: float, Lip: float, mu: float = 0) -> TwoFloats:
        """ Calculate next alpha, A sequence

            Arguments:
                alpha_k: Last alpha value
                A_k: Last A value
                Lip: Lipsitz constant of gradient https://encyclopediaofmath.org/wiki/Lipschitz_constant

            Return:
                New values (alpha_{k + 1}, A_{k + 1})
        """

        if Lip <= 0:
            raise ValueError('Lipsitz constant must be greater than zero, but it - {}'.format(Lip))

        if Lip < mu:
            raise ValueError('Lipsitz must be not less than strong convex')

        if mu < 0:
            raise ValueError('mu must be not less zero - mu = {}'.format(mu))

        lr = 1 / Lip
        new_alpha = (1 + mu * A) * lr / 2 + sqrt(
            (1 + mu * A) ** 2 / 4 * lr * lr + (1 + mu * A) * lr
        )

        new_A = A + new_alpha

        return (new_alpha, new_A)
    

    def _recalc_main_param(self, param_main, param_z, param_x) -> None:
        """ Calculate next tilde(x) vector
        """

        new_alpha, new_A = STM_Method._calculate_next_alpha_A(self.alpha, self.A, self.lip, self.mu)
        coef_alpha, coef_A = new_alpha / new_A, self.A / new_A

        param_main.data = coef_A * param_x.detach().clone()
        param_main.data.add_(coef_alpha * param_z.detach())


    def __init__(self, params: Params, lr: float = 1e-3, mu: float = 0): 
        
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))        
        
        if mu * lr > 1:
            raise ValueError('strong convex constant must be less than Lipstiz')

        self.lr = lr
        self.lip = 1 / lr
    
        self.alpha = lr
        self.A = lr
        self.mu = mu

        defaults = dict(lr=lr, mu=mu)

        super(STM_Method, self).__init__(params, defaults)

        self.x_groups = deepcopy(self.param_groups)
        self.z_groups = deepcopy(self.param_groups) 


    @torch.no_grad()
    def step(self, closure: LossClosure = None) -> StepResult:
        """Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for index_group, group in enumerate(self.param_groups):
            for index_param, param in enumerate(group['params']):

                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                if param.grad is None:
                    continue
                
                grad_step = param.grad.detach()

                last_A = self.A
                self.alpha, self.A = STM_Method._calculate_next_alpha_A(self.alpha, self.A, self.lip, self.mu)

                coef_alpha, coef_A = self.alpha / self.A, last_A / self.A

                param_z = self.z_groups[index_group]['params'][index_param]
                param_x = self.x_groups[index_group]['params'][index_param]

                coef_step = self.alpha / (1 + self.mu * self.A)

                grad_step_correct = grad_step.data + self.mu * (param_z.detach() - param.detach())

                param_z.data.add_(-coef_step * grad_step_correct)
                param_x.data.mul_(coef_A).add_(coef_alpha * param_z.detach())

                self._recalc_main_param(param, param_z, param_x)
