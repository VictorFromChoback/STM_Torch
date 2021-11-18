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
    def _calculate_next_alpha_A(alpha: float, A: float, Lip: float) -> TwoFloats:
        """ Calculate next alpha, A sequence

            Arguments:
                alpha_k: Last alpha value
                A_k: Last A value
                Lip: Lipsitz constant of gradient https://encyclopediaofmath.org/wiki/Lipschitz_constant

            Return:
                New values (alpha_{k + 1}, A_{k + 1})
        """

        new_alpha = 1 / (2 * Lip) + sqrt(1 / (4 * Lip * Lip) + 1 / Lip)
        new_A = A + new_alpha

        return (new_alpha, new_A)


    def _make_zero_step_tilde(self) -> None:
        """ Update vector tilde_x_k

            Arguments:
                Nothing

            Return:
                Nothing
        """

        for index_group, group in enumerate(self.param_groups):
            for index_param, param_x in enumerate(group['params']):
                
                last_A = self.A
                new_alpha, new_A = STM_Method._calculate_next_alpha_A(self.alpha, self.A, self.lip)

                coef_A = last_A / new_A
                coef_alpha = new_alpha / new_A

                param_z = self.z_groups[index_group]['params'][index_param]
                param_tilde_x = self.xtilde_groups[index_group]['params'][index_param]

                param_tilde_x.data = coef_A * param_x.detach().clone()
                param_tilde_x.data.add_(coef_alpha * param_z.data)
    

    def _recalc_main_param(self, param_main, param_z, param_x) -> None:
        """ Calculate next tilde(x) vector
        """

        new_alpha, new_A = STM_Method._calculate_next_alpha_A(self.alpha, self.A, self.lip)
        coef_alpha, coef_A = new_alpha / new_A, self.A / new_A

        param_main.data = coef_A * param_x.detach().clone()
        param_main.data.add_(coef_alpha * param_z.detach())


    def __init__(self, params: Params, lr: float = 1e-3): 
        
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))        
        
        self.lr = lr
        self.lip = 1 / lr
    
        self.alpha = lr
        self.A = lr

        defaults = dict(lr=lr)

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
                self.alpha, self.A = STM_Method._calculate_next_alpha_A(self.alpha, self.A, self.lip)

                coef_alpha, coef_A = self.alpha / self.A, last_A / self.A

                param_z = self.z_groups[index_group]['params'][index_param]
                param_x = self.x_groups[index_group]['params'][index_param]

                param_z.data.add_(-self.alpha * grad_step)
                param_x.data.mul_(coef_A).add_(coef_alpha * param_z.detach())

                self._recalc_main_param(param, param_z, param_x)
        
