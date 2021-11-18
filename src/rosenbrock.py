from ._types import TwoFloats
import torch
import numpy as np

def rosenbrock(xy):
    """ calculate Rosenbrock function value
        https://en.wikipedia.org/wiki/Rosenbrock_function

        Arguments:
            xy: Pair of two floats
    """

    x, y = xy
    return (1 - xy[0]) ** 2 + 100 * (xy[1] - xy[0] ** 2) ** 2


def optimize_rosenbrock(start_point, optim, iterations: int, **optimizer_kwargs):
    """ Optimize rosenborck function

        Arguments:
            start_point: Initialize point for optimizer

            optimizer: Class inherited from torch.optim.Optimzer

            iterations: Amount of iterations

        Returns:
            path: Array of points
    """

    xy = torch.tensor(start_point, requires_grad=True)
    optimizer = optim([xy], **optimizer_kwargs)

    path = np.empty((iterations + 1, 2))
    path[0] = start_point

    for iteration in range(1, iterations):
        optimizer.zero_grad()
        loss = rosenbrock(xy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy, 0.5)
        optimizer.step()

        path[iteration] = xy.data
    
    return path
