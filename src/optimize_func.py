from .degenerate_nesterov import degenerate_nesterov
from .rosenbrock import rosenbrock
from ._types import TensorType

import torch

from collections.abc import Callable

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def optimize(func, start_point, iterations, optim, L, **optimizer_kwargs):
    """ Optimize function and return path

        Arguments:
            func - Function calling func(x, **kwargs)
            start_point - Point to start 
            iterations - Number of iteration
            optim - Optimizer algorithm
            L - Lipsitz constant
        
        Return:
            path : Path of iterations
    """

    point = torch.tensor(start_point, requires_grad=True)
    optimizer = optim([point], **optimizer_kwargs)

    path = []
    path.append(point.data)

    for i in range(iterations):
        optimizer.zero_grad()
        loss = func(point, L)
        loss.backward()
        optimizer.step()
        path.append(point.data)

    return path


def plot_path(path, func, min_value, **func_kwargs) -> None:
    """ Plot path using path of points and func
    """

    values = [func(point, **func_kwargs) for point in path]
    grid = np.arange(len(values))

    min_value = float(min_value)
    min_values = min_value * grid

    with sns.plotting_context('notebook'), sns.axes_style('darkgrid'):
        
        plt.figure(figsize=(10, 8), dpi=130, facecolor='whitesmoke')
        plt.plot(grid, values, color='blue', label='Trace')
        plt.hlines(min_value, grid[0], grid[-1], color='red', linestyles='dashed', label='Min value')
        plt.xlabel('Iteration - n', fontsize=13)
        plt.ylabel('f(x)', fontsize=13)
        plt.title('Optimzation path', fontsize=13)
        plt.show()
    
