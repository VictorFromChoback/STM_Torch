from ._types import TensorType

import torch
import numpy as np


def degenerate_nesterov(x : TensorType, Lipsitz: float) -> TensorType:
    """ Degenerate Nestrov function 

        Arguments:
            x: torch.Tensor - argument
            Lipsitz: float - Lipsitz constant

        Return:
            f(x)
    """
    
    dim = x.shape[0]
    k = (dim - 1) // 2

    answer = x[0] ** 2

    for coordinat in range(2 * k):
        answer += (x[coordinat] - x[coordinat + 1]) ** 2

    answer += x[2 * k] ** 2
    answer *= Lipsitz / 8
    answer -= Lipsitz / 4 * x[0]

    return answer