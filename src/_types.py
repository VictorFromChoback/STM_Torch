from typing import Dict, Iterable, Optional, Union, Any, Callable, Tuple
from torch import Tensor
import torch

# Type for algo params
Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

# Type for loss closure
LossClosure = Optional[Callable[[], float]]

# Type for step result
StepResult = Optional[float]

# Tuple of float
TwoFloats = Tuple[float, float]

# Tensor type
TensorType = torch.Tensor