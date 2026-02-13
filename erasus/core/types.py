"""
Erasus Type Aliases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Common type aliases used across the framework
ModelType = nn.Module
LossHistory = List[float]
MetricDict = Dict[str, float]
BatchType = Union[Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]]
DeviceType = Union[str, torch.device]
