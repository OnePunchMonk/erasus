"""
Differential Privacy Mechanisms.

Tools for adding noise to gradients or weights.
"""

import torch
import numpy as np

class GaussianMechanism:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.sigma

class LaplacianMechanism:
    def __init__(self, b: float):
        self.b = b

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        # Pytorch doesn't have direct laplace sample_like?
        # Use simple approximation or numpy
        noise = torch.as_tensor(
            np.random.laplace(loc=0.0, scale=self.b, size=tensor.shape),
            device=tensor.device, 
            dtype=tensor.dtype
        )
        return tensor + noise
