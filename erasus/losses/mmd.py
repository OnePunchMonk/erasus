"""
Maximum Mean Discrepancy (MMD) Loss.

Measures the distance between two probability distributions based on samples.
Used in unlearning to minimize distance between Unlearned Model outputs and Retain Model outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """
    Computes MMD loss using a Gaussian kernel.
    L_mmd = || E[phi(x)] - E[phi(y)] ||^2
    """

    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = sigma

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        source : torch.Tensor
            Samples from distribution P (e.g., forget set activations). [B, D]
        target : torch.Tensor
            Samples from distribution Q (e.g., retain set activations). [B, D]
        """
        n = source.size(0)
        m = target.size(0)
        
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # Bandwidth selection
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n + m)**2
            
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = sum(kernel_val)
        
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]
        
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss
