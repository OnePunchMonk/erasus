"""
Visualization: Gradient Analysis & Flow.

Analyzes gradient magnitudes and distributions to detect vanishing/exploding gradients
or "gradient masking" during unlearning.
"""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class GradientVisualizer:
    """
    Visualizes gradient flow and magnitude distributions per layer.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def plot_gradient_flow(self, dataloader: DataLoader, save_path: Optional[str] = None):
        """
        Plots the average gradient magnitude per layer for a batch.
        Useful for checking if gradients are propagating correctly.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Get one batch
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return

        if isinstance(batch, (list, tuple)):
             inputs, targets = batch[0], batch[1]
        else:
             return # Need targets

        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward + Backward
        outputs = self.model(inputs)
        if hasattr(outputs, "logits"): outputs = outputs.logits
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()

        ave_grads = []
        max_grads = []
        layers = []

        for n, p in self.model.named_parameters():
            if p.requires_grad and ("bias" not in n) and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical", fontsize=6)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1 if max_grads else 0.02)
        plt.xlabel("Layers")
        plt.ylabel("Gradient Magnitude (Avg/Max)")
        plt.title("Gradient Flow")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
            
        self.model.zero_grad()

    def plot_weight_distribution(self, save_path: Optional[str] = None):
        """
        Plots histograms of weights for each layer group.
        """
        plt.figure(figsize=(10, 6))
        
        all_weights = []
        for p in self.model.parameters():
            if p.requires_grad:
                all_weights.append(p.data.cpu().flatten().numpy())
                
        if not all_weights:
            return

        concatenated = np.concatenate(all_weights)
        plt.hist(concatenated, bins=100, log=True, density=True)
        plt.title("Weight Distribution (Log Scale)")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
