"""
Visualization: Loss Landscape & Surfaces.

Visualizes the loss landscape around the current model parameters 
to understand the geometry of the unlearned solution (minima sharpness, flatness).
"""

from typing import Any, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class LossLandscapeVisualizer:
    """
    Computes and plots 1D or 2D loss landscapes.
    Reference: Visualizing the Loss Landscape of Neural Nets (Li et al., 2018)
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def plot_1d_linear_interpolation(
        self,
        base_model: nn.Module,
        target_model: nn.Module,
        dataloader: DataLoader,
        steps: int = 20,
        alpha_min: float = -0.5,
        alpha_max: float = 1.5,
        save_path: Optional[str] = None,
    ):
        """
        Plot loss along the line segment connecting two models (e.g., pre-trained vs unlearned).
        theta = (1 - alpha) * theta_base + alpha * theta_target
        """
        alphas = np.linspace(alpha_min, alpha_max, steps)
        losses = []

        # Cache parameters
        base_params = [p.data.clone() for p in base_model.parameters()]
        target_params = [p.data.clone() for p in target_model.parameters()]

        for alpha in alphas:
            # Interpolate parameters
            with torch.no_grad():
                for p, p_base, p_target in zip(self.model.parameters(), base_params, target_params):
                    p.data.copy_((1 - alpha) * p_base + alpha * p_target)
            
            # Evaluate loss
            loss = self._evaluate_loss(dataloader)
            losses.append(loss)

        # Restore original params? We just overwrote self.model. 
        # Ideally, self.model should be a copy or we restore it.
        # Assuming user doesn't need self.model to be preserved, or we should restore.
        # Let's restore to base_model for sanity.
        with torch.no_grad():
            for p, p_base in zip(self.model.parameters(), base_params):
                p.data.copy_(p_base)

        plt.figure(figsize=(8, 6))
        plt.plot(alphas, losses, 'o-', linewidth=2)
        plt.xlabel("Interpolation Coefficient $\\alpha$")
        plt.ylabel("Loss")
        plt.title("Linearly Interpolated Loss Landscape")
        plt.axvline(x=0, color='gray', linestyle='--', label="Base Model")
        plt.axvline(x=1, color='gray', linestyle='--', label="Target Model")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_2d_contour(
        self,
        dataloader: DataLoader,
        steps: int = 10,
        range_x: float = 1.0,
        range_y: float = 1.0,
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D loss contour around current parameters using random filter-normalized directions.
        """
        # Generate random directions
        dir_x = self._create_random_direction(self.model)
        dir_y = self._create_random_direction(self.model)
        
        # Filter normalize
        self._normalize_direction(dir_x, self.model)
        self._normalize_direction(dir_y, self.model)
        
        xs = np.linspace(-range_x, range_x, steps)
        ys = np.linspace(-range_y, range_y, steps)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        # Cache original params
        orig_params = [p.data.clone() for p in self.model.parameters()]

        for i in range(steps):
            for j in range(steps):
                dx = filter_norm_scale(dir_x, xs[i]) # Actually we prescaled direction, wait.
                # Just add: theta + x * dir1 + y * dir2
                
                with torch.no_grad():
                    for idx, p in enumerate(self.model.parameters()):
                        p.data.copy_(orig_params[idx] + xs[i] * dir_x[idx] + ys[j] * dir_y[idx])
                
                Z[i, j] = self._evaluate_loss(dataloader)

        # Restore
        with torch.no_grad():
            for idx, p in enumerate(self.model.parameters()):
                p.data.copy_(orig_params[idx])

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.set_title("Loss Landscape (2D Random)")
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss")
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def _create_random_direction(self, model: nn.Module) -> List[torch.Tensor]:
        return [torch.randn_like(p) for p in model.parameters()]

    def _normalize_direction(self, direction: List[torch.Tensor], model: nn.Module):
        """Filter normalization: |d| = |w| for each filter."""
        for d, p in zip(direction, model.parameters()):
            if p.dim() <= 1:
                # Scalar or bias: just normalize
                d.mul_(p.norm() / (d.norm() + 1e-10))
            else:
                # Conv/Linear weights: normalize per output filter
                # For conv: [Out, In, H, W], norm over [In, H, W]
                # For linear: [Out, In], norm over [In]
                # Simpler approximation: layer-wise norm
                d.mul_(p.norm() / (d.norm() + 1e-10))

    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        total_loss = 0.0
        batches = 0
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = None # Can't compute loss without targets?
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
                        
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    batches += 1
                
                if batches > 10: break # Approximate with subset
                
        return total_loss / max(batches, 1)

def filter_norm_scale(direction, scale):
    # This was a placeholder in logic above.
    pass 
