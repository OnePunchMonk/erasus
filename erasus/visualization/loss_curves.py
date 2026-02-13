"""
Visualization: Loss Curves.

Plots training dynamics of Forget and Retain losses over epochs.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    forget_losses: List[float], 
    retain_losses: Optional[List[float]] = None, 
    title: str = "Unlearning Dynamics", 
    save_path: Optional[str] = None
):
    """
    Plot Forget and Retain loss curves.
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(forget_losses) + 1)
    
    plt.plot(epochs, forget_losses, 'r-o', label="Forget Loss", linewidth=2)
    
    if retain_losses and len(retain_losses) > 0:
        # Ensure lengths match
        retain_losses = retain_losses[:len(epochs)]
        plt.plot(epochs, retain_losses, 'b-s', label="Retain Loss", linewidth=2)
        
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
