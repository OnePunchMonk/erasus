"""
Visualization: Membership Inference Attack (MIA) Plots.

Visualizes privacy risks by comparing score distributions of Forget vs. Retain sets.
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    roc_curve = None
    auc = None


def plot_mia_histogram(
    forget_scores: np.ndarray, 
    retain_scores: np.ndarray, 
    test_scores: Optional[np.ndarray] = None,
    bins: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot histograms of MIA scores (e.g., loss or entropy) for different splits.
    Significant overlap => Better Privacy (Harder to distinguish).
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(forget_scores, bins=bins, alpha=0.5, label="Forget Set", density=True, color='red')
    plt.hist(retain_scores, bins=bins, alpha=0.5, label="Retain Set", density=True, color='blue')
    
    if test_scores is not None:
        plt.hist(test_scores, bins=bins, alpha=0.5, label="Test Set", density=True, color='green')
        
    plt.title("Membership Inference Score Distribution")
    plt.xlabel("Score (e.g. Loss)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_mia_roc(
    forget_scores: np.ndarray,
    test_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve for distinguishing Forget Set from Test Set (Non-Member).
    Closer to diagonal (AUC=0.5) => Better Unlearning Privacy.
    """
    if roc_curve is None:
        print("sklearn not installed. Skipping ROC plot.")
        return

    # Label 1: Member (Forget Set), Label 0: Non-Member (Test Set)
    y_true = np.concatenate([np.ones_like(forget_scores), np.zeros_like(test_scores)])
    # Scores: Assume lower score (e.g. lower loss) => More likely member
    # So we use -score for ROC input if score is loss.
    # We assume 'score' here is 'likelihood of membership'. 
    # If passed raw losses, caller should negate them or we handle convention.
    # Let's assume input is raw loss: Lower loss = Member.
    # So predictor score = -loss.
    
    y_scores = np.concatenate([-forget_scores, -test_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Membership Inference ROC (Forget vs Test)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
