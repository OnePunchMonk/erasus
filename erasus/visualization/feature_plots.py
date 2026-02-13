"""
Visualization: Feature Plots (t-SNE, PCA).

Visualizes high-dimensional embeddings in 2D to assess unlearning efficacy.
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    PCA = None
    TSNE = None


def plot_embeddings(
    embeddings: np.ndarray, 
    labels: Optional[np.ndarray] = None, 
    method: str = "pca", 
    title: str = "Feature Space", 
    save_path: Optional[str] = None
):
    """
    Project embeddings to 2D using PCA or t-SNE and plot.
    
    Parameters
    ----------
    embeddings : np.ndarray
        [N, D] array of features.
    labels : np.ndarray, optional
        [N] array of class labels or 'Forget'/'Retain' indicators.
    method : str
        'pca' or 'tsne'.
    """
    if PCA is None:
        print("scikit-learn not installed. Skipping plot_embeddings.")
        return

    N, D = embeddings.shape
    if N < 2:
        print("Not enough points to plot.")
        return

    reduced_emb = None
    
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        reduced_emb = reducer.fit_transform(embeddings)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, perplexity=min(30, N-1), init="pca")
        reduced_emb = reducer.fit_transform(embeddings)
    else:
        # Fallback to just first 2 dims
        reduced_emb = embeddings[:, :2]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_emb[:, 0], 
        reduced_emb[:, 1], 
        c=labels if labels is not None else 'blue', 
        cmap='tab10' if labels is not None else None, 
        alpha=0.6,
        s=10
    )
    
    if labels is not None:
        plt.legend(*scatter.legend_elements(), title="Classes")
        
    plt.title(f"{title} ({method.upper()})")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
