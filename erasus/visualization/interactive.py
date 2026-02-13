"""
Visualization: Interactive Plots (Plotly/Dash).

Provides interactive versions of embeddings and loss curves for detailed exploration.
"""

from typing import Optional
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go = None
    px = None

def plot_interactive_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    hover_texts: Optional[np.ndarray] = None,
    title: str = "Interactive Embedding Space",
    save_path: Optional[str] = None
):
    """
    Generate an interactive scatter plot of embeddings.
    """
    if px is None:
        print("plotly not installed. Skipping interactive plot.")
        return

    # Check dimensions
    if embeddings.shape[1] > 3:
        # Use simple PCA
        from sklearn.decomposition import PCA
        embeddings = PCA(n_components=3).fit_transform(embeddings)

    dim = embeddings.shape[1]
    
    data = {
        "x": embeddings[:, 0],
        "y": embeddings[:, 1],
        "label": labels
    }
    
    if dim == 3:
        data["z"] = embeddings[:, 2]
        
    if hover_texts is not None:
        data["text"] = hover_texts
        
    if dim == 3:
        fig = px.scatter_3d(
            data, x='x', y='y', z='z', 
            color='label', 
            hover_data=['text'] if hover_texts is not None else None,
            title=title
        )
    else:
        fig = px.scatter(
            data, x='x', y='y', 
            color='label', 
            hover_data=['text'] if hover_texts is not None else None,
            title=title
        )
        
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_interactive_loss(
    forget_losses: list,
    retain_losses: list,
    title: str = "Interactive Loss Curves"
):
    if go is None:
        return

    fig = go.Figure()
    epochs = list(range(1, len(forget_losses) + 1))
    
    fig.add_trace(go.Scatter(x=epochs, y=forget_losses, mode='lines+markers', name='Forget Loss'))
    
    if retain_losses:
        fig.add_trace(go.Scatter(x=epochs, y=retain_losses, mode='lines+markers', name='Retain Loss'))
        
    fig.update_layout(title=title, xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()
