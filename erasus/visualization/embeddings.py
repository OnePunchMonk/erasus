"""
Visualization: Embeddings & Feature Space Analysis.

Extracts and visualizes high-dimensional representations from models
to analyze the impact of unlearning (e.g., feature collapse, cluster separation).
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    PCA = None
    TSNE = None


class EmbeddingVisualizer:
    """
    Handles feature extraction and dimensionality reduction for visualization.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(
        self, 
        dataloader: DataLoader, 
        layer_name: Optional[str] = None,
        modality: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from the model for a given dataloader.
        
        Returns:
            (embeddings, labels) as numpy arrays.
        """
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch structures
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    batch_labels = batch[1] if len(batch) > 1 else np.zeros(len(inputs))
                else:
                    inputs = batch
                    batch_labels = np.zeros(len(inputs))

                inputs = inputs.to(self.device) if isinstance(inputs, torch.Tensor) else inputs
                
                # Extract features based on model type
                feats = self._forward_features(inputs, modality)
                
                embeddings.append(feats.cpu().numpy())
                if isinstance(batch_labels, torch.Tensor):
                    labels.append(batch_labels.cpu().numpy())
                else:
                    labels.append(batch_labels)

        return np.concatenate(embeddings), np.concatenate(labels)

    def _forward_features(self, inputs: Any, modality: str) -> torch.Tensor:
        """
        Helper to extract features from various model architectures.
        Similar to MultimodalUnlearner._detect_type logic but for inference.
        """
        # 1. HuggingFace / Transformers models (BERT, ViT, etc.)
        if hasattr(self.model, "helpers") and hasattr(self.model, "get_image_features"):
             # CLIP-like
             if modality == "text":
                 return self.model.get_text_features(inputs)
             return self.model.get_image_features(inputs)

        # 2. Standard ResNet/VGG (torchvision)
        if hasattr(self.model, "fc") and isinstance(self.model, nn.Module):
             # Hook extraction or modifying forward?
             # Simple approach: forward pass until penultimate layer
             # But for unlearning, we often use the *output* of the forget layer.
             # Let's assume the model returns logits or embeddings.
             output = self.model(inputs)
             if isinstance(output, torch.Tensor):
                 if output.dim() > 1:
                     return output # Logits are embeddings too
                 return output
             if hasattr(output, "pooler_output"):
                 return output.pooler_output
             if hasattr(output, "last_hidden_state"):
                 # Average pooling for sequence
                 return output.last_hidden_state.mean(dim=1)
             if hasattr(output, "logits"):
                 return output.logits
        
        # 3. LLMs (CausalLM)
        if hasattr(self.model, "transformer") or hasattr(self.model, "model"):
            # Generative model
            # Inputs are likely token ids
            outputs = self.model(inputs, output_hidden_states=True)
            # Use last layer hidden state of last token
            last_hidden = outputs.hidden_states[-1] # [B, Seq, D]
            # [B, -1, D] -> last token embedding
            return last_hidden[:, -1, :]

        # Fallback: Just call forward
        out = self.model(inputs)
        return out if isinstance(out, torch.Tensor) else out[0]

    def plot(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "pca",
        title: str = "Embedding Space",
        save_path: Optional[str] = None,
    ):
        """
        Project and plot embeddings.
        """
        if PCA is None:
            print("sklearn not installed. Skipping plot.")
            return

        N, D = embeddings.shape
        if N < 2:
            print("Not enough points to plot.")
            return

        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, N - 1), init="pca", random_state=42)
        else:
            raise ValueError(f"Unknown method {method}")

        reduced = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels if labels is not None else "blue",
            cmap="viridis" if labels is not None else None,
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )
        
        if labels is not None:
             # Try to create a legendary legend
            pass # TODO: handle discrete vs continuous labels
            
        plt.title(f"{title} ({method.upper()})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
