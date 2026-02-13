"""
erasus.losses.triplet_loss — Triplet-based separation loss.

Pushes forget embeddings away from retain embeddings in the
representation space while maintaining retain-set cohesion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletUnlearningLoss(nn.Module):
    """
    Triplet loss adapted for unlearning.

    - Anchor: retain-set embedding
    - Positive: other retain-set embedding
    - Negative: forget-set embedding (to be pushed away)

    Parameters
    ----------
    margin : float
        Minimum distance between retain and forget embeddings.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        retain_embeddings: torch.Tensor,
        forget_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet unlearning loss.

        Parameters
        ----------
        retain_embeddings : (N, D)
            Embeddings from retain data.
        forget_embeddings : (M, D)
            Embeddings from forget data.
        """
        n_retain = retain_embeddings.size(0)

        if n_retain < 2:
            return torch.tensor(0.0, device=retain_embeddings.device)

        # Anchor = first half of retain, Positive = second half
        half = n_retain // 2
        anchors = retain_embeddings[:half]
        positives = retain_embeddings[half: half + half]

        # Negative = forget embeddings (cycle if needed)
        n_triplets = min(len(anchors), len(positives), len(forget_embeddings))
        anchors = anchors[:n_triplets]
        positives = positives[:n_triplets]
        negatives = forget_embeddings[:n_triplets]

        # distances
        d_pos = F.pairwise_distance(anchors, positives)
        d_neg = F.pairwise_distance(anchors, negatives)

        # We want forget (negatives) to be far, retain (positives) to be close
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()
