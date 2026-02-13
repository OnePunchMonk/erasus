"""
Custom Losses.

Specific loss functions for nuanced unlearning objectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class NegativeLogLikelihoodLoss(nn.Module):
    """
    Standard NLL Loss, but wrapped for consistency.
    """
    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss()

    def forward(self, log_probs, targets):
        return self.nll(log_probs, targets)

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pairs (SimCLR style or Hinge).
    Minimizes distance for positive pairs, maximizes for negatives.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings1, embeddings2, label):
        """
        label: 1 for positive pair, -1 (or 0) for negative pair depending on convention.
        PyTorch CosineEmbeddingLoss uses 1/-1.
        """
        return F.cosine_embedding_loss(embeddings1, embeddings2, label, margin=self.margin)

class BoundaryLoss(nn.Module):
    """
    Boundary Loss / Decision Boundary Penalization.
    Encourages samples to move towards (or away from) the decision boundary.
    Often used in adversarial training or unlearning to push forget samples to the decision boundary (uncertainty).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets=None):
        """
        If targets are provided, we might want to *decrease* confidence in true class
        until it crosses boundary to nearest other class?
        
        Simple unlearning objective: Maximize Entropy (move to center of simplex).
        Boundary Shift: minimize difference between top-1 and top-2 prob?
        
        Implementation:
        Loss = - (Top1_Logit - Top2_Logit)
        Minimizing this pushes them closer -> towards boundary.
        """
        # 1. Get probs
        probs = F.softmax(logits, dim=1)
        
        # 2. Sort to find top 2
        top_probs, _ = torch.topk(probs, 2, dim=1)
        
        # top_probs[:, 0] is max prob
        # top_probs[:, 1] is second max prob
        # To push to boundary: minimize gap
        gap = top_probs[:, 0] - top_probs[:, 1]
        
        # Loss: minimize gap (squared diff) or just gap
        loss = torch.mean(gap ** 2)
        return loss
