"""
EL2N Selector (Gradient-Based / Error-Based).

EL2N: Error at L2 Norm.
Identifies important examples early in training by looking at the 
L2 norm of the error vector (probability margin).
Paper: "Deep Learning on a Data Diet: Finding Important Examples Early" (Paul et al., NeurIPS 2021).
Section 3.1.4.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_selector import BaseSelector
from erasus.core.exceptions import SelectorError
from erasus.core.registry import selector_registry


@selector_registry.register("el2n")
class EL2NSelector(BaseSelector):
    """
    Selects samples with high EL2N score.
    Score = || p(y|x) - y_onehot ||_2
    
    Proxy for "difficulty" or influence.
    """

    def select(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        k: int,
        **kwargs: Any,
    ) -> List[int]:
        
        device = next(model.parameters()).device
        model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else None
                
                if labels is None:
                    raise SelectorError(
                        "EL2NSelector requires labels (batch[1]) to compute the error vector. "
                        "Your DataLoader yielded unlabeled batches. Provide a dataset that yields "
                        "(inputs, labels) tuples or use a selector that does not require labels."
                    )
                
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=-1)
                
                # Construct one-hot targets
                num_classes = logits.size(-1)
                one_hot = torch.zeros_like(probs)
                one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
                
                # EL2N = Norm(probs - one_hot)
                error_vec = probs - one_hot
                batch_scores = torch.norm(error_vec, p=2, dim=1)
                
                scores.extend(batch_scores.cpu().tolist())
                
        n = len(scores)
        k = min(k, n)
        
        # High EL2N = High Error = Important/Hard example
        top_k_indices = np.argsort(scores)[-k:].tolist()
        return [int(i) for i in top_k_indices]
