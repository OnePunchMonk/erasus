"""
Representer Point Selector.

Based on "Representer Point Selection for Explaining Deep Neural Networks" (Yeh et al., NeurIPS 2018).
Calculates influence scores: alpha_i = (dL/dF_test) * (dF_i/dW).
This implementation approximates the representer value using the dot product of gradients 
(similar to linear influence).
"""
from __future__ import annotations
from typing import Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry

@selector_registry.register("representer")
class RepresenterSelector(BaseSelector):
    """
    Selects training points with large positive representer values to a specific target 
    (or self-representer value for self-influence).
    """

    def select(
        self, 
        model: nn.Module, 
        data_loader: DataLoader, 
        k: int, 
        target_loader: DataLoader = None,
        **kwargs
    ) -> List[int]:
        
        device = next(model.parameters()).device
        model.eval()
        scores = []
        
        # If no target, we calculate 'Self-Representer' value (contribution to own prediction).
        # Wrapper around generic influence logic.
        
        # 1. Capture Target Gradient/Embedding
        # For Representer Theorem: Pre-activation features f_i are key.
        # w* = sum alpha_i f_i
        # Phi(x_test) = sum alpha_i K(x_i, x_test)
        
        # Simplified: Use TracIn approximation (grad dot product).
        # We enforce explicit implementation here rather than delegation to avoid dependency loops.
        
        # If target provided, compute target grad
        target_vec = None
        if target_loader:
             model.zero_grad()
             for batch in target_loader:
                 inputs = batch[0].to(device)
                 labels = batch[1].to(device) if len(batch) > 1 else None
                 loss = nn.functional.cross_entropy(model(inputs), labels)
                 loss.backward()
             
             target_grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
             if target_grads:
                 target_vec = torch.cat(target_grads)
        
        # 2. Iterate Training Data
        for batch in data_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else None
            
            # Per-sample gradient
            batch_size = inputs.size(0)
            for b in range(batch_size):
                model.zero_grad()
                out = model(inputs[b:b+1])
                # Logistic loss or CrossEntropy
                l = nn.functional.cross_entropy(out, labels[b:b+1]) if labels is not None else out.sum()
                
                # Grads
                params = [p for p in model.parameters() if p.requires_grad]
                g = torch.autograd.grad(l, params)
                flat_g = torch.cat([gi.flatten() for gi in g])
                
                if target_vec is not None:
                    # Influence on target
                    score = torch.dot(flat_g, target_vec).item()
                else:
                    # Self influence (Norm squared)
                    score = torch.norm(flat_g, p=2).item() ** 2
                    
                scores.append(score)
                
        n = len(scores)
        k = min(k, n)
        top_k = np.argsort(scores)[-k:].tolist()
        return [int(i) for i in top_k]
