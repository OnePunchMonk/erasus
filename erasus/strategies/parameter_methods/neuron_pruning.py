"""
Neuron Pruning Strategy.

Identifies specific neurons that are disproportionately active for the
forget set and virtually prunes them (zeroes weights) to remove the
associated features.

Similar to `SelectiveSynapticDampening` but effectively a binary removal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("neuron_pruning")
class NeuronPruningStrategy(BaseStrategy):
    """
    1. Hook activations.
    2. Feed Forget Set -> Record mean activation per neuron.
    3. Feed Retain Set (optional) -> Record mean activation per neuron.
    4. Score neurons: High in Forget, Low in Retain (if available).
    5. Prune Top-K neurons (set weights to 0).
    """

    def __init__(
        self,
        pruning_ratio: float = 0.05,
        layer_types: Tuple[type, ...] = (nn.Linear,),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pruning_ratio = pruning_ratio
        self.layer_types = layer_types

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 1, # Not iterative, but one-shot
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        model.eval()
        
        # Store accumulated activations
        activations: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        
        def get_hook(name):
            def hook(module, inp, out):
                # out: [batch, ..., neurons]
                # Average over batch and spatial dims
                if isinstance(out, tuple): 
                    out = out[0]
                
                dims = list(range(len(out.shape) - 1)) # All except last logic dimension
                mean_act = out.detach().mean(dim=dims)
                
                if name not in activations:
                    activations[name] = torch.zeros_like(mean_act)
                    counts[name] = 0
                
                activations[name] += mean_act
                counts[name] += 1
            return hook

        hooks = []
        target_layers = []
        for n, m in model.named_modules():
            if isinstance(m, self.layer_types):
                hooks.append(m.register_forward_hook(get_hook(n)))
                target_layers.append((n, m))

        # Pass 1: Forget Set
        # If we had retain set, we would do weighted difference.
        # Here naive implementation: Prune highly active in forget.
        
        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device)
                model(inputs)
        
        # Cleanup hooks
        for h in hooks:
            h.remove()
            
        # Compute scores and prune
        pruned_count = 0
        total_neurons = 0
        
        for name, module in target_layers:
            if name not in activations:
                continue
                
            mean_act = activations[name] / max(counts[name], 1)
            n_neurons = mean_act.numel()
            total_neurons += n_neurons
            
            # Determine threshold for this layer
            k = int(n_neurons * self.pruning_ratio)
            if k == 0:
                continue
                
            # Find indices of top-k activations
            _, indices = torch.topk(mean_act, k)
            
            # Prune weights connected to these neurons
            # For Linear: [out_features, in_features]
            # Pruning output neuron i => row i = 0 & bias i = 0
            
            with torch.no_grad():
                module.weight.data[indices] = 0.0
                if module.bias is not None:
                    module.bias.data[indices] = 0.0
            
            pruned_count += k

        return model, [], []

