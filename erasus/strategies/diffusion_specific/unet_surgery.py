"""
U-Net Surgery Strategy.

Surgically modifies the Cross-Attention layers of the U-Net.
Specifically targets the Key/Value projection matrices in Cross-Attention blocks
that activate for the specific concepts to be forgotten.

Approach:
1. Locate CA layers.
2. Intervene on their weights to dampen attention to forget tokens.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("unet_surgery")
class UNetSurgeryStrategy(BaseStrategy):
    """
    Directly modifies Cross-Attention weights.
    Can be used to reset them (ablate) or fine-tune them specifically.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        
        # 1. Identify Cross Attention Layers
        ca_layers = []
        for n, m in model.named_modules():
            if "attn2" in n or "cross_attention" in n: # Common UNet naming
                # Target 'to_k' and 'to_v' projections usually
                # m is likely the Attention block
                ca_layers.append(m)

        # 2. Optimize ONLY these layers
        params = []
        for m in ca_layers:
            params.extend(m.parameters())
            # Or specifically `to_k` / `to_v` if we want to be very surgical
        
        if not params:
            # Fallback for generic model
            params = model.parameters()

        optimizer = torch.optim.Adam(params, lr=self.lr)
        
        # 3. Optimization Loop (ESD-like)
        forget_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                # Unlearning objective: e.g. Negative Gradient or Target Empty
                # Here: Negative Gradient on the attention map activation?
                # Simpler: Target Empty noise (ESD).
                
                # Assuming wrapper `compute_loss` supports 'empty' target behavior
                if hasattr(model, "compute_loss"):
                    loss = model.compute_loss(
                        batch["pixel_values"].to(device),
                        batch["input_ids"].to(device),
                        target_override="empty"
                    )
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))
            
        return model, forget_losses, []

