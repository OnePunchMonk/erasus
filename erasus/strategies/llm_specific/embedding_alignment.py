"""
Embedding Alignment Strategy.

Adjusts the embedding space of an LLM/VLM to push 'forget' concepts
away from their semantic neighbors or towards a null/generic concept.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


@strategy_registry.register("embedding_alignment")
class EmbeddingAlignmentStrategy(BaseStrategy):
    """
    Minimizes Cosine Similarity between Forget Inputs and their specific Representations,
    or pushes them towards a 'refusal' vector.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        margin: float = 0.5,
        target_layer: str = "model.embed_tokens", # Generalized name
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lr = lr
        self.margin = margin
        self.target_layer = target_layer

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        
        device = next(model.parameters()).device
        
        # We only optimize the embeddings? Or the whole model to emit different embeddings?
        # Typically alignment optimizes the whole model to change the output embedding/hidden state.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        forget_losses = []
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n = 0
            for batch in forget_loader:
                input_ids = batch[0].to(device)
                
                optimizer.zero_grad()
                
                # Hook the final hidden state or use the model's output_hidden_states
                outputs = model(input_ids, output_hidden_states=True)
                
                # Last layer hidden state: [batch, seq, dim]
                # We usually care about the last token embedding for classification/generation
                if hasattr(outputs, "hidden_states"):
                    last_hidden = outputs.hidden_states[-1]
                else:
                    # Fallback if model doesn't support it standardly
                    last_hidden = outputs.logits # Proxy
                
                # Representation of the sequence (e.g. mean or last token)
                representation = last_hidden.mean(dim=1)
                
                # Objective 1: Maximize distance to "original" representation?
                # or Objective 2: Minimize likelihood (GA) + regularization
                
                # Implementation: Negative Cosine Similarity to itself?
                # Push norm down?
                # Simple approach: Maximize Entropy of the embedding distribution? 
                
                # Let's Implement: Minimize Norm of the representation for forget inputs?
                # i.e. make it "meaningless" / "zero".
                loss = torch.norm(representation, p=2)
                
                # Note: We want to MINIMIZE this norm (dampen the signal) 
                # OR if we have a target "refusal" vector, minimize distance to that.
                # Here: Dampening.
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n += 1
            forget_losses.append(epoch_loss / max(n, 1))

        return model, forget_losses, []

