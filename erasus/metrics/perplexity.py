"""
Perplexity Metric.

Evaluates how 'confused' the model is on a given dataset.
Lower is better for Retain set. Higher is potentially better for Forget set (if goal is forgetting).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class PerplexityMetric(BaseMetric):
    """Computes perplexity on a dataset."""

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        
        results = {}
        # Ensure model is on correct device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model might be empty or wrapped strangely
            device = torch.device("cpu")
            
        model.eval()

        if forget_data:
            results["forget_perplexity"] = self._compute_ppl(model, forget_data, device)
            
        if retain_data:
            results["retain_perplexity"] = self._compute_ppl(model, retain_data, device)
            
        return results

    def _compute_ppl(self, model: nn.Module, loader: DataLoader, device: torch.device) -> float:
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in loader:
                # Handle different batch structures
                if isinstance(batch, torch.Tensor):
                    input_ids = batch.to(device)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(device)
                elif isinstance(batch, dict):
                    input_ids = batch.get("input_ids").to(device)
                else:
                    continue # Unknown format

                # Standard causal language modeling forward pass
                # We assume model returns CausalLMOutput or object with .loss
                try:
                    outputs = model(input_ids, labels=input_ids)
                    if hasattr(outputs, "loss"):
                        loss = outputs.loss
                    else:
                        # Fallback: manually compute if logits returned
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        # Shift logits and labels for causal LM
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1)
                        )
                except Exception as e:
                    # Model might not accept labels argument or other mismatch
                    # For stability, we log and return inf or 0
                    print(f"Warning: Perplexity calc failed: {e}")
                    return 0.0
                
                # Weighted average calculation
                num_valid_tokens = input_ids.numel() # approximation, normally ignore padding
                total_loss += loss.item() * num_valid_tokens
                total_tokens += num_valid_tokens
                
        if total_tokens == 0:
            return 0.0
            
        avg_loss = total_loss / total_tokens
        try:
            ppl = torch.exp(torch.tensor(avg_loss)).item()
        except OverflowError:
            ppl = float('inf')
            
        return ppl
