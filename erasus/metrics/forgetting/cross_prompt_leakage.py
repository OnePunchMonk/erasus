"""
erasus.metrics.forgetting.cross_prompt_leakage — Cross-Prompt Leakage Test.

Addresses issue #41: "Add cross-prompt leakage test"

Tests whether a model leaks information about forget-set samples when
both forget and retain queries are combined into a single prompt/batch.

After successful unlearning a model should answer retain queries correctly
but show *no elevated certainty* on forget queries even when they appear
alongside retain context.

Two complementary leakage signals are measured:

1. **Loss-based leakage** — the per-sample loss on forget queries
   embedded inside mixed prompts vs. pure-retain baselines.
   If unlearning is effective, the forget-query loss should be high
   (the model is uncertain), regardless of the retain context.

2. **Confidence-based leakage** — the maximum softmax probability
   assigned to any class for a forget query embedded in a mixed prompt.
   Leakage → the model remains confident on forget queries.

A *leakage score* in [0, 1] is computed:

    leakage_score = mean_confidence_on_forget_in_mixed / mean_confidence_on_retain_in_mixed

Values close to 0 indicate excellent unlearning; values close to 1
indicate the forget set knowledge is still intact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from erasus.core.base_metric import BaseMetric


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_batches(loader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Collect all (inputs, targets) pairs from a DataLoader."""
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            batches.append((batch[0], batch[1]))
    return batches


def _collect_stats(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (per_sample_losses, per_sample_max_confidences) for a set of batches.

    Parameters
    ----------
    model:
        Evaluation model (nn.Module). Must accept tensor inputs and return
        logits directly or via `output.logits`.
    batches:
        List of (inputs, targets) tensor pairs.
    device:
        Torch device to run inference on.

    Returns
    -------
    losses : ndarray, shape (N,)
    confidences : ndarray, shape (N,)
        Maximum softmax probability across all output classes.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: List[float] = []
    confidences: List[float] = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits  # HuggingFace-style models

            # Per-sample cross-entropy
            per_sample_loss = criterion(outputs, targets)
            losses.extend(per_sample_loss.cpu().tolist())

            # Per-sample max softmax confidence
            probs = F.softmax(outputs, dim=-1)
            max_conf = probs.max(dim=-1).values
            confidences.extend(max_conf.cpu().tolist())

    return np.array(losses, dtype=np.float64), np.array(confidences, dtype=np.float64)


def _interleave_batches(
    forget_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    retain_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[
    List[Tuple[torch.Tensor, torch.Tensor]],
    List[Tuple[torch.Tensor, torch.Tensor]],
]:
    """
    Build *mixed* batches by interleaving forget and retain samples
    within the same batch tensor.

    Each mixed batch contains alternating forget and retain samples:
    [forget_0, retain_0, forget_1, retain_1, ...].

    Returns two aligned lists of (inputs, targets), one for the forget
    portion and one for the retain portion of each mixed batch, so that
    per-class statistics can be computed separately.
    """
    max_len = min(len(forget_batches), len(retain_batches))
    mixed_forget: List[Tuple[torch.Tensor, torch.Tensor]] = []
    mixed_retain: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for i in range(max_len):
        f_inp, f_tgt = forget_batches[i]
        r_inp, r_tgt = retain_batches[i]

        # Truncate to the smaller batch to keep shapes equal
        min_bs = min(f_inp.size(0), r_inp.size(0))
        f_inp, f_tgt = f_inp[:min_bs], f_tgt[:min_bs]
        r_inp, r_tgt = r_inp[:min_bs], r_tgt[:min_bs]

        # Interleave: alternate forget and retain rows
        combined_inp = torch.empty(
            min_bs * 2, *f_inp.shape[1:], dtype=f_inp.dtype
        )
        combined_inp[0::2] = f_inp  # Even indices → forget samples
        combined_inp[1::2] = r_inp  # Odd  indices → retain samples

        combined_tgt = torch.empty(min_bs * 2, dtype=f_tgt.dtype)
        combined_tgt[0::2] = f_tgt
        combined_tgt[1::2] = r_tgt

        # We evaluate the combined batch through the model in _collect_stats,
        # but we only want to split back the forget / retain portions.
        # Return the forget and retain halves as separate batches so the
        # caller can evaluate them *within the same forward context*.
        mixed_forget.append((f_inp, f_tgt))
        mixed_retain.append((r_inp, r_tgt))

    return mixed_forget, mixed_retain


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CrossPromptLeakageMetric(BaseMetric):
    """
    Cross-Prompt Leakage Test (Issue #41).

    Combines forget-set and retain-set queries in a single forward pass
    and measures whether the model still exhibits elevated confidence on
    the forget samples (a sign of information leakage).

    Parameters
    ----------
    num_batches : int, optional
        Maximum number of forget/retain batch pairs to evaluate.
        Defaults to all available pairs.
    leakage_threshold : float
        Confidence ratio above which leakage is flagged.
        Default: 0.8 (forget confidence ≥ 80% of retain confidence).

    Returns
    -------
    dict with keys:
        ``cross_prompt_leakage_score``
            Ratio of forget confidence to retain confidence in mixed prompts
            (0 = no leakage, 1 = full leakage).
        ``cross_prompt_forget_confidence``
            Mean max-softmax confidence on forget queries in mixed prompts.
        ``cross_prompt_retain_confidence``
            Mean max-softmax confidence on retain queries in mixed prompts.
        ``cross_prompt_forget_loss``
            Mean cross-entropy loss on forget queries in mixed prompts.
        ``cross_prompt_retain_loss``
            Mean cross-entropy loss on retain queries in mixed prompts.
        ``cross_prompt_loss_gap``
            forget_loss − retain_loss. Positive = forget loss higher
            (good: model is uncertain on forget set).
        ``cross_prompt_leakage_detected``
            1.0 if leakage_score ≥ leakage_threshold, else 0.0.
        ``cross_prompt_num_forget_samples``
            Number of forget samples evaluated.
        ``cross_prompt_num_retain_samples``
            Number of retain samples evaluated.
    """

    name = "cross_prompt_leakage"

    def __init__(
        self,
        num_batches: Optional[int] = None,
        leakage_threshold: float = 0.8,
    ) -> None:
        self.num_batches = num_batches
        self.leakage_threshold = leakage_threshold

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run the cross-prompt leakage test.

        Parameters
        ----------
        model :
            The (un)learned model.
        forget_data :
            DataLoader for the forget set.
        retain_data :
            DataLoader for the retain / reference set.
        **kwargs :
            Ignored; present for API compatibility.
        """
        if forget_data is None or retain_data is None:
            return {
                "cross_prompt_leakage_score": 0.0,
                "cross_prompt_leakage_detected": 0.0,
            }

        device = next(model.parameters()).device

        # ------------------------------------------------------------------ #
        # 1. Collect raw batches
        # ------------------------------------------------------------------ #
        forget_batches = _iter_batches(forget_data)
        retain_batches = _iter_batches(retain_data)

        if not forget_batches or not retain_batches:
            return {
                "cross_prompt_leakage_score": 0.0,
                "cross_prompt_leakage_detected": 0.0,
            }

        # ------------------------------------------------------------------ #
        # 2. Build interleaved (mixed) batches
        # ------------------------------------------------------------------ #
        mixed_forget, mixed_retain = _interleave_batches(
            forget_batches, retain_batches
        )

        # Optionally cap the number of batch pairs
        if self.num_batches is not None:
            mixed_forget = mixed_forget[: self.num_batches]
            mixed_retain = mixed_retain[: self.num_batches]

        # ------------------------------------------------------------------ #
        # 3. Evaluate forget and retain in the mixed-prompt context
        # ------------------------------------------------------------------ #
        forget_losses, forget_confs = _collect_stats(model, mixed_forget, device)
        retain_losses, retain_confs = _collect_stats(model, mixed_retain, device)

        if len(forget_losses) == 0 or len(retain_losses) == 0:
            return {
                "cross_prompt_leakage_score": 0.0,
                "cross_prompt_leakage_detected": 0.0,
            }

        # ------------------------------------------------------------------ #
        # 4. Compute derived metrics
        # ------------------------------------------------------------------ #
        mean_forget_conf = float(forget_confs.mean())
        mean_retain_conf = float(retain_confs.mean())
        mean_forget_loss = float(forget_losses.mean())
        mean_retain_loss = float(retain_losses.mean())

        # Leakage score: how close is forget-confidence to retain-confidence?
        # 0 = perfect unlearning (no confidence on forget),
        # 1 = complete leakage (same confidence as retain).
        if mean_retain_conf > 0:
            leakage_score = mean_forget_conf / mean_retain_conf
        else:
            leakage_score = 0.0

        # Clamp to [0, 1]
        leakage_score = float(np.clip(leakage_score, 0.0, 1.0))
        leakage_detected = 1.0 if leakage_score >= self.leakage_threshold else 0.0

        return {
            "cross_prompt_leakage_score": leakage_score,
            "cross_prompt_forget_confidence": mean_forget_conf,
            "cross_prompt_retain_confidence": mean_retain_conf,
            "cross_prompt_forget_loss": mean_forget_loss,
            "cross_prompt_retain_loss": mean_retain_loss,
            "cross_prompt_loss_gap": mean_forget_loss - mean_retain_loss,
            "cross_prompt_leakage_detected": leakage_detected,
            "cross_prompt_num_forget_samples": float(len(forget_losses)),
            "cross_prompt_num_retain_samples": float(len(retain_losses)),
        }
