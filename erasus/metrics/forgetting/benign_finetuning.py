"""
erasus.metrics.forgetting.benign_finetuning — Benign fine-tuning attack metric.

Issue #46: Fine-tune the unlearned model on benign related data and measure
knowledge restoration on the forget set.

Background
----------
"Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign
Relearning" (ICLR 2025) shows that many unlearning methods merely obfuscate
forgotten knowledge rather than truly erasing it.  Fine-tuning the unlearned
model on a small set of *benign* (non-forget) data from the same broad domain
often restores forget-set performance — a strong signal that the model still
encodes the forgotten information at the representation level.

This metric wraps the attack as a ``BaseMetric``-compatible class so it can be
used directly in metric suites, CLI pipelines, and evaluation harnesses.

What is measured
----------------
- **Pre-attack baseline** — accuracy, loss, and max-class confidence on the
  forget set *before* any fine-tuning.
- **Per-epoch restoration** — after each fine-tuning epoch the same stats are
  re-evaluated on the forget set, producing a restoration curve.
- **Post-attack summary** — final accuracy, loss delta, confidence delta.
- **Restoration rate** — fraction of pre-unlearning accuracy recovered
  (requires ``original_accuracy`` kwarg or is approximated as post/1.0).
- **Restoration AUC** — area under the per-epoch accuracy curve (normalised to
  [0, 1]) — high AUC means knowledge recovered quickly, a worse sign.
- **Verdict** — ``passed=True`` iff ``forget_accuracy_recovery`` is below
  ``recovery_threshold``.

Usage
-----
>>> metric = BenignFinetuningMetric(epochs=5, lr=1e-3)
>>> result = metric.compute(
...     model=unlearned_model,
...     forget_data=forget_loader,
...     retain_data=retain_loader,        # used as benign fine-tuning data
...     benign_data=domain_loader,        # optional: separate domain data
... )
>>> print(result["benign_ft_forget_accuracy_recovery"])
>>> print(result["benign_ft_passed"])
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and extract logits from plain or HF-style output."""
    outputs = model(inputs)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """
    Compute accuracy, mean loss, and mean max-class confidence on *loader*.

    Parameters
    ----------
    model:
        Model in eval-mode (called inside ``torch.no_grad()``).
    loader:
        DataLoader producing ``(inputs, targets)`` batches.
    device:
        Device to run on.
    criterion:
        Loss function (reduction='sum' so we can average across samples).

    Returns
    -------
    dict with keys ``accuracy``, ``loss``, ``confidence``, ``n_samples``.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_conf = 0.0

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            logits = _forward_logits(model, inputs)

            total_loss += criterion(logits, targets).item()
            probs = torch.softmax(logits, dim=-1)
            total_conf += probs.max(dim=-1).values.sum().item()
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)

    n = max(total, 1)
    return {
        "accuracy": correct / n,
        "loss": total_loss / n,
        "confidence": total_conf / n,
        "n_samples": total,
    }


def _finetune_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """
    Fine-tune *model* for one epoch on *loader*.

    Parameters
    ----------
    model:
        Model to fine-tune (already set to train mode by the caller).
    loader:
        Fine-tuning data.
    optimizer:
        Pre-configured optimizer.
    criterion:
        Loss function (reduction='mean').
    device:
        Device.
    max_batches:
        Hard cap on number of batches processed per epoch.

    Returns
    -------
    Mean training loss for this epoch.
    """
    epoch_loss = 0.0
    n_batches = 0

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            continue

        inputs, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = _forward_logits(model, inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / max(n_batches, 1)


def _auc_trapezoid(values: List[float]) -> float:
    """
    Compute the AUC of a curve using the trapezoidal rule.

    The x-axis is ``[0, 1, ..., len(values)-1]`` normalised to ``[0, 1]``.
    The y-axis is normalised to ``[0, 1]`` using the range of *values*.

    Returns 0.0 for sequences of length < 2.
    """
    n = len(values)
    if n < 2:
        return float(values[0]) if n == 1 else 0.0

    x = np.linspace(0.0, 1.0, n)
    y = np.array(values, dtype=float)
    y_min, y_max = y.min(), y.max()
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min)
    else:
        y = np.zeros_like(y)

    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
# BenignFinetuningMetric
# ---------------------------------------------------------------------------

class BenignFinetuningMetric(BaseMetric):
    """
    Measure knowledge restoration after benign fine-tuning.

    Fine-tunes a *deep copy* of the unlearned model on benign data (either
    ``benign_data`` or ``retain_data``) and tracks how much forget-set
    performance recovers over training.  A large recovery indicates that the
    unlearning was obfuscation rather than true forgetting.

    Parameters
    ----------
    epochs : int
        Number of fine-tuning epochs (default 5).
    lr : float
        Adam learning rate for fine-tuning (default ``1e-3``).
    optimizer_cls : str
        Optimizer to use: ``"adam"`` (default) or ``"sgd"``.
    finetune_fraction : float
        Fraction of benign-data batches to use per epoch (default ``1.0``).
        Values < 1.0 cap the number of batches, simulating data scarcity.
    recovery_threshold : float
        Maximum acceptable absolute accuracy recovery before the test is
        considered failed (default ``0.15``, i.e. 15 pp).
    weight_decay : float
        L2 regularisation for the optimizer (default ``0.0``).

    Notes
    -----
    The model is **never mutated** — all fine-tuning is performed on a deep
    copy so the original weights are preserved for downstream metrics.
    """

    name = "benign_finetuning"

    def __init__(
        self,
        epochs: int = 5,
        lr: float = 1e-3,
        optimizer_cls: str = "adam",
        finetune_fraction: float = 1.0,
        recovery_threshold: float = 0.15,
        weight_decay: float = 0.0,
    ) -> None:
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if not 0.0 < finetune_fraction <= 1.0:
            raise ValueError(f"finetune_fraction must be in (0, 1], got {finetune_fraction}")
        if optimizer_cls not in ("adam", "sgd"):
            raise ValueError(f"optimizer_cls must be 'adam' or 'sgd', got {optimizer_cls!r}")

        self.epochs = epochs
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.finetune_fraction = finetune_fraction
        self.recovery_threshold = recovery_threshold
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        *,
        benign_data: Optional[DataLoader] = None,
        original_accuracy: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run the benign fine-tuning attack and return knowledge restoration metrics.

        Parameters
        ----------
        model : nn.Module
            Unlearned model.  **Never mutated.**
        forget_data : DataLoader, optional
            Loader for the data that was supposed to be forgotten.
            If ``None``, most metrics are zero and the test trivially passes.
        retain_data : DataLoader, optional
            Loader for the retain set.  Used as benign fine-tuning data when
            ``benign_data`` is not provided.
        benign_data : DataLoader, optional
            Explicit benign domain data to fine-tune on.  If provided, this
            takes precedence over ``retain_data`` as the fine-tuning source.
            ``retain_data`` is still used for *measuring* retain performance.
        original_accuracy : float, optional
            The forget-set accuracy of the *original* (pre-unlearning) model,
            used to compute the restoration rate.  When not provided,
            restoration rate is measured relative to perfect accuracy (1.0).
        **kwargs : Any
            Ignored (for API compatibility).

        Returns
        -------
        Dict[str, float]
            Keys are prefixed with ``benign_ft_``.  Important entries:

            - ``benign_ft_pre_forget_accuracy`` — baseline before fine-tuning.
            - ``benign_ft_post_forget_accuracy`` — after all epochs.
            - ``benign_ft_forget_accuracy_recovery`` — absolute pp recovered.
            - ``benign_ft_restoration_rate`` — fraction of original accuracy
              recovered (0 = no restoration, 1 = full restoration).
            - ``benign_ft_restoration_auc`` — AUC of per-epoch accuracy curve.
            - ``benign_ft_epoch_{i}_forget_accuracy`` — per-epoch checkpoints.
            - ``benign_ft_passed`` — 1.0 if attack failed (robust unlearning).
        """
        # Trivially safe: nothing to measure
        if forget_data is None:
            return self._default_result()

        # Determine fine-tuning source
        ft_loader = benign_data if benign_data is not None else retain_data
        if ft_loader is None:
            return self._default_result(
                reason="no benign_data or retain_data provided for fine-tuning"
            )

        device = next(model.parameters()).device
        eval_criterion = nn.CrossEntropyLoss(reduction="sum")
        train_criterion = nn.CrossEntropyLoss(reduction="mean")

        # -- Baseline evaluation (original unlearned model) ----------------
        pre = _evaluate(model, forget_data, device, eval_criterion)
        pre_retain = (
            _evaluate(model, retain_data, device, eval_criterion)
            if retain_data is not None
            else {}
        )

        # -- Deep-copy and set up optimizer --------------------------------
        attack_model = copy.deepcopy(model).to(device)
        attack_model.train()

        if self.optimizer_cls == "adam":
            optimizer: torch.optim.Optimizer = torch.optim.Adam(
                attack_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                attack_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # Max batches per epoch (clamp by finetune_fraction)
        max_batches_per_epoch: Optional[int] = None
        n_total_batches = len(ft_loader)
        if self.finetune_fraction < 1.0:
            max_batches_per_epoch = max(1, int(n_total_batches * self.finetune_fraction))

        # -- Per-epoch training + evaluation loop --------------------------
        epoch_forget_accs: List[float] = []
        epoch_forget_losses: List[float] = []
        epoch_forget_confs: List[float] = []
        epoch_train_losses: List[float] = []

        for epoch in range(self.epochs):
            attack_model.train()
            train_loss = _finetune_one_epoch(
                attack_model, ft_loader, optimizer, train_criterion,
                device, max_batches_per_epoch,
            )
            epoch_train_losses.append(float(train_loss))

            # Evaluate on forget set after each epoch
            attack_model.eval()
            ep_forget = _evaluate(attack_model, forget_data, device, eval_criterion)
            epoch_forget_accs.append(ep_forget["accuracy"])
            epoch_forget_losses.append(ep_forget["loss"])
            epoch_forget_confs.append(ep_forget["confidence"])

        # -- Post-attack evaluation ----------------------------------------
        post = _evaluate(attack_model, forget_data, device, eval_criterion)
        post_retain = (
            _evaluate(attack_model, retain_data, device, eval_criterion)
            if retain_data is not None
            else {}
        )

        # -- Derived statistics -------------------------------------------
        acc_recovery = post["accuracy"] - pre["accuracy"]
        loss_recovery = pre["loss"] - post["loss"]   # +ve = loss went down (worse)
        conf_recovery = post["confidence"] - pre["confidence"]

        # Restoration rate: fraction of "original capacity" recovered.
        # If original_accuracy is unknown, use 1.0 (perfect pre-unlearning).
        orig_acc = original_accuracy if original_accuracy is not None else 1.0
        max_recoverable = max(orig_acc - pre["accuracy"], 1e-8)
        restoration_rate = float(
            min(1.0, max(0.0, acc_recovery / max_recoverable))
        )

        restoration_auc = _auc_trapezoid(epoch_forget_accs)

        # -- Assemble result dict ------------------------------------------
        result: Dict[str, float] = {
            "benign_ft_pre_forget_accuracy": float(pre["accuracy"]),
            "benign_ft_pre_forget_loss": float(pre["loss"]),
            "benign_ft_pre_forget_confidence": float(pre["confidence"]),
            "benign_ft_post_forget_accuracy": float(post["accuracy"]),
            "benign_ft_post_forget_loss": float(post["loss"]),
            "benign_ft_post_forget_confidence": float(post["confidence"]),
            "benign_ft_forget_accuracy_recovery": float(acc_recovery),
            "benign_ft_forget_loss_recovery": float(loss_recovery),
            "benign_ft_forget_confidence_recovery": float(conf_recovery),
            "benign_ft_restoration_rate": restoration_rate,
            "benign_ft_restoration_auc": float(restoration_auc),
            "benign_ft_epochs": float(self.epochs),
            "benign_ft_n_forget_samples": float(pre["n_samples"]),
            "benign_ft_passed": float(acc_recovery < self.recovery_threshold),
        }

        # Per-epoch checkpoints
        for i, (acc, loss, conf, tl) in enumerate(zip(
            epoch_forget_accs, epoch_forget_losses,
            epoch_forget_confs, epoch_train_losses,
        )):
            result[f"benign_ft_epoch_{i}_forget_accuracy"] = float(acc)
            result[f"benign_ft_epoch_{i}_forget_loss"] = float(loss)
            result[f"benign_ft_epoch_{i}_forget_confidence"] = float(conf)
            result[f"benign_ft_epoch_{i}_train_loss"] = float(tl)

        # Retain set changes
        if retain_data is not None and pre_retain and post_retain:
            result["benign_ft_pre_retain_accuracy"] = float(pre_retain["accuracy"])
            result["benign_ft_post_retain_accuracy"] = float(post_retain["accuracy"])
            result["benign_ft_retain_accuracy_change"] = float(
                post_retain["accuracy"] - pre_retain["accuracy"]
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_result(reason: str = "no forget data") -> Dict[str, float]:
        """Return a trivially-passing zero result when data is unavailable."""
        return {
            "benign_ft_pre_forget_accuracy": 0.0,
            "benign_ft_post_forget_accuracy": 0.0,
            "benign_ft_forget_accuracy_recovery": 0.0,
            "benign_ft_forget_loss_recovery": 0.0,
            "benign_ft_forget_confidence_recovery": 0.0,
            "benign_ft_restoration_rate": 0.0,
            "benign_ft_restoration_auc": 0.0,
            "benign_ft_epochs": 0.0,
            "benign_ft_n_forget_samples": 0.0,
            "benign_ft_passed": 1.0,   # trivially passes when no data to test
        }
