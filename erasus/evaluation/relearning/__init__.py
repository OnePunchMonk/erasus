"""
erasus.evaluation.relearning — Relearning robustness evaluation.

Tests whether unlearning can be reversed by common post-processing
operations, exposing methods that merely obfuscate rather than truly
forget.

Based on:
- "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs
   via Benign Relearning" (ICLR 2025)
- "The Illusion of Unlearning" (CVPR 2025)

Attacks
-------
BenignFinetuningAttack
    Fine-tune the unlearned model on a small amount of benign data
    from the same domain.  If forget-set performance recovers,
    the model was obfuscating rather than forgetting.

QuantizationAttack
    Quantize the model to lower precision (8-bit, 4-bit).  Weight
    quantization can undo subtle parameter changes made during
    unlearning, reviving erased knowledge.

LoRARelearningAttack
    Attach a LoRA adapter and fine-tune on tangentially related data.
    LoRA modifies a small number of parameters; if this is enough to
    restore forgotten knowledge, unlearning was shallow.

PromptExtractionAttack
    Use adversarial prompting strategies (prefilling, role-playing,
    chain-of-thought elicitation) to extract supposedly-forgotten
    information.  For classifier models, this translates to input
    manipulation techniques.

RelearningRobustnessEvaluator
    Runs all attacks and produces a consolidated report.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from erasus.evaluation.relearning.base import BaseRelearningAttack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Dict[str, float]:
    """Compute accuracy and mean loss on a loader."""
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            total_loss += criterion(outputs, targets).item()
            correct += (outputs.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)

    return {
        "accuracy": correct / max(total, 1),
        "loss": total_loss / max(total, 1),
        "n_samples": total,
    }


# ---------------------------------------------------------------------------
# Benign Fine-tuning Attack
# ---------------------------------------------------------------------------

class BenignFinetuningAttack(BaseRelearningAttack):
    """
    Fine-tune the unlearned model on benign (non-forget) data and
    check if forget-set performance recovers.

    The intuition: if the model learned a representation that
    "encodes" the forget data and unlearning merely rotated the
    output head, then fine-tuning on related data can rotate it back.

    Parameters
    ----------
    epochs : int
        Number of fine-tuning epochs (default 3).
    lr : float
        Learning rate for fine-tuning (default 1e-3).
    finetune_fraction : float
        Fraction of retain data to use for fine-tuning (default 0.5).
        The rest is held out for measuring unrelated impact.
    recovery_threshold : float
        Maximum allowed accuracy recovery on forget set before the
        test is considered failed (default 0.15 = 15%).
    """

    def __init__(
        self,
        epochs: int = 3,
        lr: float = 1e-3,
        finetune_fraction: float = 0.5,
        recovery_threshold: float = 0.15,
    ):
        self.epochs = epochs
        self.lr = lr
        self.finetune_fraction = finetune_fraction
        self.recovery_threshold = recovery_threshold

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device

        # Measure baseline (pre-attack) metrics
        pre_forget = _compute_metrics(model, forget_data, device)
        pre_retain = _compute_metrics(model, retain_data, device)

        # Deep copy the model for the attack
        attack_model = copy.deepcopy(model).to(device)
        attack_model.train()

        optimizer = torch.optim.SGD(attack_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Fine-tune on a subset of retain data
        for epoch in range(self.epochs):
            batch_count = 0
            max_batches = max(1, int(len(retain_data) * self.finetune_fraction))

            for batch in retain_data:
                if batch_count >= max_batches:
                    break
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue

                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = attack_model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_count += 1

        # Measure post-attack metrics
        post_forget = _compute_metrics(attack_model, forget_data, device)
        post_retain = _compute_metrics(attack_model, retain_data, device)

        # Compute recovery
        acc_recovery = post_forget["accuracy"] - pre_forget["accuracy"]
        loss_recovery = pre_forget["loss"] - post_forget["loss"]

        return {
            "test": "benign_finetuning",
            "epochs": self.epochs,
            "lr": self.lr,
            "pre_forget_accuracy": float(pre_forget["accuracy"]),
            "post_forget_accuracy": float(post_forget["accuracy"]),
            "forget_accuracy_recovery": float(acc_recovery),
            "pre_forget_loss": float(pre_forget["loss"]),
            "post_forget_loss": float(post_forget["loss"]),
            "forget_loss_recovery": float(loss_recovery),
            "pre_retain_accuracy": float(pre_retain["accuracy"]),
            "post_retain_accuracy": float(post_retain["accuracy"]),
            "retain_accuracy_change": float(post_retain["accuracy"] - pre_retain["accuracy"]),
            "passed": acc_recovery < self.recovery_threshold,
            "interpretation": (
                f"Benign fine-tuning recovered only {acc_recovery:.1%} accuracy (robust)"
                if acc_recovery < self.recovery_threshold
                else f"Benign fine-tuning recovered {acc_recovery:.1%} accuracy — "
                     f"model was obfuscating, not truly forgetting"
            ),
        }


# ---------------------------------------------------------------------------
# Quantization Attack
# ---------------------------------------------------------------------------

class QuantizationAttack(BaseRelearningAttack):
    """
    Quantize model weights and check if forget-set performance recovers.

    Unlearning methods that make small, precise parameter changes can
    be undone by the rounding effects of weight quantization.  This is
    a real threat in production where models are routinely quantized
    for inference.

    Parameters
    ----------
    bit_widths : list[int]
        Bit widths to test (default [8, 4]).
    recovery_threshold : float
        Maximum allowed accuracy recovery (default 0.10 = 10%).
    """

    def __init__(
        self,
        bit_widths: Optional[List[int]] = None,
        recovery_threshold: float = 0.10,
    ):
        self.bit_widths = bit_widths or [8, 4]
        self.recovery_threshold = recovery_threshold

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device

        # Baseline
        pre_forget = _compute_metrics(model, forget_data, device)
        pre_retain = _compute_metrics(model, retain_data, device) if retain_data else {}

        results_by_bits: Dict[str, float] = {}
        worst_recovery = 0.0

        for bits in self.bit_widths:
            quantized = self._quantize_model(model, bits)
            quantized = quantized.to(device)

            post_forget = _compute_metrics(quantized, forget_data, device)
            acc_recovery = post_forget["accuracy"] - pre_forget["accuracy"]
            worst_recovery = max(worst_recovery, acc_recovery)

            key = f"{bits}bit"
            results_by_bits[f"quant_{key}_forget_accuracy"] = float(post_forget["accuracy"])
            results_by_bits[f"quant_{key}_forget_recovery"] = float(acc_recovery)

            if retain_data:
                post_retain = _compute_metrics(quantized, retain_data, device)
                results_by_bits[f"quant_{key}_retain_accuracy"] = float(post_retain["accuracy"])

            del quantized

        return {
            "test": "quantization",
            "bit_widths_tested": self.bit_widths,
            "pre_forget_accuracy": float(pre_forget["accuracy"]),
            **results_by_bits,
            "worst_recovery": float(worst_recovery),
            "passed": worst_recovery < self.recovery_threshold,
            "interpretation": (
                f"Quantization recovered at most {worst_recovery:.1%} accuracy (robust)"
                if worst_recovery < self.recovery_threshold
                else f"Quantization recovered {worst_recovery:.1%} accuracy — "
                     f"unlearning changes are too fine-grained to survive quantization"
            ),
        }

    @staticmethod
    def _quantize_model(model: nn.Module, bits: int) -> nn.Module:
        """
        Simulate uniform symmetric quantization of model weights.

        This is a simple simulation — not hardware-native quantization —
        but captures the core effect of rounding weights to lower precision.
        """
        quantized = copy.deepcopy(model)

        n_levels = 2 ** (bits - 1)  # Symmetric quantization

        with torch.no_grad():
            for param in quantized.parameters():
                if param.numel() == 0:
                    continue
                # Per-tensor symmetric quantization
                abs_max = param.abs().max().clamp(min=1e-8)
                scale = abs_max / n_levels

                # Quantize and dequantize
                quantized_vals = torch.round(param / scale).clamp(-n_levels, n_levels - 1)
                param.copy_(quantized_vals * scale)

        return quantized


# ---------------------------------------------------------------------------
# LoRA Relearning Attack
# ---------------------------------------------------------------------------

class LoRARelearningAttack(BaseRelearningAttack):
    """
    Attach a low-rank adapter and fine-tune on retain data to test
    if forgotten knowledge can be recovered through LoRA.

    LoRA modifies only a small number of parameters (rank r).  If
    this is sufficient to restore forget-set performance, the
    unlearning was not deep enough.

    Parameters
    ----------
    rank : int
        LoRA rank (default 4).
    epochs : int
        Fine-tuning epochs (default 3).
    lr : float
        Learning rate (default 1e-3).
    recovery_threshold : float
        Maximum allowed accuracy recovery (default 0.15).
    related_data_fraction : float
        Fraction of retain data used as related-data fine-tuning signal.
    """

    def __init__(
        self,
        rank: int = 4,
        epochs: int = 3,
        lr: float = 1e-3,
        recovery_threshold: float = 0.15,
        related_data_fraction: float = 1.0,
    ):
        self.rank = rank
        self.epochs = epochs
        self.lr = lr
        self.recovery_threshold = recovery_threshold
        self.related_data_fraction = related_data_fraction

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device

        # Baseline
        pre_forget = _compute_metrics(model, forget_data, device)
        pre_related = _compute_metrics(model, retain_data, device)

        # Create LoRA-augmented model
        lora_model = _LoRAWrapper(model, rank=self.rank).to(device)
        lora_model.train()

        # Only train LoRA parameters
        lora_params = [p for n, p in lora_model.named_parameters() if "lora_" in n]
        if not lora_params:
            return {
                "test": "lora_relearning",
                "error": "No LoRA parameters found (model may not have linear layers)",
                "passed": True,
            }

        optimizer = torch.optim.Adam(lora_params, lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        related_loader = _subset_loader(retain_data, self.related_data_fraction)

        # Fine-tune LoRA on related retain-domain data
        for epoch in range(self.epochs):
            for batch in related_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                outputs = lora_model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Measure post-LoRA metrics
        post_forget = _compute_metrics(lora_model, forget_data, device)
        post_related = _compute_metrics(lora_model, retain_data, device)
        acc_recovery = post_forget["accuracy"] - pre_forget["accuracy"]
        loss_recovery = pre_forget["loss"] - post_forget["loss"]
        concept_revival = acc_recovery + max(loss_recovery, 0.0)

        # Count trainable parameters
        n_lora_params = sum(p.numel() for p in lora_params)
        n_total_params = sum(p.numel() for p in model.parameters())

        return {
            "test": "lora_relearning",
            "rank": self.rank,
            "epochs": self.epochs,
            "lora_params": int(n_lora_params),
            "total_params": int(n_total_params),
            "lora_param_ratio": float(n_lora_params / max(n_total_params, 1)),
            "related_data_fraction": float(self.related_data_fraction),
            "pre_forget_accuracy": float(pre_forget["accuracy"]),
            "post_forget_accuracy": float(post_forget["accuracy"]),
            "pre_related_accuracy": float(pre_related["accuracy"]),
            "post_related_accuracy": float(post_related["accuracy"]),
            "forget_accuracy_recovery": float(acc_recovery),
            "forget_loss_recovery": float(loss_recovery),
            "concept_revival": float(concept_revival),
            "passed": concept_revival < self.recovery_threshold,
            "interpretation": (
                f"LoRA (rank {self.rank}) produced only {concept_revival:.1%} concept revival (robust)"
                if concept_revival < self.recovery_threshold
                else f"LoRA (rank {self.rank}, {n_lora_params} params) revived forgotten behavior by "
                     f"{concept_revival:.1%} after related-data fine-tuning"
            ),
        }


class _LoRAWrapper(nn.Module):
    """
    Wraps a model with LoRA adapters on all Linear layers.

    This is a lightweight LoRA implementation for evaluation purposes
    (does not depend on the ``peft`` library).
    """

    def __init__(self, base_model: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.rank = rank
        self.alpha = alpha
        self._adapters: nn.ModuleDict = nn.ModuleDict()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add LoRA adapters to Linear layers
        self._inject_lora(self.base_model, prefix="")

    def _inject_lora(self, module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                in_f, out_f = child.in_features, child.out_features
                r = min(self.rank, in_f, out_f)
                # Store adapter using a safe key name
                safe_key = full_name.replace(".", "_")
                self._adapters[f"lora_A_{safe_key}"] = nn.Linear(in_f, r, bias=False)
                self._adapters[f"lora_B_{safe_key}"] = nn.Linear(r, out_f, bias=False)
                # Initialize B to zero so LoRA starts as identity
                nn.init.zeros_(self._adapters[f"lora_B_{safe_key}"].weight)
                nn.init.normal_(self._adapters[f"lora_A_{safe_key}"].weight, std=0.02)

                # Replace the forward of this Linear
                original_forward = child.forward

                def make_hook(a_key: str, b_key: str, orig_fn):
                    def new_forward(x):
                        base_out = orig_fn(x)
                        lora_out = self._adapters[b_key](self._adapters[a_key](x))
                        return base_out + self.alpha * lora_out
                    return new_forward

                child.forward = make_hook(f"lora_A_{safe_key}", f"lora_B_{safe_key}", original_forward)
            else:
                self._inject_lora(child, full_name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model(*args, **kwargs)


def _subset_loader(loader: DataLoader, fraction: float) -> DataLoader:
    """Create a deterministic prefix subset of a loader's dataset."""
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction >= 1.0 or not hasattr(loader, "dataset"):
        return loader

    dataset = loader.dataset
    subset_size = max(1, int(len(dataset) * fraction))
    subset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
    )


# ---------------------------------------------------------------------------
# Prompt Extraction Attack
# ---------------------------------------------------------------------------

class PromptExtractionAttack(BaseRelearningAttack):
    """
    Attempt to extract forgotten information via input manipulation.

    For classifier models, this uses:
    1. Gradient-guided adversarial perturbation toward the correct class
    2. Input interpolation between forget and retain samples
    3. Feature amplification (scaling input dimensions that the model
       attends to most)

    For generative models (if `generate` method exists), this would
    use prompt engineering — but that is left for future implementation.

    Parameters
    ----------
    n_pgd_steps : int
        PGD steps for adversarial perturbation (default 10).
    epsilon : float
        Maximum perturbation L∞ norm (default 0.1).
    recovery_threshold : float
        Maximum allowed accuracy recovery (default 0.10).
    """

    def __init__(
        self,
        n_pgd_steps: int = 10,
        epsilon: float = 0.1,
        recovery_threshold: float = 0.10,
    ):
        self.n_pgd_steps = n_pgd_steps
        self.epsilon = epsilon
        self.recovery_threshold = recovery_threshold

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device

        # Baseline accuracy on forget set
        pre_metrics = _compute_metrics(model, forget_data, device)

        # Attack 1: PGD adversarial perturbation toward correct class
        pgd_acc = self._pgd_attack(model, forget_data, device)

        # Attack 2: Input interpolation with retain samples
        interp_acc = 0.0
        if retain_data is not None:
            interp_acc = self._interpolation_attack(model, forget_data, retain_data, device)

        # Attack 3: Feature amplification
        amp_acc = self._amplification_attack(model, forget_data, device)

        worst_recovery = max(
            pgd_acc - pre_metrics["accuracy"],
            interp_acc - pre_metrics["accuracy"],
            amp_acc - pre_metrics["accuracy"],
        )

        return {
            "test": "prompt_extraction",
            "pre_forget_accuracy": float(pre_metrics["accuracy"]),
            "pgd_accuracy": float(pgd_acc),
            "pgd_recovery": float(pgd_acc - pre_metrics["accuracy"]),
            "interpolation_accuracy": float(interp_acc),
            "interpolation_recovery": float(interp_acc - pre_metrics["accuracy"]),
            "amplification_accuracy": float(amp_acc),
            "amplification_recovery": float(amp_acc - pre_metrics["accuracy"]),
            "worst_recovery": float(worst_recovery),
            "passed": worst_recovery < self.recovery_threshold,
            "interpretation": (
                f"Prompt extraction recovered at most {worst_recovery:.1%} (robust)"
                if worst_recovery < self.recovery_threshold
                else f"Prompt extraction recovered {worst_recovery:.1%} accuracy — "
                     f"forgotten information is still accessible via input manipulation"
            ),
        }

    def _pgd_attack(
        self, model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> float:
        """PGD attack to maximize P(correct class) on forget set."""
        model.eval()
        correct = 0
        total = 0
        step_size = self.epsilon / max(self.n_pgd_steps, 1) * 2

        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)
            adv_inputs = inputs.clone().detach().requires_grad_(True)

            for step in range(self.n_pgd_steps):
                outputs = model(adv_inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                # Maximize probability of the true class (minimize negative log-prob)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()

                with torch.no_grad():
                    # Gradient descent on the loss = moving toward correct class
                    grad = adv_inputs.grad.sign()
                    adv_inputs = adv_inputs - step_size * grad
                    # Project back to epsilon ball
                    delta = (adv_inputs - inputs).clamp(-self.epsilon, self.epsilon)
                    adv_inputs = (inputs + delta).detach().requires_grad_(True)

            with torch.no_grad():
                outputs = model(adv_inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)

        return correct / max(total, 1)

    @staticmethod
    def _interpolation_attack(
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        device: torch.device,
        alpha: float = 0.3,
    ) -> float:
        """Interpolate forget inputs toward retain inputs."""
        model.eval()
        correct = 0
        total = 0

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        while True:
            try:
                f_batch = next(forget_iter)
                r_batch = next(retain_iter)
            except StopIteration:
                break

            if not isinstance(f_batch, (list, tuple)) or len(f_batch) < 2:
                continue
            if not isinstance(r_batch, (list, tuple)) or len(r_batch) < 2:
                continue

            f_inputs, f_targets = f_batch[0].to(device), f_batch[1].to(device)
            r_inputs = r_batch[0].to(device)

            # Match batch sizes
            min_size = min(f_inputs.size(0), r_inputs.size(0))
            mixed = (1 - alpha) * f_inputs[:min_size] + alpha * r_inputs[:min_size]

            with torch.no_grad():
                outputs = model(mixed)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                correct += (outputs.argmax(dim=-1) == f_targets[:min_size]).sum().item()
                total += min_size

        return correct / max(total, 1)

    @staticmethod
    def _amplification_attack(
        model: nn.Module, loader: DataLoader, device: torch.device,
        factor: float = 2.0,
    ) -> float:
        """Amplify input features to boost signal."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                amplified = inputs * factor
                outputs = model(amplified)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)

        return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Unified Relearning Robustness Evaluator
# ---------------------------------------------------------------------------

class RelearningRobustnessEvaluator:
    """
    Runs all relearning robustness attacks and produces a consolidated report.

    Parameters
    ----------
    attacks : list[str], optional
        Subset of attacks to run.  Default: all.
        Valid names: ``benign_finetuning``, ``quantization``,
        ``lora_relearning``, ``prompt_extraction``.

    Example
    -------
    >>> evaluator = RelearningRobustnessEvaluator()
    >>> report = evaluator.evaluate(model, forget_loader, retain_loader)
    >>> print(report["overall"]["verdict"])
    PASS
    """

    ALL_ATTACKS = (
        "benign_finetuning",
        "quantization",
        "lora_relearning",
        "prompt_extraction",
    )

    def __init__(
        self,
        attacks: Optional[List[str]] = None,
        benign_finetuning_kwargs: Optional[Dict[str, Any]] = None,
        quantization_kwargs: Optional[Dict[str, Any]] = None,
        lora_relearning_kwargs: Optional[Dict[str, Any]] = None,
        prompt_extraction_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.attack_names = list(attacks or self.ALL_ATTACKS)
        self._attacks = {
            "benign_finetuning": BenignFinetuningAttack(**(benign_finetuning_kwargs or {})),
            "quantization": QuantizationAttack(**(quantization_kwargs or {})),
            "lora_relearning": LoRARelearningAttack(**(lora_relearning_kwargs or {})),
            "prompt_extraction": PromptExtractionAttack(**(prompt_extraction_kwargs or {})),
        }

    def evaluate(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run all selected relearning robustness attacks."""
        results: Dict[str, Any] = {}
        n_passed = 0
        n_total = 0

        for attack_name in self.attack_names:
            attack = self._attacks.get(attack_name)
            if attack is None:
                continue

            try:
                result = attack.run(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                results[attack_name] = result
                if result.get("passed", False):
                    n_passed += 1
                n_total += 1
            except Exception as e:
                results[attack_name] = {"error": str(e), "passed": False}
                n_total += 1

        results["overall"] = {
            "attacks_passed": n_passed,
            "attacks_total": n_total,
            "passed": n_passed == n_total,
            "verdict": "PASS" if n_passed == n_total else (
                "PARTIAL" if n_passed > 0 else "FAIL"
            ),
        }

        return results


__all__ = [
    "BaseRelearningAttack",
    "BenignFinetuningAttack",
    "QuantizationAttack",
    "LoRARelearningAttack",
    "PromptExtractionAttack",
    "RelearningRobustnessEvaluator",
]
