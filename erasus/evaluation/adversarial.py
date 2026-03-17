"""
erasus.evaluation.adversarial — Adversarial unlearning evaluation.

Implements stress-tests that expose weaknesses in unlearning methods
missed by standard benchmarks.

Based on findings from:
- "LLM Unlearning Benchmarks are Weak Measures of Progress" (CMU, 2025)
- "The Illusion of Unlearning" (CVPR 2025)
- NeurIPS 2023 Machine Unlearning Challenge findings

Tests
-----
CrossPromptLeakageTest
    Combine forget and retain queries in a single prompt/batch.
    If unlearned information resurfaces in the joint context, the
    method has failed — even if it passes independent evaluation.

KeywordInjectionTest
    Insert forget-set keywords or features into evaluation inputs
    (e.g., into incorrect MCQ options or unrelated prompts).  Methods
    that merely suppress surface patterns rather than truly forgetting
    will show accuracy degradation.

ParaphraseRobustnessTest
    Apply input perturbations (noise, permutation, augmentation) to
    forget-set queries.  Robust unlearning should be invariant to
    input reformulation.

AdversarialEvaluator
    Runs all adversarial tests and produces a consolidated report.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Cross-Prompt Leakage
# ---------------------------------------------------------------------------

class CrossPromptLeakageTest:
    """
    Test whether forget-set information leaks when forget and retain
    samples are presented in the same batch or context.

    Standard benchmarks evaluate forget and retain sets independently.
    This test combines them: for each forget sample, we pair it with
    a retain sample and measure whether the model's behavior on the
    forget sample changes compared to independent evaluation.

    A significant change indicates that context leaks information
    about the "forgotten" data — the model hasn't truly unlearned.

    Parameters
    ----------
    n_pairs : int
        Number of (forget, retain) pairs to test.  Default: min of
        available samples.
    change_threshold : float
        Minimum relative change in loss to flag as leakage (default 0.1 = 10%).
    """

    def __init__(self, n_pairs: Optional[int] = None, change_threshold: float = 0.10):
        self.n_pairs = n_pairs
        self.change_threshold = change_threshold

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()

        # Collect all samples
        forget_samples = self._collect_samples(forget_data)
        retain_samples = self._collect_samples(retain_data)

        if len(forget_samples) == 0 or len(retain_samples) == 0:
            return {"error": "Insufficient data", "passed": False}

        n_pairs = self.n_pairs or min(len(forget_samples), len(retain_samples))
        n_pairs = min(n_pairs, len(forget_samples), len(retain_samples))

        # Phase 1: Independent evaluation — loss on forget samples alone
        independent_losses = []
        for i in range(n_pairs):
            fx, fy = forget_samples[i]
            loss = self._compute_loss(model, fx.unsqueeze(0).to(device), fy.unsqueeze(0).to(device))
            independent_losses.append(loss)

        # Phase 2: Joint evaluation — loss on forget samples when batched with retain
        joint_losses = []
        for i in range(n_pairs):
            fx, fy = forget_samples[i]
            rx, ry = retain_samples[i]
            # Create a mini-batch containing both
            batch_x = torch.stack([fx, rx]).to(device)
            batch_y = torch.stack([fy, ry]).to(device)

            with torch.no_grad():
                outputs = model(batch_x)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                # Loss on the forget sample (index 0) within the joint batch
                loss = F.cross_entropy(outputs[0:1], batch_y[0:1]).item()
            joint_losses.append(loss)

        independent_arr = np.array(independent_losses)
        joint_arr = np.array(joint_losses)

        # Compute leakage: relative change in loss
        # If loss decreases in joint context → model is "remembering" when given context
        relative_change = (joint_arr - independent_arr) / (np.abs(independent_arr) + 1e-8)
        leakage_mask = np.abs(relative_change) > self.change_threshold
        leakage_rate = leakage_mask.mean()

        # Directional analysis
        loss_decreased = (relative_change < -self.change_threshold).mean()
        loss_increased = (relative_change > self.change_threshold).mean()

        return {
            "test": "cross_prompt_leakage",
            "n_pairs_tested": int(n_pairs),
            "leakage_rate": float(leakage_rate),
            "loss_decreased_rate": float(loss_decreased),
            "loss_increased_rate": float(loss_increased),
            "mean_relative_change": float(relative_change.mean()),
            "std_relative_change": float(relative_change.std()),
            "independent_loss_mean": float(independent_arr.mean()),
            "joint_loss_mean": float(joint_arr.mean()),
            "passed": float(leakage_rate) < 0.2,
            "interpretation": (
                "Low leakage: model behavior is consistent across contexts (good)"
                if leakage_rate < 0.2
                else f"High leakage ({leakage_rate:.1%}): forget-set behavior changes "
                     f"when presented alongside retain data"
            ),
        }

    @staticmethod
    def _collect_samples(loader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        samples = []
        for batch in loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                continue
            inputs, targets = batch[0], batch[1]
            for i in range(inputs.size(0)):
                samples.append((inputs[i].cpu(), targets[i].cpu()))
        return samples

    @staticmethod
    def _compute_loss(
        model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            return F.cross_entropy(outputs, targets).item()


# ---------------------------------------------------------------------------
# Keyword Injection
# ---------------------------------------------------------------------------

class KeywordInjectionTest:
    """
    Test whether injecting forget-set features into unrelated inputs
    disrupts the model's predictions.

    Motivation: if a model has truly forgotten a concept, encountering
    features associated with that concept in new contexts should have
    no effect.  But if the model merely suppresses a surface pattern,
    injecting related features will cause measurable degradation.

    Implementation: for classifier models, we blend forget-set features
    into retain-set inputs at various strengths and measure accuracy
    change on the retain set.

    Parameters
    ----------
    injection_strengths : list[float]
        Blending weights for feature injection (default [0.05, 0.1, 0.2]).
    """

    def __init__(
        self,
        injection_strengths: Optional[List[float]] = None,
    ):
        self.injection_strengths = injection_strengths or [0.05, 0.1, 0.2]

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()

        # Compute baseline accuracy on retain set
        baseline_acc = self._compute_accuracy(model, retain_data, device)

        # Compute mean forget-set features (the "keyword" representation)
        forget_centroid = self._compute_centroid(forget_data)
        if forget_centroid is None:
            return {"error": "Could not compute forget centroid", "passed": False}

        # Inject forget features into retain inputs at various strengths
        results_by_strength: Dict[str, float] = {}
        worst_drop = 0.0

        for alpha in self.injection_strengths:
            injected_acc = self._compute_injected_accuracy(
                model, retain_data, forget_centroid, alpha, device
            )
            drop = baseline_acc - injected_acc
            worst_drop = max(worst_drop, drop)
            key = f"alpha_{alpha:.2f}"
            results_by_strength[f"injection_{key}_accuracy"] = float(injected_acc)
            results_by_strength[f"injection_{key}_drop"] = float(drop)

        return {
            "test": "keyword_injection",
            "baseline_retain_accuracy": float(baseline_acc),
            **results_by_strength,
            "worst_accuracy_drop": float(worst_drop),
            "passed": worst_drop < 0.15,
            "interpretation": (
                "Model is robust to forget-feature injection (good)"
                if worst_drop < 0.15
                else f"Accuracy drops {worst_drop:.1%} when forget features are "
                     f"injected — model is fragile, not forgetful"
            ),
        }

    @staticmethod
    def _compute_accuracy(
        model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> float:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)

    @staticmethod
    def _compute_centroid(loader: DataLoader) -> Optional[torch.Tensor]:
        """Compute the mean feature vector of the dataset."""
        sum_vec = None
        count = 0
        for batch in loader:
            if not isinstance(batch, (list, tuple)):
                continue
            inputs = batch[0]
            if sum_vec is None:
                sum_vec = torch.zeros_like(inputs[0], dtype=torch.float64)
            sum_vec += inputs.sum(dim=0).double()
            count += inputs.size(0)
        if sum_vec is None or count == 0:
            return None
        return (sum_vec / count).float()

    @staticmethod
    def _compute_injected_accuracy(
        model: nn.Module,
        loader: DataLoader,
        centroid: torch.Tensor,
        alpha: float,
        device: torch.device,
    ) -> float:
        """Accuracy on retain data with forget centroid blended in."""
        correct = 0
        total = 0
        centroid_dev = centroid.to(device)

        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                # Blend: x' = (1 - alpha) * x + alpha * centroid
                injected = (1 - alpha) * inputs + alpha * centroid_dev.unsqueeze(0)
                outputs = model(injected)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Paraphrase Robustness
# ---------------------------------------------------------------------------

class ParaphraseRobustnessTest:
    """
    Test whether unlearning is robust to input perturbations.

    If a model has truly unlearned, it should remain uncertain on
    forget-set samples even when they are perturbed.  If small
    perturbations restore confident predictions, the unlearning is
    superficial.

    Perturbation types:
    - Gaussian noise addition
    - Feature permutation (shuffle dimensions)
    - Scaling (multiply by random factors)

    Parameters
    ----------
    noise_levels : list[float]
        Standard deviations for Gaussian noise (default [0.01, 0.05, 0.1]).
    n_perturbations : int
        Number of random perturbations per noise level (default 3).
    """

    def __init__(
        self,
        noise_levels: Optional[List[float]] = None,
        n_perturbations: int = 3,
    ):
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1]
        self.n_perturbations = n_perturbations

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()

        # Baseline: accuracy and confidence on unperturbed forget set
        baseline_acc, baseline_conf = self._eval(model, forget_data, device)

        # Perturbed evaluations
        noise_results: Dict[str, float] = {}
        max_acc_recovery = 0.0

        for sigma in self.noise_levels:
            accs = []
            confs = []
            for _ in range(self.n_perturbations):
                acc, conf = self._eval_perturbed(
                    model, forget_data, device, noise_std=sigma,
                )
                accs.append(acc)
                confs.append(conf)

            mean_acc = float(np.mean(accs))
            mean_conf = float(np.mean(confs))
            acc_recovery = mean_acc - baseline_acc
            max_acc_recovery = max(max_acc_recovery, acc_recovery)

            key = f"noise_{sigma:.3f}"
            noise_results[f"perturbed_{key}_accuracy"] = mean_acc
            noise_results[f"perturbed_{key}_confidence"] = mean_conf
            noise_results[f"perturbed_{key}_acc_recovery"] = float(acc_recovery)

        # Permutation test
        perm_acc, perm_conf = self._eval_permuted(model, forget_data, device)
        perm_recovery = perm_acc - baseline_acc

        # Scale test
        scale_acc, scale_conf = self._eval_scaled(model, forget_data, device)
        scale_recovery = scale_acc - baseline_acc

        max_acc_recovery = max(max_acc_recovery, perm_recovery, scale_recovery)

        return {
            "test": "paraphrase_robustness",
            "baseline_forget_accuracy": float(baseline_acc),
            "baseline_forget_confidence": float(baseline_conf),
            **noise_results,
            "permutation_accuracy": float(perm_acc),
            "permutation_acc_recovery": float(perm_recovery),
            "scaling_accuracy": float(scale_acc),
            "scaling_acc_recovery": float(scale_recovery),
            "max_accuracy_recovery": float(max_acc_recovery),
            "passed": max_acc_recovery < 0.15,
            "interpretation": (
                "Unlearning is robust to input perturbations (good)"
                if max_acc_recovery < 0.15
                else f"Perturbations recover {max_acc_recovery:.1%} accuracy — "
                     f"unlearning is superficial"
            ),
        }

    @staticmethod
    def _eval(
        model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> Tuple[float, float]:
        """Return (accuracy, mean_confidence) on a loader."""
        correct = 0
        total = 0
        confs: list = []
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=-1)
                confs.extend(probs.max(dim=-1).values.cpu().tolist())
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1), float(np.mean(confs)) if confs else 0.0

    def _eval_perturbed(
        self, model: nn.Module, loader: DataLoader, device: torch.device,
        noise_std: float,
    ) -> Tuple[float, float]:
        """Evaluate with Gaussian noise added to inputs."""
        correct = 0
        total = 0
        confs: list = []
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                noisy = inputs + torch.randn_like(inputs) * noise_std
                outputs = model(noisy)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=-1)
                confs.extend(probs.max(dim=-1).values.cpu().tolist())
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1), float(np.mean(confs)) if confs else 0.0

    @staticmethod
    def _eval_permuted(
        model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> Tuple[float, float]:
        """Evaluate with feature dimensions permuted."""
        correct = 0
        total = 0
        confs: list = []
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                # Permute the last dimension
                perm = torch.randperm(inputs.size(-1), device=device)
                permuted = inputs.index_select(-1, perm)
                outputs = model(permuted)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=-1)
                confs.extend(probs.max(dim=-1).values.cpu().tolist())
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1), float(np.mean(confs)) if confs else 0.0

    @staticmethod
    def _eval_scaled(
        model: nn.Module, loader: DataLoader, device: torch.device,
    ) -> Tuple[float, float]:
        """Evaluate with random per-sample scaling."""
        correct = 0
        total = 0
        confs: list = []
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    continue
                inputs, targets = batch[0].to(device), batch[1].to(device)
                # Scale each sample by a random factor in [0.8, 1.2]
                scale = 0.8 + 0.4 * torch.rand(inputs.size(0), 1, device=device)
                # Handle multi-dimensional inputs
                while scale.dim() < inputs.dim():
                    scale = scale.unsqueeze(-1)
                scaled = inputs * scale
                outputs = model(scaled)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                probs = torch.softmax(outputs, dim=-1)
                confs.extend(probs.max(dim=-1).values.cpu().tolist())
                correct += (outputs.argmax(dim=-1) == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1), float(np.mean(confs)) if confs else 0.0


# ---------------------------------------------------------------------------
# Unified Adversarial Evaluator
# ---------------------------------------------------------------------------

class AdversarialEvaluator:
    """
    Runs all adversarial evaluation tests and produces a consolidated report.

    Parameters
    ----------
    tests : list[str], optional
        Subset of tests to run.  Default: all.
        Valid names: ``cross_prompt``, ``keyword_injection``, ``paraphrase``.

    Example
    -------
    >>> evaluator = AdversarialEvaluator()
    >>> report = evaluator.evaluate(model, forget_loader, retain_loader)
    >>> print(report["overall"]["passed"])
    True
    """

    ALL_TESTS = ("cross_prompt", "keyword_injection", "paraphrase")

    def __init__(
        self,
        tests: Optional[List[str]] = None,
        cross_prompt_kwargs: Optional[Dict[str, Any]] = None,
        keyword_injection_kwargs: Optional[Dict[str, Any]] = None,
        paraphrase_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.test_names = list(tests or self.ALL_TESTS)
        self._tests = {
            "cross_prompt": CrossPromptLeakageTest(**(cross_prompt_kwargs or {})),
            "keyword_injection": KeywordInjectionTest(**(keyword_injection_kwargs or {})),
            "paraphrase": ParaphraseRobustnessTest(**(paraphrase_kwargs or {})),
        }

    def evaluate(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run all selected adversarial tests."""
        results: Dict[str, Any] = {}
        n_passed = 0
        n_total = 0

        for test_name in self.test_names:
            test = self._tests.get(test_name)
            if test is None:
                continue

            try:
                result = test.run(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                results[test_name] = result
                if result.get("passed", False):
                    n_passed += 1
                n_total += 1
            except Exception as e:
                results[test_name] = {"error": str(e), "passed": False}
                n_total += 1

        results["overall"] = {
            "tests_passed": n_passed,
            "tests_total": n_total,
            "passed": n_passed == n_total,
            "verdict": "PASS" if n_passed == n_total else (
                "PARTIAL" if n_passed > 0 else "FAIL"
            ),
        }

        return results
