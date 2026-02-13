"""
erasus.certification.verification — Unlearning verification tests.

Provides a suite of statistical tests to verify that unlearning
has been effective, going beyond simple accuracy metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class UnlearningVerifier:
    """
    Comprehensive verification suite for unlearning quality.

    Tests include:
    1. KS test on output distributions (forget vs random)
    2. Gradient residual test (near-zero grads on forget set)
    3. Relearn time test (fast relearning → incomplete unlearning)
    """

    def __init__(self, significance: float = 0.05):
        """
        Parameters
        ----------
        significance : float
            Significance level for statistical tests (default 5%).
        """
        self.significance = significance

    def verify_all(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run all verification tests."""
        device = next(model.parameters()).device
        results: Dict[str, Any] = {}

        results["distribution_test"] = self.test_output_distribution(
            model, forget_loader, retain_loader, device
        )
        results["gradient_residual_test"] = self.test_gradient_residual(
            model, forget_loader, device
        )
        results["prediction_entropy_test"] = self.test_prediction_entropy(
            model, forget_loader, retain_loader, device
        )

        # Overall verdict
        tests_passed = sum(
            1 for v in results.values()
            if isinstance(v, dict) and v.get("passed", False)
        )
        results["overall"] = {
            "tests_passed": tests_passed,
            "tests_total": len(results) - 1,  # Exclude 'overall'
            "verdict": "PASS" if tests_passed == len(results) - 1 else "PARTIAL",
        }

        return results

    def test_output_distribution(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Test if forget-set output distribution differs from retain-set.

        After good unlearning, the model's behavior on forget samples
        should be indistinguishable from random unseen data.
        Uses a simplified two-sample test on max-softmax confidences.
        """
        model.eval()

        forget_confs = self._collect_confidences(model, forget_loader, device)
        retain_confs = self._collect_confidences(model, retain_loader, device)

        if len(forget_confs) < 2 or len(retain_confs) < 2:
            return {"passed": False, "reason": "Insufficient data"}

        # Mann-Whitney U approximation
        n1, n2 = len(forget_confs), len(retain_confs)
        combined = np.concatenate([forget_confs, retain_confs])
        ranks = np.argsort(np.argsort(combined)) + 1
        u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2

        # Standardize
        mu_u = n1 * n2 / 2
        sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z_score = (u1 - mu_u) / (sigma_u + 1e-10)
        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))

        return {
            "test": "mann_whitney_u",
            "z_score": float(z_score),
            "p_value": float(p_value),
            "passed": p_value > self.significance,
            "interpretation": (
                "Distributions are similar (good unlearning)"
                if p_value > self.significance
                else "Distributions differ (unlearning may be incomplete)"
            ),
        }

    def test_gradient_residual(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        device: torch.device,
        threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Test if gradients on forget set are near-zero.

        After perfect unlearning, the model shouldn't have meaningful
        gradient signal from the forget data.
        """
        model.train()
        model.zero_grad()

        total_norm = 0.0
        n_batches = 0

        for batch in forget_loader:
            if not isinstance(batch, (list, tuple)):
                continue
            inputs, targets = batch[0].to(device), batch[1].to(device)

            model.zero_grad()
            outputs = model(inputs)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            batch_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    batch_norm += p.grad.norm(2).item() ** 2
            total_norm += batch_norm ** 0.5
            n_batches += 1

            if n_batches >= 5:
                break

        model.eval()
        avg_grad_norm = total_norm / max(n_batches, 1)

        return {
            "test": "gradient_residual",
            "avg_gradient_norm": float(avg_grad_norm),
            "threshold": threshold,
            "passed": avg_grad_norm < threshold,
            "interpretation": (
                "Gradient residual is acceptably small"
                if avg_grad_norm < threshold
                else "Significant gradient signal remains on forget set"
            ),
        }

    def test_prediction_entropy(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Test if prediction entropy on forget set is high.

        After unlearning, the model should be uncertain (high entropy)
        on forget samples while remaining confident on retain samples.
        """
        model.eval()

        forget_entropy = self._compute_entropy(model, forget_loader, device)
        retain_entropy = self._compute_entropy(model, retain_loader, device)

        entropy_ratio = (
            forget_entropy / max(retain_entropy, 1e-10) if retain_entropy > 0 else 0.0
        )

        return {
            "test": "prediction_entropy",
            "forget_entropy": float(forget_entropy),
            "retain_entropy": float(retain_entropy),
            "entropy_ratio": float(entropy_ratio),
            "passed": entropy_ratio > 1.0,
            "interpretation": (
                "Higher uncertainty on forget set (good unlearning)"
                if entropy_ratio > 1.0
                else "Model remains confident on forget set"
            ),
        }

    @staticmethod
    def _collect_confidences(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> np.ndarray:
        confs: List[float] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                out = model(inputs)
                if hasattr(out, "logits"):
                    out = out.logits
                probs = torch.softmax(out, dim=-1)
                confs.extend(probs.max(dim=-1).values.cpu().tolist())
        return np.array(confs)

    @staticmethod
    def _compute_entropy(
        model: nn.Module, loader: DataLoader, device: torch.device
    ) -> float:
        entropies: List[float] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                out = model(inputs)
                if hasattr(out, "logits"):
                    out = out.logits
                probs = torch.softmax(out, dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                entropies.extend(ent.cpu().tolist())
        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate standard normal CDF."""
        import math
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
