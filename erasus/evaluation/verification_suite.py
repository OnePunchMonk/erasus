"""
erasus.evaluation.verification_suite — Unified unlearning verification.

Combines standard metrics, adversarial tests, relearning robustness
checks, and the full MIA suite into a single comprehensive report
that answers: "Did unlearning actually work?"

This is the top-level entry point for rigorous unlearning evaluation.

Example
-------
>>> from erasus.evaluation import UnlearningVerificationSuite
>>> suite = UnlearningVerificationSuite()
>>> report = suite.verify(model, forget_loader, retain_loader)
>>> print(report["verdict"])
PASS
>>> print(report["confidence"])
0.85
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.evaluation.adversarial import AdversarialEvaluator
from erasus.evaluation.relearning import RelearningRobustnessEvaluator
from erasus.metrics.forgetting.mia_suite import MIASuite
from erasus.metrics.forgetting.memorization import (
    ExactMemorizationMetric,
    ExtractionStrengthMetric,
    VerbatimMemorizationMetric,
)


class UnlearningVerificationSuite:
    """
    Comprehensive unlearning verification.

    Runs four categories of evaluation:

    1. **MIA Suite** — 6-attack membership inference battery
    2. **Memorization Metrics** — extraction strength, exact memorization,
       verbatim memorization
    3. **Adversarial Tests** — cross-prompt leakage, keyword injection,
       paraphrase robustness
    4. **Relearning Robustness** — benign fine-tuning, quantization,
       LoRA relearning, prompt extraction attacks

    Produces a single verdict (PASS/PARTIAL/FAIL) with a confidence
    score based on how many tests pass and the severity of failures.

    Parameters
    ----------
    categories : list[str], optional
        Which categories to run.  Default: all.
        Valid: ``mia``, ``memorization``, ``adversarial``, ``relearning``.
    reference_model : nn.Module, optional
        Reference model (pre-unlearning) for the Reference MIA attack.
    strict : bool
        If True, requires ALL tests to pass for a PASS verdict.
        If False (default), allows partial passes with reduced confidence.

    Example
    -------
    >>> suite = UnlearningVerificationSuite(categories=["mia", "adversarial"])
    >>> report = suite.verify(model, forget_loader, retain_loader)
    """

    ALL_CATEGORIES = ("mia", "memorization", "adversarial", "relearning")

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        reference_model: Optional[nn.Module] = None,
        strict: bool = False,
        # Pass-through kwargs for sub-evaluators
        mia_kwargs: Optional[Dict[str, Any]] = None,
        adversarial_kwargs: Optional[Dict[str, Any]] = None,
        relearning_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.categories = list(categories or self.ALL_CATEGORIES)
        self.reference_model = reference_model
        self.strict = strict

        # Initialize sub-evaluators
        self._mia = MIASuite(reference_model=reference_model, **(mia_kwargs or {}))
        self._extraction = ExtractionStrengthMetric()
        self._exact_mem = ExactMemorizationMetric()
        self._verbatim = VerbatimMemorizationMetric()
        self._adversarial = AdversarialEvaluator(**(adversarial_kwargs or {}))
        self._relearning = RelearningRobustnessEvaluator(**(relearning_kwargs or {}))

    def verify(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the full verification suite.

        Returns
        -------
        dict
            Nested results for each category, plus top-level:
            - ``verdict``: PASS / PARTIAL / FAIL
            - ``confidence``: float in [0, 1]
            - ``summary``: human-readable summary
            - ``_meta``: timing and configuration info
        """
        t0 = time.time()
        results: Dict[str, Any] = {}
        scores: List[float] = []  # Per-category scores [0, 1]

        # --- Category 1: MIA Suite ---
        if "mia" in self.categories:
            try:
                mia_results = self._mia.compute(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                results["mia"] = mia_results
                # Score: how close mean AUC is to 0.5 (ideal)
                mean_auc = mia_results.get("mia_suite_mean_auc", 0.5)
                mia_score = 1.0 - 2 * abs(mean_auc - 0.5)
                scores.append(max(0.0, mia_score))
            except Exception as e:
                results["mia"] = {"error": str(e)}
                scores.append(0.0)

        # --- Category 2: Memorization Metrics ---
        if "memorization" in self.categories:
            mem_results: Dict[str, Any] = {}
            try:
                es = self._extraction.compute(model, forget_data, retain_data)
                mem_results["extraction_strength"] = es
            except Exception as e:
                mem_results["extraction_strength"] = {"error": str(e)}

            try:
                em = self._exact_mem.compute(model, forget_data, retain_data)
                mem_results["exact_memorization"] = em
            except Exception as e:
                mem_results["exact_memorization"] = {"error": str(e)}

            try:
                vm = self._verbatim.compute(model, forget_data, retain_data)
                mem_results["verbatim_memorization"] = vm
            except Exception as e:
                mem_results["verbatim_memorization"] = {"error": str(e)}

            results["memorization"] = mem_results

            # Score: based on extraction resistance and low memorization
            es_resist = mem_results.get("extraction_strength", {}).get("extraction_resistance", 0.5)
            em_gap = mem_results.get("exact_memorization", {}).get("exact_memorization_gap", 0.0)
            # Higher gap (retain > forget) = better
            mem_score = min(1.0, max(0.0, es_resist * 0.7 + min(em_gap + 0.5, 1.0) * 0.3))
            scores.append(mem_score)

        # --- Category 3: Adversarial Tests ---
        if "adversarial" in self.categories:
            try:
                adv_results = self._adversarial.evaluate(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                results["adversarial"] = adv_results
                overall = adv_results.get("overall", {})
                n_passed = overall.get("tests_passed", 0)
                n_total = overall.get("tests_total", 1)
                scores.append(n_passed / max(n_total, 1))
            except Exception as e:
                results["adversarial"] = {"error": str(e)}
                scores.append(0.0)

        # --- Category 4: Relearning Robustness ---
        if "relearning" in self.categories:
            try:
                relearn_results = self._relearning.evaluate(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                results["relearning"] = relearn_results
                overall = relearn_results.get("overall", {})
                n_passed = overall.get("attacks_passed", 0)
                n_total = overall.get("attacks_total", 1)
                scores.append(n_passed / max(n_total, 1))
            except Exception as e:
                results["relearning"] = {"error": str(e)}
                scores.append(0.0)

        # --- Compute overall verdict ---
        elapsed = time.time() - t0

        if not scores:
            confidence = 0.0
        else:
            confidence = float(sum(scores) / len(scores))

        if self.strict:
            passed = all(s >= 0.9 for s in scores)
        else:
            passed = confidence >= 0.6

        if passed:
            verdict = "PASS"
        elif confidence >= 0.3:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        # Build summary
        summary_parts = []
        if "mia" in results and "error" not in results["mia"]:
            mean_auc = results["mia"].get("mia_suite_mean_auc", "N/A")
            summary_parts.append(f"MIA AUC: {mean_auc:.3f}" if isinstance(mean_auc, float) else f"MIA AUC: {mean_auc}")
        if "adversarial" in results and "error" not in results["adversarial"]:
            adv_overall = results["adversarial"].get("overall", {})
            summary_parts.append(
                f"Adversarial: {adv_overall.get('tests_passed', 0)}/{adv_overall.get('tests_total', 0)} passed"
            )
        if "relearning" in results and "error" not in results["relearning"]:
            rl_overall = results["relearning"].get("overall", {})
            summary_parts.append(
                f"Relearning: {rl_overall.get('attacks_passed', 0)}/{rl_overall.get('attacks_total', 0)} passed"
            )

        results["verdict"] = verdict
        results["confidence"] = confidence
        results["summary"] = " | ".join(summary_parts) if summary_parts else "No tests completed"
        results["_meta"] = {
            "categories_evaluated": self.categories,
            "category_scores": dict(zip(self.categories, scores)),
            "strict_mode": self.strict,
            "elapsed_seconds": elapsed,
        }

        return results
