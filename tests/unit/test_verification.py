"""
Tests for the unlearning verification features:
- MIA Suite (6-attack battery)
- Memorization metrics (extraction strength, exact memorization, verbatim)
- Adversarial evaluation (cross-prompt leakage, keyword injection, paraphrase)
- Relearning robustness (benign finetuning, quantization, LoRA, prompt extraction)
- Unified verification suite
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal classifier for fast testing."""

    def __init__(self, input_dim: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _make_loader(n: int = 32, dim: int = 16, n_classes: int = 4, bs: int = 8):
    x = torch.randn(n, dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)


@pytest.fixture
def model():
    m = _TinyModel()
    # Train briefly so it has some signal
    loader = _make_loader(64)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()
    for _ in range(5):
        for x, y in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(m(x), y).backward()
            opt.step()
    m.eval()
    return m


@pytest.fixture
def forget_loader():
    return _make_loader(32, dim=16, n_classes=4, bs=8)


@pytest.fixture
def retain_loader():
    return _make_loader(64, dim=16, n_classes=4, bs=8)


# ===================================================================
# MIA Suite
# ===================================================================

class TestMIASuite:
    """Tests for the 6-attack MIA suite."""

    def test_all_attacks_run(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        suite = MIASuite()
        results = suite.compute(model, forget_loader, retain_loader)

        # Should have AUC for each attack
        for atk in MIASuite.ALL_ATTACKS:
            key = f"mia_{atk}_auc"
            assert key in results or f"mia_{atk}_error" in results, f"Missing {key}"

        # Should have aggregate scores
        assert "mia_suite_mean_auc" in results
        assert "mia_suite_worst_auc" in results
        assert "mia_suite_forgetting_quality" in results

    def test_auc_in_valid_range(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        suite = MIASuite()
        results = suite.compute(model, forget_loader, retain_loader)

        for key, value in results.items():
            if "auc" in key and isinstance(value, float):
                assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    def test_subset_attacks(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        suite = MIASuite(attacks=["loss", "zlib"])
        results = suite.compute(model, forget_loader, retain_loader)

        assert "mia_loss_auc" in results
        assert "mia_zlib_auc" in results
        # Should NOT have gradnorm (not requested)
        assert "mia_gradnorm_auc" not in results

    def test_with_reference_model(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        ref = _TinyModel()  # Untrained reference
        suite = MIASuite(attacks=["reference"], reference_model=ref)
        results = suite.compute(model, forget_loader, retain_loader)

        assert "mia_reference_auc" in results

    def test_tpr_at_fpr_reported(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        suite = MIASuite(attacks=["loss"], fpr_thresholds=[0.05])
        results = suite.compute(model, forget_loader, retain_loader)

        assert "mia_loss_tpr@fpr005" in results

    def test_empty_loaders(self, model):
        from erasus.metrics.forgetting.mia_suite import MIASuite

        empty = DataLoader(TensorDataset(torch.empty(0, 16), torch.empty(0, dtype=torch.long)), batch_size=1)
        suite = MIASuite(attacks=["loss"])
        results = suite.compute(model, empty, empty)
        # Should not crash, should return 0.5 AUC
        assert "mia_loss_auc" in results


# ===================================================================
# Memorization Metrics
# ===================================================================

class TestExtractionStrength:
    def test_basic(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.memorization import ExtractionStrengthMetric

        metric = ExtractionStrengthMetric(top_k=1)
        results = metric.compute(model, forget_loader, retain_loader)

        assert "extraction_strength_forget" in results
        assert "extraction_strength_retain" in results
        assert "extraction_strength_gap" in results
        assert "extraction_resistance" in results
        assert 0.0 <= results["extraction_resistance"] <= 1.0

    def test_top_k(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.memorization import ExtractionStrengthMetric

        m1 = ExtractionStrengthMetric(top_k=1).compute(model, forget_loader, retain_loader)
        m2 = ExtractionStrengthMetric(top_k=3).compute(model, forget_loader, retain_loader)

        # top-3 ES should be >= top-1 ES
        assert m2["extraction_strength_forget"] >= m1["extraction_strength_forget"]


class TestExactMemorization:
    def test_basic(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.memorization import ExactMemorizationMetric

        metric = ExactMemorizationMetric(confidence_threshold=0.5)
        results = metric.compute(model, forget_loader, retain_loader)

        assert "exact_memorization_forget" in results
        assert "exact_memorization_retain" in results
        assert "exact_memorization_gap" in results
        assert 0.0 <= results["exact_memorization_forget"] <= 1.0

    def test_high_threshold(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.memorization import ExactMemorizationMetric

        low = ExactMemorizationMetric(confidence_threshold=0.3).compute(model, forget_loader, retain_loader)
        high = ExactMemorizationMetric(confidence_threshold=0.99).compute(model, forget_loader, retain_loader)

        # Higher threshold → fewer samples counted as memorised
        assert high["exact_memorization_forget"] <= low["exact_memorization_forget"]


class TestVerbatimMemorization:
    def test_basic(self, model, forget_loader, retain_loader):
        from erasus.metrics.forgetting.memorization import VerbatimMemorizationMetric

        metric = VerbatimMemorizationMetric()
        results = metric.compute(model, forget_loader, retain_loader)

        assert "verbatim_kl_forget" in results
        assert "verbatim_entropy_forget" in results
        assert "verbatim_memorization_forget" in results
        assert "verbatim_kl_retain" in results


# ===================================================================
# Adversarial Evaluation
# ===================================================================

class TestCrossPromptLeakage:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import CrossPromptLeakageTest

        test = CrossPromptLeakageTest(n_pairs=10)
        result = test.run(model, forget_loader, retain_loader)

        assert "test" in result
        assert result["test"] == "cross_prompt_leakage"
        assert "leakage_rate" in result
        assert "passed" in result
        assert 0.0 <= result["leakage_rate"] <= 1.0

    def test_custom_threshold(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import CrossPromptLeakageTest

        strict = CrossPromptLeakageTest(change_threshold=0.001)
        result = strict.run(model, forget_loader, retain_loader)
        assert "leakage_rate" in result


class TestKeywordInjection:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import KeywordInjectionTest

        test = KeywordInjectionTest(injection_strengths=[0.05, 0.1])
        result = test.run(model, forget_loader, retain_loader)

        assert "test" in result
        assert result["test"] == "keyword_injection"
        assert "baseline_retain_accuracy" in result
        assert "worst_accuracy_drop" in result
        assert "passed" in result

    def test_zero_injection_no_change(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import KeywordInjectionTest

        test = KeywordInjectionTest(injection_strengths=[0.0])
        result = test.run(model, forget_loader, retain_loader)
        # Zero injection should have approximately zero drop
        assert abs(result.get("injection_alpha_0.00_drop", 0)) < 0.05


class TestParaphraseRobustness:
    def test_runs(self, model, forget_loader):
        from erasus.evaluation.adversarial import ParaphraseRobustnessTest

        test = ParaphraseRobustnessTest(noise_levels=[0.01], n_perturbations=2)
        result = test.run(model, forget_loader)

        assert "test" in result
        assert result["test"] == "paraphrase_robustness"
        assert "baseline_forget_accuracy" in result
        assert "max_accuracy_recovery" in result
        assert "permutation_accuracy" in result
        assert "scaling_accuracy" in result
        assert "passed" in result


class TestAdversarialEvaluator:
    def test_runs_all(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import AdversarialEvaluator

        evaluator = AdversarialEvaluator()
        report = evaluator.evaluate(model, forget_loader, retain_loader)

        assert "cross_prompt" in report
        assert "keyword_injection" in report
        assert "paraphrase" in report
        assert "overall" in report
        assert "tests_passed" in report["overall"]
        assert "verdict" in report["overall"]

    def test_subset(self, model, forget_loader, retain_loader):
        from erasus.evaluation.adversarial import AdversarialEvaluator

        evaluator = AdversarialEvaluator(tests=["cross_prompt"])
        report = evaluator.evaluate(model, forget_loader, retain_loader)

        assert "cross_prompt" in report
        assert "keyword_injection" not in report


# ===================================================================
# Relearning Robustness
# ===================================================================

class TestBenignFinetuning:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import BenignFinetuningAttack

        attack = BenignFinetuningAttack(epochs=1, lr=1e-3)
        result = attack.run(model, forget_loader, retain_loader)

        assert result["test"] == "benign_finetuning"
        assert "pre_forget_accuracy" in result
        assert "post_forget_accuracy" in result
        assert "forget_accuracy_recovery" in result
        assert "passed" in result

    def test_does_not_modify_original(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import BenignFinetuningAttack

        # Snapshot original params
        original_params = {n: p.clone() for n, p in model.named_parameters()}

        attack = BenignFinetuningAttack(epochs=1)
        attack.run(model, forget_loader, retain_loader)

        # Original model should be unchanged
        for n, p in model.named_parameters():
            assert torch.equal(p, original_params[n]), f"Original model param {n} was modified"


class TestQuantization:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import QuantizationAttack

        attack = QuantizationAttack(bit_widths=[8])
        result = attack.run(model, forget_loader, retain_loader)

        assert result["test"] == "quantization"
        assert "quant_8bit_forget_accuracy" in result
        assert "worst_recovery" in result
        assert "passed" in result

    def test_multiple_bit_widths(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import QuantizationAttack

        attack = QuantizationAttack(bit_widths=[8, 4])
        result = attack.run(model, forget_loader, retain_loader)

        assert "quant_8bit_forget_accuracy" in result
        assert "quant_4bit_forget_accuracy" in result


class TestLoRARelearning:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import LoRARelearningAttack

        attack = LoRARelearningAttack(rank=2, epochs=1, lr=1e-3)
        result = attack.run(model, forget_loader, retain_loader)

        assert result["test"] == "lora_relearning"
        assert "lora_params" in result
        assert "forget_accuracy_recovery" in result
        assert "passed" in result

    def test_does_not_modify_original(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import LoRARelearningAttack

        original_params = {n: p.clone() for n, p in model.named_parameters()}

        attack = LoRARelearningAttack(rank=2, epochs=1)
        attack.run(model, forget_loader, retain_loader)

        for n, p in model.named_parameters():
            assert torch.equal(p, original_params[n]), f"Original model param {n} was modified"


class TestPromptExtraction:
    def test_runs(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import PromptExtractionAttack

        attack = PromptExtractionAttack(n_pgd_steps=3, epsilon=0.05)
        result = attack.run(model, forget_loader, retain_loader)

        assert result["test"] == "prompt_extraction"
        assert "pgd_accuracy" in result
        assert "interpolation_accuracy" in result
        assert "amplification_accuracy" in result
        assert "worst_recovery" in result
        assert "passed" in result


class TestRelearningEvaluator:
    def test_runs_all(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import RelearningRobustnessEvaluator

        evaluator = RelearningRobustnessEvaluator(
            benign_finetuning_kwargs={"epochs": 1},
            lora_relearning_kwargs={"rank": 2, "epochs": 1},
            prompt_extraction_kwargs={"n_pgd_steps": 2},
        )
        report = evaluator.evaluate(model, forget_loader, retain_loader)

        assert "benign_finetuning" in report
        assert "quantization" in report
        assert "lora_relearning" in report
        assert "prompt_extraction" in report
        assert "overall" in report
        assert "verdict" in report["overall"]

    def test_subset(self, model, forget_loader, retain_loader):
        from erasus.evaluation.relearning import RelearningRobustnessEvaluator

        evaluator = RelearningRobustnessEvaluator(attacks=["quantization"])
        report = evaluator.evaluate(model, forget_loader, retain_loader)

        assert "quantization" in report
        assert "benign_finetuning" not in report


# ===================================================================
# Unified Verification Suite
# ===================================================================

class TestVerificationSuite:
    def test_runs_all_categories(self, model, forget_loader, retain_loader):
        from erasus.evaluation.verification_suite import UnlearningVerificationSuite

        suite = UnlearningVerificationSuite(
            relearning_kwargs={
                "benign_finetuning_kwargs": {"epochs": 1},
                "lora_relearning_kwargs": {"rank": 2, "epochs": 1},
                "prompt_extraction_kwargs": {"n_pgd_steps": 2},
            },
        )
        report = suite.verify(model, forget_loader, retain_loader)

        assert "verdict" in report
        assert report["verdict"] in ("PASS", "PARTIAL", "FAIL")
        assert "confidence" in report
        assert 0.0 <= report["confidence"] <= 1.0
        assert "summary" in report
        assert "_meta" in report
        assert "mia" in report
        assert "memorization" in report
        assert "adversarial" in report
        assert "relearning" in report

    def test_subset_categories(self, model, forget_loader, retain_loader):
        from erasus.evaluation.verification_suite import UnlearningVerificationSuite

        suite = UnlearningVerificationSuite(categories=["mia", "memorization"])
        report = suite.verify(model, forget_loader, retain_loader)

        assert "mia" in report
        assert "memorization" in report
        assert "adversarial" not in report
        assert "relearning" not in report

    def test_strict_mode(self, model, forget_loader, retain_loader):
        from erasus.evaluation.verification_suite import UnlearningVerificationSuite

        suite = UnlearningVerificationSuite(
            categories=["mia"],
            strict=True,
        )
        report = suite.verify(model, forget_loader, retain_loader)

        # Strict mode requires all scores >= 0.9
        assert report["verdict"] in ("PASS", "PARTIAL", "FAIL")

    def test_meta_has_timing(self, model, forget_loader, retain_loader):
        from erasus.evaluation.verification_suite import UnlearningVerificationSuite

        suite = UnlearningVerificationSuite(categories=["mia"])
        report = suite.verify(model, forget_loader, retain_loader)

        assert report["_meta"]["elapsed_seconds"] > 0
        assert "mia" in report["_meta"]["category_scores"]


# ===================================================================
# Registry integration
# ===================================================================

class TestRegistryIntegration:
    def test_new_metrics_registered(self):
        from erasus.core.registry import metric_registry

        for name in ["mia_suite", "extraction_strength", "exact_memorization", "verbatim_memorization"]:
            cls = metric_registry.get(name)
            assert cls is not None, f"{name} not registered"

    def test_mia_suite_via_registry(self, model, forget_loader, retain_loader):
        from erasus.core.registry import metric_registry

        cls = metric_registry.get("mia_suite")
        instance = cls(attacks=["loss"])
        results = instance.compute(model, forget_loader, retain_loader)
        assert "mia_loss_auc" in results

    def test_imports(self):
        from erasus.metrics import MIASuite, ExtractionStrengthMetric, ExactMemorizationMetric, VerbatimMemorizationMetric
        from erasus.evaluation import (
            AdversarialEvaluator,
            CrossPromptLeakageTest,
            KeywordInjectionTest,
            ParaphraseRobustnessTest,
            RelearningRobustnessEvaluator,
            BenignFinetuningAttack,
            QuantizationAttack,
            LoRARelearningAttack,
            PromptExtractionAttack,
            UnlearningVerificationSuite,
        )
        # All imports succeeded
        assert MIASuite is not None
        assert AdversarialEvaluator is not None
        assert UnlearningVerificationSuite is not None
