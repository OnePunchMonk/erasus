"""
Tests for benchmark suite — TOFU, lm-eval integration, and runners.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.benchmarks import (
    TOFUDataset,
    TOFULoader,
    TOFUEvaluator,
    LMEvalWrapper,
    LMEvalBenchmark,
    PostUnlearningBenchmarkSuite,
    BenchmarkRunner,
    BenchmarkResult,
)


@pytest.fixture
def tiny_model():
    """Simple language model for testing."""

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 100)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embedding(input_ids)
            logits = self.linear(x)

            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, 100), labels.view(-1), reduction="mean"
                )
                return type("Output", (), {"loss": loss, "logits": logits})()

            return type("Output", (), {"logits": logits})()

    return TinyLM()


@pytest.fixture
def dummy_tokenizer():
    """Dummy tokenizer for testing."""

    class DummyTokenizer:
        def __init__(self):
            self.vocab_size = 100

        def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None, **kwargs):
            # Simple: hash text to token IDs
            tokens = [abs(hash(text)) % 100 for _ in range(5)]
            if max_length:
                if len(tokens) < max_length:
                    tokens = tokens + [0] * (max_length - len(tokens))
                else:
                    tokens = tokens[:max_length]

            result = {
                "input_ids": torch.tensor([tokens]),
                "attention_mask": torch.ones(1, len(tokens)),
            }
            return result

        def encode(self, text, return_tensors=None, **kwargs):
            # Simple: hash text to token IDs
            tokens = [abs(hash(text)) % 100 for _ in range(5)]
            if return_tensors == "pt":
                return torch.tensor([tokens])
            return tokens

        def decode(self, tokens, **kwargs):
            return f"decoded_text_{len(tokens)}"

    return DummyTokenizer()


class TestTOFUDataset:
    """Test TOFU dataset class."""

    def test_creation(self):
        """Test creating TOFU dataset."""
        data = [
            {"question": "Who is author A?", "answer": "An author"},
            {"question": "Who is author B?", "answer": "Another author"},
        ]

        dataset = TOFUDataset(data)

        assert len(dataset) == 2

    def test_getitem(self):
        """Test accessing dataset items."""
        data = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]

        dataset = TOFUDataset(data)
        item = dataset[0]

        assert "text" in item
        assert "question" in item
        assert "answer" in item
        assert item["question"] == "Q1?"
        assert item["answer"] == "A1"

    def test_with_tokenizer(self, dummy_tokenizer):
        """Test dataset with tokenizer."""
        data = [{"question": "Q?", "answer": "A"}]

        dataset = TOFUDataset(data, tokenizer=dummy_tokenizer, max_length=10)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape[0] == 10  # padded to max_length


class TestTOFULoader:
    """Test TOFU dataset loader."""

    def test_init(self):
        """Test initialization."""
        loader = TOFULoader(batch_size=8)

        assert loader.batch_size == 8

    def test_load_synthetic(self):
        """Test loading synthetic TOFU data."""
        loader = TOFULoader()

        forget_loader, retain_loader, eval_loader = loader.load_synthetic_tofu(
            num_forget=16,
            num_retain=16,
            num_eval=8,
        )

        assert isinstance(forget_loader, DataLoader)
        assert isinstance(retain_loader, DataLoader)
        assert isinstance(eval_loader, DataLoader)

        # Check sizes
        forget_size = sum(1 for _ in forget_loader)
        retain_size = sum(1 for _ in retain_loader)
        eval_size = sum(1 for _ in eval_loader)

        assert forget_size > 0
        assert retain_size > 0
        assert eval_size > 0

    def test_synthetic_data_content(self):
        """Test that synthetic data has correct structure."""
        loader = TOFULoader()

        forget_loader, _, _ = loader.load_synthetic_tofu(num_forget=4)

        for batch in forget_loader:
            assert "text" in batch
            assert "question" in batch
            assert "answer" in batch
            break


class TestTOFUEvaluator:
    """Test TOFU evaluation."""

    def test_init(self, tiny_model):
        """Test evaluator initialization."""
        evaluator = TOFUEvaluator(tiny_model, device="cpu")

        assert evaluator.model is not None

    def test_measure_utility(self, tiny_model, dummy_tokenizer):
        """Test utility measurement."""
        evaluator = TOFUEvaluator(tiny_model, device="cpu")

        # Create simple loader
        loader = TOFULoader(tokenizer=dummy_tokenizer, max_length=5)
        forget_loader, _, _ = loader.load_synthetic_tofu(num_forget=8)

        utility = evaluator.evaluate_on_loader(forget_loader, metric_type="loss")

        assert isinstance(utility, float)
        assert utility >= 0.0

    def test_compute_metrics(self, tiny_model, dummy_tokenizer):
        """Test unlearning metric computation."""
        evaluator = TOFUEvaluator(tiny_model, device="cpu")

        loader = TOFULoader(tokenizer=dummy_tokenizer, max_length=5)
        forget_loader, retain_loader, _ = loader.load_synthetic_tofu(
            num_forget=8,
            num_retain=8,
        )

        metrics = evaluator.compute_unlearning_metrics(forget_loader, retain_loader)

        assert "forget_loss" in metrics
        assert "retain_loss" in metrics
        assert "forget_perplexity" in metrics
        assert "retain_perplexity" in metrics
        assert "forget_effectiveness" in metrics


class TestLMEvalWrapper:
    """Test lm-eval wrapper."""

    def test_init(self, tiny_model, dummy_tokenizer):
        """Test wrapper initialization."""
        wrapper = LMEvalWrapper(tiny_model, dummy_tokenizer, device="cpu")

        assert wrapper.model is not None
        assert wrapper.tokenizer is not None

    def test_encode(self, tiny_model, dummy_tokenizer):
        """Test text encoding."""
        wrapper = LMEvalWrapper(tiny_model, dummy_tokenizer)

        tokens = wrapper._encode("test text")

        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape[0] == 1  # batch size


class TestLMEvalBenchmark:
    """Test lm-eval benchmark runner."""

    def test_init(self, tiny_model, dummy_tokenizer):
        """Test benchmark initialization."""
        benchmark = LMEvalBenchmark(tiny_model, dummy_tokenizer)

        assert benchmark.wrapper is not None

    def test_benchmark_creation(self, tiny_model, dummy_tokenizer):
        """Test that benchmark can be instantiated."""
        benchmark = LMEvalBenchmark(tiny_model, dummy_tokenizer, device="cpu")

        # Just check it doesn't crash on instantiation
        assert benchmark.wrapper is not None


class TestPostUnlearningBenchmarkSuite:
    """Test the post-unlearning wrapper for standard tasks."""

    def test_init(self, tiny_model, dummy_tokenizer):
        suite = PostUnlearningBenchmarkSuite(tiny_model, dummy_tokenizer)
        assert suite.benchmark is not None

    def test_default_tasks(self):
        assert PostUnlearningBenchmarkSuite.DEFAULT_TASKS == [
            "mmlu",
            "gsm8k",
            "truthfulqa",
            "hellaswag",
            "arc",
        ]


class TestBenchmarkRunner:
    """Test end-to-end benchmark runner."""

    def test_init(self, tiny_model, dummy_tokenizer):
        """Test runner initialization."""
        runner = BenchmarkRunner(tiny_model, dummy_tokenizer)

        assert runner.model is not None
        assert runner.tokenizer is not None

    def test_compute_degradation(self, tiny_model, dummy_tokenizer):
        """Test degradation computation."""
        runner = BenchmarkRunner(tiny_model, dummy_tokenizer)

        baseline = {"retain_loss": 2.0}
        unlearned = {"retain_loss": 2.5}

        degradation = runner._compute_degradation(baseline, unlearned)

        assert degradation >= 0.0
        assert degradation > 0.0  # Should show some degradation

    def test_degradation_no_loss(self, tiny_model, dummy_tokenizer):
        """Test degradation when loss is missing."""
        runner = BenchmarkRunner(tiny_model, dummy_tokenizer)

        baseline = {"other_metric": 1.0}
        unlearned = {"other_metric": 1.1}

        degradation = runner._compute_degradation(baseline, unlearned)

        assert degradation == 0.0


class TestBenchmarkResult:
    """Test benchmark result dataclass."""

    def test_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            strategy_name="test_strategy",
            tofu_metrics={"forget_loss": 0.5, "retain_loss": 0.3},
            utility_degradation=0.1,
            total_time=10.5,
        )

        assert result.strategy_name == "test_strategy"
        assert result.utility_degradation == 0.1
        assert result.total_time == 10.5
        assert len(result.tofu_metrics) == 2

    def test_metadata(self):
        """Test result with metadata."""
        metadata = {"model": "gpt-2", "dataset": "tofu"}
        result = BenchmarkResult(
            strategy_name="npo",
            metadata=metadata,
        )

        assert result.metadata == metadata


class TestBenchmarkIntegration:
    """Integration tests for benchmark suite."""

    def test_tofu_pipeline(self, dummy_tokenizer):
        """Test TOFU loading and evaluation pipeline."""
        # Create model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Load TOFU
        loader = TOFULoader()
        forget_loader, retain_loader, _ = loader.load_synthetic_tofu(num_forget=8)

        # Evaluate
        evaluator = TOFUEvaluator(model)
        metrics = evaluator.compute_unlearning_metrics(forget_loader, retain_loader)

        assert "forget_loss" in metrics
        assert isinstance(metrics["forget_loss"], float)

    def test_multiple_loaders(self):
        """Test creating multiple dataset loaders."""
        loader = TOFULoader(batch_size=4)

        # Load different sizes
        f1, r1, e1 = loader.load_synthetic_tofu(num_forget=8, num_retain=8)
        f2, r2, e2 = loader.load_synthetic_tofu(num_forget=16, num_retain=16)

        # Check they're different
        assert len(list(f1)) != len(list(f2))


class TestRealBenchmarkEntrypoints:
    """Test importable real benchmark entrypoints with fake local data."""

    @staticmethod
    def _load_module(path: str, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_wmdp_run_real_with_fake_scorer(self, tmp_path):
        module = self._load_module(
            "/Users/avaya.aggarwal@zomato.com/erasus/benchmarks/wmdp/run_real.py",
            "wmdp_run_real_test",
        )

        data_dir = tmp_path / "wmdp"
        data_dir.mkdir()
        sample_file = data_dir / "wmdp_bio.json"
        sample_file.write_text(json.dumps([
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
            }
        ]), encoding="utf-8")

        class FakeMCQModel:
            name_or_path = "fake-zephyr"

            def score_choices(self, prompt, choices):
                return [0.0, 10.0, 0.0, 0.0]

        results = module.run_real_wmdp(
            subset="bio",
            data_dir=str(data_dir),
            model=FakeMCQModel(),
            tokenizer=object(),
            max_samples=1,
        )

        assert results["benchmark"] == "wmdp_real"
        assert results["accuracy"] == 1.0

    def test_tofu_run_real_on_local_split_files(self, tmp_path):
        module = self._load_module(
            "/Users/avaya.aggarwal@zomato.com/erasus/benchmarks/tofu/run_real.py",
            "tofu_run_real_test",
        )

        data_dir = tmp_path / "tofu"
        data_dir.mkdir(parents=True)
        (data_dir / "forget_01.json").write_text(
            json.dumps([{"question": "Who is A?", "answer": "Author A"}]),
            encoding="utf-8",
        )
        (data_dir / "retain.json").write_text(
            json.dumps([{"question": "Who is B?", "answer": "Author B"}]),
            encoding="utf-8",
        )

        class FakeTokenizer:
            def __call__(self, text, max_length=None, truncation=None, padding=None, return_tensors=None):
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.ones(1, 4),
                }

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8, device=input_ids.device)
                loss = torch.tensor(1.0, device=input_ids.device)
                return type("Output", (), {"loss": loss, "logits": logits})()

        results = module.run_real_tofu(
            data_dir=str(data_dir),
            model=FakeLM(),
            tokenizer=FakeTokenizer(),
            batch_size=1,
            max_length=8,
        )

        assert results["benchmark"] == "tofu_real"
        assert "forget_loss" in results
        assert "retain_loss" in results

    def test_muse_run_real_on_local_split_files(self, tmp_path):
        module = self._load_module(
            "/Users/avaya.aggarwal@zomato.com/erasus/benchmarks/muse/run_real.py",
            "muse_run_real_test",
        )

        data_dir = tmp_path / "muse" / "news"
        data_dir.mkdir(parents=True)
        for split in ("forget", "retain", "holdout", "test"):
            (data_dir / f"{split}.json").write_text(
                json.dumps([{"text": f"{split} sample text"}]),
                encoding="utf-8",
            )

        class FakeTokenizer:
            def __call__(self, text, max_length=None, truncation=None, padding=None, return_tensors=None):
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.ones(1, 4),
                }

        class FakeLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids=None, attention_mask=None, labels=None):
                logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8, device=input_ids.device)
                loss = torch.tensor(1.0, device=input_ids.device)
                return type("Output", (), {"loss": loss, "logits": logits})()

        results = module.run_real_muse(
            subset="news",
            data_dir=str(tmp_path / "muse"),
            model=FakeLM(),
            tokenizer=FakeTokenizer(),
            batch_size=1,
            max_length=8,
        )

        assert results["benchmark"] == "muse_real"
        assert "six_way" in results
        assert set(results["six_way"]) == {
            "forget_quality",
            "model_utility",
            "privacy_leakage",
            "knowledge_retention",
            "consistency",
            "efficiency",
        }
