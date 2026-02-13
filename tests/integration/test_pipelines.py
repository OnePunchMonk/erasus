"""
Integration tests for the full unlearning pipeline with different modalities.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import erasus.strategies  # noqa: F401
import erasus.selectors   # noqa: F401
import erasus.metrics      # noqa: F401


class TinyCLIP(nn.Module):
    """Minimal VLM model."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(32, 4)
        self.visual = True
        self.text_model = True

    def forward(self, x):
        return self.net(x)


class TinyLLM(nn.Module):
    """Minimal LLM model."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(64, 16)
        self.head = nn.Linear(16, 64)
        self.config = type("C", (), {"model_type": "llama", "vocab_size": 64})()

    def forward(self, x):
        return self.head(self.emb(x).mean(dim=1))


class TinyDiffusion(nn.Module):
    """Minimal diffusion model."""
    def __init__(self):
        super().__init__()
        self.unet = nn.Linear(16, 16)
        self.scheduler = True
        self.vae = True

    def forward(self, x):
        return self.unet(x)


@pytest.fixture
def float_data():
    return (
        DataLoader(TensorDataset(torch.randn(16, 32), torch.randint(0, 4, (16,))), batch_size=8),
        DataLoader(TensorDataset(torch.randn(32, 32), torch.randint(0, 4, (32,))), batch_size=8),
    )


@pytest.fixture
def token_data():
    return (
        DataLoader(TensorDataset(torch.randint(0, 64, (16, 8)), torch.randint(0, 64, (16,))), batch_size=8),
        DataLoader(TensorDataset(torch.randint(0, 64, (32, 8)), torch.randint(0, 64, (32,))), batch_size=8),
    )


@pytest.fixture
def latent_data():
    return (
        DataLoader(TensorDataset(torch.randn(16, 16), torch.zeros(16, dtype=torch.long)), batch_size=8),
        DataLoader(TensorDataset(torch.randn(32, 16), torch.zeros(32, dtype=torch.long)), batch_size=8),
    )


class TestCLIPPipeline:
    """Integration tests for CLIP/VLM pipeline."""

    def test_vlm_unlearner_full_pipeline(self, float_data):
        from erasus.unlearners.vlm_unlearner import VLMUnlearner

        model = TinyCLIP()
        forget, retain = float_data

        unlearner = VLMUnlearner(
            model=model, strategy="gradient_ascent",
            selector="random", device="cpu",
        )
        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=1, prune_ratio=0.5)
        assert result.elapsed_time > 0


class TestLLMPipeline:
    """Integration tests for LLM pipeline."""

    def test_llm_unlearner_full_pipeline(self, token_data):
        from erasus.unlearners.llm_unlearner import LLMUnlearner

        model = TinyLLM()
        forget, retain = token_data

        unlearner = LLMUnlearner(
            model=model, strategy="gradient_ascent",
            selector=None, device="cpu",
        )
        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=1)
        assert result.elapsed_time > 0


class TestDiffusionPipeline:
    """Integration tests for Diffusion pipeline."""

    def test_diffusion_unlearner_full_pipeline(self, latent_data):
        from erasus.unlearners.diffusion_unlearner import DiffusionUnlearner

        model = TinyDiffusion()
        forget, retain = latent_data

        unlearner = DiffusionUnlearner(
            model=model, strategy="gradient_ascent",
            selector=None, device="cpu",
        )
        result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=1)
        assert result.elapsed_time > 0


class TestExperimentTracker:
    """Integration tests for experiment tracking."""

    def test_local_tracker(self, tmp_path):
        from erasus.experiments.experiment_tracker import ExperimentTracker

        with ExperimentTracker(
            name="test_run", backend="local",
            output_dir=str(tmp_path / "runs"),
        ) as tracker:
            tracker.log_config({"lr": 0.01, "strategy": "gradient_ascent"})
            tracker.log_metrics({"loss": 1.5, "accuracy": 0.8}, step=1)
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=2)

        # Verify files created
        runs = list((tmp_path / "runs").iterdir())
        assert len(runs) == 1
        run_dir = runs[0]
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "run_info.json").exists()


class TestCertification:
    """Integration tests for certification module."""

    def test_certified_removal_verifier(self):
        from erasus.certification.certified_removal import CertifiedRemovalVerifier

        model_a = nn.Linear(10, 4)
        model_b = nn.Linear(10, 4)

        verifier = CertifiedRemovalVerifier(epsilon=1.0, delta=1e-5)
        result = verifier.verify(
            unlearned_model=model_a,
            retrained_model=model_b,
            n_total=1000,
            n_forget=100,
        )

        assert "removal_bound" in result
        assert "actual_distance" in result
        assert "certified" in result
        assert isinstance(result["certified"], bool)

    def test_unlearning_verifier(self, float_data):
        from erasus.certification.verification import UnlearningVerifier

        model = nn.Linear(32, 4)
        forget, retain = float_data

        verifier = UnlearningVerifier(significance=0.05)
        result = verifier.verify_all(
            model=model,
            forget_loader=forget,
            retain_loader=retain,
        )

        assert "distribution_test" in result
        assert "gradient_residual_test" in result
        assert "prediction_entropy_test" in result
        assert "overall" in result
