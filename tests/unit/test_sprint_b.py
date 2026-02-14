"""
Sprint B — Comprehensive unit tests.

Tests all new Sprint B components:
- Models: Flamingo, T5, DALL-E, Imagen, diffusion_utils, Wav2Vec, CLAP, VideoCLIP, ViT utils
- Data: MUSE, ImageNet, augmentation, bias_generator, privacy_generator
- Privacy: gradient_clipping, secure_aggregation
- Utils: profiling, reproducibility
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_model():
    """A small MLP for testing."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


@pytest.fixture
def simple_cnn():
    """A small CNN for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )


# ======================================================================
# 1. Diffusion Utilities
# ======================================================================


class TestDiffusionUtils:
    def test_linear_beta_schedule(self):
        from erasus.models.diffusion.diffusion_utils import linear_beta_schedule

        betas = linear_beta_schedule(100)
        assert betas.shape == (100,)
        assert betas[0] < betas[-1]
        assert (betas > 0).all()

    def test_cosine_beta_schedule(self):
        from erasus.models.diffusion.diffusion_utils import cosine_beta_schedule

        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)
        assert (betas >= 0).all()
        assert (betas < 1).all()

    def test_sqrt_beta_schedule(self):
        from erasus.models.diffusion.diffusion_utils import sqrt_beta_schedule

        betas = sqrt_beta_schedule(100)
        assert betas.shape == (100,)
        assert (betas > 0).all()

    def test_get_noise_schedule(self):
        from erasus.models.diffusion.diffusion_utils import get_noise_schedule

        for name in ["linear", "cosine", "sqrt"]:
            betas = get_noise_schedule(name, 100)
            assert betas.shape == (100,)

        with pytest.raises(ValueError):
            get_noise_schedule("unknown")

    def test_compute_alphas(self):
        from erasus.models.diffusion.diffusion_utils import (
            linear_beta_schedule,
            compute_alphas,
        )

        betas = linear_beta_schedule(100)
        alphas, alpha_cumprod = compute_alphas(betas)
        assert alphas.shape == (100,)
        assert alpha_cumprod.shape == (100,)
        assert alpha_cumprod[0] > alpha_cumprod[-1]

    def test_signal_to_noise_ratio(self):
        from erasus.models.diffusion.diffusion_utils import (
            linear_beta_schedule,
            compute_alphas,
            signal_to_noise_ratio,
            log_snr,
        )

        betas = linear_beta_schedule(100)
        _, alpha_cumprod = compute_alphas(betas)
        snr = signal_to_noise_ratio(alpha_cumprod)
        assert snr.shape == (100,)
        assert snr[0] > snr[-1]  # SNR decreases with noise

        lsnr = log_snr(alpha_cumprod)
        assert lsnr.shape == (100,)

    def test_uniform_timestep_sample(self):
        from erasus.models.diffusion.diffusion_utils import uniform_timestep_sample

        t = uniform_timestep_sample(8, 100)
        assert t.shape == (8,)
        assert (t >= 0).all() and (t < 100).all()

    def test_add_noise(self):
        from erasus.models.diffusion.diffusion_utils import (
            linear_beta_schedule,
            compute_alphas,
            add_noise,
        )

        betas = linear_beta_schedule(100)
        _, alpha_cumprod = compute_alphas(betas)

        x_0 = torch.randn(4, 3, 8, 8)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0, 25, 50, 99])

        x_t = add_noise(x_0, noise, t, alpha_cumprod)
        assert x_t.shape == x_0.shape

    def test_predict_x0_from_noise(self):
        from erasus.models.diffusion.diffusion_utils import (
            linear_beta_schedule,
            compute_alphas,
            add_noise,
            predict_x0_from_noise,
        )

        betas = linear_beta_schedule(100)
        _, alpha_cumprod = compute_alphas(betas)

        x_0 = torch.randn(2, 3, 8, 8)
        noise = torch.randn_like(x_0)
        t = torch.tensor([10, 50])

        x_t = add_noise(x_0, noise, t, alpha_cumprod)
        x_0_pred = predict_x0_from_noise(x_t, noise, t, alpha_cumprod)

        # Should reconstruct perfectly with true noise
        assert torch.allclose(x_0, x_0_pred, atol=1e-5)

    def test_encode_decode_latent(self):
        from erasus.models.diffusion.diffusion_utils import (
            encode_to_latent,
            decode_from_latent,
        )
        # Test with a dummy VAE
        class DummyVAE(nn.Module):
            def encode(self, x):
                class Dist:
                    def sample(self):
                        return x * 0.5
                class Out:
                    latent_dist = Dist()
                return Out()
            def decode(self, z):
                class Out:
                    sample = z * 2
                return Out()

        vae = DummyVAE()
        images = torch.randn(2, 3, 64, 64)

        latent = encode_to_latent(vae, images)
        assert latent.shape == images.shape

        decoded = decode_from_latent(vae, latent)
        assert decoded.shape == images.shape


# ======================================================================
# 2. ViT Feature Extractor
# ======================================================================


class TestViTFeatureExtractor:
    def test_cls_pool(self):
        from erasus.models.vlm.vision_transformer import ViTFeatureExtractor

        hidden = torch.randn(2, 197, 768)  # 196 patches + CLS
        cls = ViTFeatureExtractor.cls_pool(hidden)
        assert cls.shape == (2, 768)

    def test_mean_pool(self):
        from erasus.models.vlm.vision_transformer import ViTFeatureExtractor

        hidden = torch.randn(2, 197, 768)
        pooled = ViTFeatureExtractor.mean_pool(hidden)
        assert pooled.shape == (2, 768)

    def test_spatial_pool(self):
        from erasus.models.vlm.vision_transformer import ViTFeatureExtractor

        hidden = torch.randn(2, 197, 768)  # 14x14 = 196 patches + CLS
        spatial = ViTFeatureExtractor.spatial_pool(hidden, grid_size=(14, 14))
        assert spatial.shape == (2, 768, 14, 14)

    def test_rollout_attention(self):
        from erasus.models.vlm.vision_transformer import ViTFeatureExtractor

        # Simulate 3 layers of attention
        attn_maps = [
            torch.softmax(torch.randn(2, 4, 10, 10), dim=-1)
            for _ in range(3)
        ]
        rollout = ViTFeatureExtractor.rollout_attention(attn_maps)
        assert rollout.shape == (2, 10, 10)
        # Rows should sum to ~1
        assert torch.allclose(rollout.sum(-1), torch.ones(2, 10), atol=0.1)

    def test_compute_patch_importance(self):
        from erasus.models.vlm.vision_transformer import compute_patch_importance

        hidden = torch.randn(2, 197, 768)
        importance = compute_patch_importance(hidden)
        assert importance.shape == (2, 196)


# ======================================================================
# 3. Data: MUSE Dataset
# ======================================================================


class TestMUSEDataset:
    def test_init(self, tmp_path):
        from erasus.data.datasets.muse import MUSEDataset

        ds = MUSEDataset(data_dir=str(tmp_path), split="forget", subset="news")
        # No data found → empty
        assert len(ds) == 0

    def test_invalid_split(self, tmp_path):
        from erasus.data.datasets.muse import MUSEDataset

        with pytest.raises(ValueError, match="Invalid split"):
            MUSEDataset(data_dir=str(tmp_path), split="invalid")

    def test_invalid_subset(self, tmp_path):
        from erasus.data.datasets.muse import MUSEDataset

        with pytest.raises(ValueError, match="Invalid subset"):
            MUSEDataset(data_dir=str(tmp_path), subset="invalid")

    def test_with_local_data(self, tmp_path):
        import json

        from erasus.data.datasets.muse import MUSEDataset

        # Create fake data
        news_dir = tmp_path / "news"
        news_dir.mkdir()
        samples = [
            {"text": "Some news article about AI."},
            {"text": "Another news article."},
        ]
        with open(news_dir / "forget.json", "w") as f:
            json.dump(samples, f)

        ds = MUSEDataset(data_dir=str(tmp_path), split="forget", subset="news")
        assert len(ds) == 2
        item = ds[0]
        assert "text" in item
        assert "AI" in item["text"]

    def test_get_forget_retain_split(self, tmp_path):
        from erasus.data.datasets.muse import MUSEDataset

        forget, retain = MUSEDataset.get_forget_retain_split(
            data_dir=str(tmp_path), subset="news",
        )
        assert isinstance(forget, MUSEDataset)
        assert isinstance(retain, MUSEDataset)


# ======================================================================
# 4. Data: ImageNet Dataset
# ======================================================================


class TestImageNetDataset:
    def test_init_empty(self, tmp_path):
        from erasus.data.datasets.imagenet import ImageNetDataset

        ds = ImageNetDataset(data_dir=str(tmp_path))
        assert len(ds) == 0

    def test_forget_retain_split(self, tmp_path):
        from erasus.data.datasets.imagenet import ImageNetDataset

        ds = ImageNetDataset(
            data_dir=str(tmp_path),
            forget_classes=[0, 1],
        )
        forget_idx, retain_idx = ds.get_forget_retain_split()
        # Both should be lists
        assert isinstance(forget_idx, list)
        assert isinstance(retain_idx, list)


# ======================================================================
# 5. Data: Augmentation
# ======================================================================


class TestAugmentation:
    def test_identity(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="image", forget_strategy="identity")
        data = torch.randn(4, 3, 32, 32)
        out = aug.augment_forget(data)
        assert torch.equal(data, out)

    def test_mild_augment(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="image", retain_strategy="mild")
        data = torch.randn(4, 3, 32, 32)
        out = aug.augment_retain(data)
        assert out.shape == data.shape

    def test_strong_augment(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="image", forget_strategy="strong")
        data = torch.randn(4, 3, 32, 32)
        out = aug.augment_forget(data)
        assert out.shape == data.shape

    def test_mix_augment(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="image", forget_strategy="mix")
        data = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        result = aug.augment_forget(data, labels)
        # MixUp returns (mixed_data, labels, shuffled_labels, lambda)
        assert len(result) == 4

    def test_text_augment(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="text", forget_strategy="strong")
        data = torch.randint(0, 100, (4, 64))
        out = aug.augment_forget(data)
        assert out.shape == data.shape

    def test_audio_augment(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="audio", retain_strategy="mild")
        data = torch.randn(4, 16000)
        out = aug.augment_retain(data)
        assert out.shape == data.shape

    def test_get_unlearning_augmentation(self):
        from erasus.data.augmentation import get_unlearning_augmentation

        for preset in ["default", "aggressive", "gentle", "none"]:
            aug = get_unlearning_augmentation(modality="image", preset=preset)
            assert aug is not None

    def test_callable(self):
        from erasus.data.augmentation import UnlearningAugmentation

        aug = UnlearningAugmentation(modality="image")
        data = torch.randn(4, 3, 32, 32)

        # Test as callable
        out_retain = aug(data, is_forget=False)
        out_forget = aug(data, is_forget=True)
        assert out_retain.shape == data.shape
        assert out_forget.shape == data.shape


# ======================================================================
# 6. Data: Bias Generator
# ======================================================================


class TestBiasGenerator:
    def test_label_bias(self):
        from erasus.data.synthetic.bias_generator import BiasGenerator

        gen = BiasGenerator(n_protected_groups=2, bias_strength=0.9, bias_type="label")
        data, labels = BiasGenerator.make_clean_dataset(200, 32, 4)
        biased_ds, groups, stats = gen.generate(data, labels)
        assert len(biased_ds) == 200
        assert groups.shape == (200,)
        assert "demographic_parity_gap" in stats

    def test_feature_bias(self):
        from erasus.data.synthetic.bias_generator import BiasGenerator

        gen = BiasGenerator(bias_type="feature")
        data, labels = BiasGenerator.make_clean_dataset(100, 64)
        biased_ds, groups, stats = gen.generate(data, labels)
        assert len(biased_ds) == 100

    def test_representation_bias(self):
        from erasus.data.synthetic.bias_generator import BiasGenerator

        gen = BiasGenerator(bias_type="representation")
        data, labels = BiasGenerator.make_clean_dataset(100, 64)
        biased_ds, groups, stats = gen.generate(data, labels)
        assert len(biased_ds) == 100

    def test_invalid_bias_type(self):
        from erasus.data.synthetic.bias_generator import BiasGenerator

        with pytest.raises(ValueError):
            BiasGenerator(bias_type="invalid")


# ======================================================================
# 7. Data: Privacy Generator
# ======================================================================


class TestPrivacyGenerator:
    def test_pii_generation(self):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        gen = PrivacyDataGenerator(data_type="pii", n_private_samples=50, embedding_dim=64)
        private_ds, public_ds, metadata = gen.generate(n_public_samples=200)
        assert len(private_ds) == 50
        assert len(public_ds) == 200
        assert metadata["data_type"] == "pii"

    def test_medical_generation(self):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        gen = PrivacyDataGenerator(data_type="medical", n_private_samples=30)
        private_ds, public_ds, metadata = gen.generate()
        assert len(private_ds) == 30

    def test_financial_generation(self):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        gen = PrivacyDataGenerator(data_type="financial", n_private_samples=40)
        private_ds, public_ds, metadata = gen.generate()
        assert len(private_ds) == 40

    def test_mixed_generation(self):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        gen = PrivacyDataGenerator(data_type="mixed", n_private_samples=60)
        private_ds, public_ds, metadata = gen.generate()
        assert len(private_ds) == 60

    def test_invalid_type(self):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        with pytest.raises(ValueError):
            PrivacyDataGenerator(data_type="invalid")

    def test_memorization_score(self, simple_model):
        from erasus.data.synthetic.privacy_generator import PrivacyDataGenerator

        gen = PrivacyDataGenerator(n_private_samples=10, embedding_dim=16)
        private_ds, public_ds, _ = gen.generate(n_public_samples=20, num_classes=10)

        scores = gen.compute_memorization_score(
            simple_model,
            private_ds.tensors[0],
            public_ds.tensors[0],
        )
        assert "memorization_score" in scores
        assert "private_loss" in scores
        assert "public_loss" in scores


# ======================================================================
# 8. Privacy: Gradient Clipping
# ======================================================================


class TestGradientClipping:
    def test_flat_clip(self, simple_model):
        from erasus.privacy.gradient_clipping import GradientClipper

        clipper = GradientClipper(max_grad_norm=1.0)

        # Create fake gradients
        x = torch.randn(4, 16)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        original_norm = clipper.clip_gradients(simple_model)
        assert original_norm > 0

        # After clipping, norm should be <= max_grad_norm
        clipped_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in simple_model.parameters()
            if p.grad is not None
        ) ** 0.5
        assert clipped_norm <= 1.0 + 1e-5

    def test_per_layer_clip(self, simple_model):
        from erasus.privacy.gradient_clipping import GradientClipper

        clipper = GradientClipper(max_grad_norm=0.5, flat_clipping=False)

        x = torch.randn(4, 16)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        clipper.clip_gradients(simple_model)

    def test_clip_per_sample_gradients(self):
        from erasus.privacy.gradient_clipping import GradientClipper

        clipper = GradientClipper(max_grad_norm=2.0)

        # Create per-sample gradients
        per_sample = {
            "layer1.weight": torch.randn(8, 16, 32),  # 8 samples
            "layer1.bias": torch.randn(8, 32),
        }

        clipped = clipper.clip_per_sample_gradients(per_sample)
        assert clipped["layer1.weight"].shape == (8, 16, 32)
        assert clipped["layer1.bias"].shape == (8, 32)

    def test_clip_grad_norm_func(self, simple_model):
        from erasus.privacy.gradient_clipping import clip_grad_norm_

        x = torch.randn(4, 16)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()

        norm = clip_grad_norm_(simple_model, max_norm=1.0)
        assert norm > 0

    def test_compute_sensitivity(self):
        from erasus.privacy.gradient_clipping import compute_sensitivity

        sens = compute_sensitivity(max_grad_norm=1.0, batch_size=32)
        assert abs(sens - 1.0 / 32) < 1e-8

    def test_calibrate_noise_gaussian(self):
        from erasus.privacy.gradient_clipping import calibrate_noise

        sigma = calibrate_noise(epsilon=1.0, delta=1e-5, sensitivity=0.1)
        assert sigma > 0

    def test_calibrate_noise_laplace(self):
        from erasus.privacy.gradient_clipping import calibrate_noise

        b = calibrate_noise(
            epsilon=1.0, delta=0, sensitivity=0.1, mechanism="laplace",
        )
        assert abs(b - 0.1) < 1e-8


# ======================================================================
# 9. Privacy: Secure Aggregation
# ======================================================================


class TestSecureAggregation:
    def _make_updates(self, n: int = 5) -> list:
        """Create dummy client updates."""
        return [
            {"fc.weight": torch.randn(10, 16) * 0.01, "fc.bias": torch.randn(10) * 0.01}
            for _ in range(n)
        ]

    def test_masking_protocol(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        agg = SecureAggregator(n_clients=5, threshold=3, protocol="masking")
        updates = self._make_updates(5)

        result = agg.aggregate(updates)
        assert "fc.weight" in result
        assert result["fc.weight"].shape == (10, 16)

    def test_secret_sharing_protocol(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        agg = SecureAggregator(n_clients=5, threshold=3, protocol="secret_sharing")
        updates = self._make_updates(5)

        result = agg.aggregate(updates)
        assert "fc.weight" in result

    def test_threshold_protocol(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        agg = SecureAggregator(n_clients=5, threshold=3, protocol="threshold")
        updates = self._make_updates(5)

        result = agg.aggregate(updates, active_client_ids=[0, 1, 2, 3, 4])
        assert "fc.weight" in result

    def test_threshold_too_few_clients(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        agg = SecureAggregator(n_clients=5, threshold=4, protocol="threshold")
        updates = self._make_updates(5)

        with pytest.raises(RuntimeError, match="Only 2 clients active"):
            agg.aggregate(updates, active_client_ids=[0, 1])

    def test_quantization(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        agg = SecureAggregator(
            n_clients=3, threshold=2, protocol="secret_sharing", quantize_bits=8,
        )
        updates = self._make_updates(3)

        result = agg.aggregate(updates)
        assert "fc.weight" in result

    def test_create_factory(self):
        from erasus.privacy.secure_aggregation import create_secure_aggregator

        agg = create_secure_aggregator(n_clients=10, protocol="masking")
        assert agg.n_clients == 10
        assert agg.threshold == 5

    def test_invalid_protocol(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        with pytest.raises(ValueError):
            SecureAggregator(protocol="invalid")

    def test_invalid_threshold(self):
        from erasus.privacy.secure_aggregation import SecureAggregator

        with pytest.raises(ValueError):
            SecureAggregator(n_clients=3, threshold=5)


# ======================================================================
# 10. Utils: Profiling
# ======================================================================


class TestProfiling:
    def test_profile_timing(self):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)

        with profiler.profile("test_op"):
            time.sleep(0.01)

        report = profiler.get_report()
        assert "test_op" in report["timings"]
        assert report["timings"]["test_op"]["total_s"] >= 0.01

    def test_profile_multiple_calls(self):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)

        for _ in range(3):
            with profiler.profile("repeated"):
                time.sleep(0.001)

        report = profiler.get_report()
        assert report["timings"]["repeated"]["count"] == 3

    def test_log_memory(self):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)
        snap = profiler.log_memory("initial")
        assert isinstance(snap, dict)

    def test_count_parameters(self, simple_model):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)
        counts = profiler.count_parameters(simple_model)
        assert counts["total"] > 0
        assert counts["trainable"] > 0
        assert counts["frozen"] == 0

    def test_estimate_flops(self, simple_model):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)
        flops = profiler.estimate_flops(simple_model, (16,))
        assert flops > 0

    def test_summary(self):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)
        with profiler.profile("my_op"):
            pass
        profiler.increment("epoch")

        summary = profiler.summary()
        assert "my_op" in summary
        assert "epoch" in summary

    def test_reset(self):
        from erasus.utils.profiling import UnlearningProfiler

        profiler = UnlearningProfiler(enable_cuda=False)
        with profiler.profile("op"):
            pass
        profiler.reset()
        assert profiler.get_report()["timings"] == {}

    def test_profile_section(self, capsys):
        from erasus.utils.profiling import profile_section

        with profile_section("test", verbose=True):
            time.sleep(0.005)

        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_profile_model_memory(self, simple_model):
        from erasus.utils.profiling import profile_model_memory

        mem = profile_model_memory(simple_model)
        assert "param_mb" in mem
        assert "total_mb" in mem
        assert mem["param_mb"] > 0


# ======================================================================
# 11. Utils: Reproducibility
# ======================================================================


class TestReproducibility:
    def test_make_reproducible(self):
        from erasus.utils.reproducibility import make_reproducible

        make_reproducible(seed=123, deterministic_algorithms=False)

        a = torch.randn(5)
        make_reproducible(seed=123, deterministic_algorithms=False)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_set_seed(self):
        from erasus.utils.reproducibility import set_seed

        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_experiment_snapshot_capture(self, simple_model):
        from erasus.utils.reproducibility import ExperimentSnapshot

        snap = ExperimentSnapshot(experiment_name="test")
        result = snap.capture(
            model=simple_model,
            config={"lr": 0.001, "epochs": 10},
            seed=42,
        )

        assert result["experiment_name"] == "test"
        assert result["seed"] == 42
        assert "environment" in result
        assert "model" in result
        assert result["model"]["total_parameters"] > 0
        assert "config" in result
        assert result["config"]["lr"] == 0.001

    def test_experiment_snapshot_save_load(self, simple_model, tmp_path):
        from erasus.utils.reproducibility import ExperimentSnapshot

        snap = ExperimentSnapshot(
            experiment_name="save_test", output_dir=str(tmp_path),
        )
        snap.capture(model=simple_model, seed=42)
        path = snap.save("test_snapshot.json")

        assert path.exists()

        loaded = ExperimentSnapshot.load(str(path))
        assert loaded["experiment_name"] == "save_test"
        assert loaded["seed"] == 42

    def test_get_reproducibility_info(self):
        from erasus.utils.reproducibility import get_reproducibility_info

        info = get_reproducibility_info()
        assert "environment" in info
        assert "pytorch" in info["environment"]


# ======================================================================
# 12. Model wrappers (import-only tests — no HuggingFace model downloads)
# ======================================================================


class TestModelImports:
    """Test that all new model wrappers can be imported and instantiated
    without downloading actual models."""

    def test_flamingo_import(self):
        from erasus.models.vlm.flamingo import FlamingoWrapper
        assert FlamingoWrapper is not None

    def test_t5_import(self):
        from erasus.models.llm.t5 import T5Wrapper
        assert T5Wrapper is not None

    def test_dalle_import(self):
        from erasus.models.diffusion.dalle import DALLEWrapper
        assert DALLEWrapper is not None

    def test_imagen_import(self):
        from erasus.models.diffusion.imagen import ImagenWrapper
        assert ImagenWrapper is not None

    def test_wav2vec_import(self):
        from erasus.models.audio.wav2vec import Wav2VecWrapper
        assert Wav2VecWrapper is not None

    def test_clap_import(self):
        from erasus.models.audio.clap import CLAPWrapper
        assert CLAPWrapper is not None

    def test_video_clip_import(self):
        from erasus.models.video.video_clip import VideoCLIPWrapper
        assert VideoCLIPWrapper is not None


# ======================================================================
# 13. Package-level init imports
# ======================================================================


class TestPackageImports:
    def test_vlm_package(self):
        from erasus.models.vlm import CLIPWrapper, FlamingoWrapper, ViTFeatureExtractor
        assert CLIPWrapper is not None

    def test_llm_package(self):
        from erasus.models.llm import GPTWrapper, MistralWrapper, T5Wrapper
        assert T5Wrapper is not None

    def test_diffusion_package(self):
        from erasus.models.diffusion import StableDiffusionWrapper, DALLEWrapper, ImagenWrapper
        assert DALLEWrapper is not None

    def test_audio_package(self):
        from erasus.models.audio import WhisperWrapper, Wav2VecWrapper, CLAPWrapper
        assert CLAPWrapper is not None

    def test_video_package(self):
        from erasus.models.video import VideoMAEWrapper, VideoCLIPWrapper
        assert VideoCLIPWrapper is not None

    def test_data_datasets_package(self):
        from erasus.data.datasets import MUSEDataset, ImageNetDataset
        assert MUSEDataset is not None

    def test_data_synthetic_package(self):
        from erasus.data.synthetic import BackdoorGenerator, BiasGenerator, PrivacyDataGenerator
        assert BiasGenerator is not None

    def test_privacy_package(self):
        from erasus.privacy import (
            PrivacyAccountant,
            GradientClipper,
            clip_grad_norm_,
            calibrate_noise,
            SecureAggregator,
            create_secure_aggregator,
        )
        assert GradientClipper is not None

    def test_utils_package(self):
        from erasus.utils import (
            UnlearningProfiler,
            profile_section,
            profile_model_memory,
            make_reproducible,
            ExperimentSnapshot,
        )
        assert UnlearningProfiler is not None
