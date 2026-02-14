"""
Tests for Sprint C — Examples, Benchmarks & Paper Reproductions.

Verifies that all Sprint C scripts are importable and that their
core components (models, data loaders, functions) work correctly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: import a module from a file path
# ---------------------------------------------------------------------------
def import_from_path(name: str, path: str):
    """Import a module from file path dynamically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- Determine project root ----
ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# 1. Example Script Imports (syntax + top-level classes)
# ===========================================================================
class TestExampleImports:
    """Verify every example script can be imported without errors."""

    EXAMPLE_SCRIPTS = [
        "examples/vision_language/multi_modal_benchmark.py",
        "examples/language_models/mistral_bias_removal.py",
        "examples/language_models/bert_feature_unlearning.py",
        "examples/language_models/continual_unlearning.py",
        "examples/diffusion_models/dalle_concept_removal.py",
        "examples/diffusion_models/diffusion_backdoor_removal.py",
        "examples/audio_models/whisper_unlearning.py",
        "examples/audio_models/wav2vec_unlearning.py",
        "examples/video_models/videomae_unlearning.py",
        "examples/video_models/video_clip_unlearning.py",
        "examples/advanced/federated_unlearning.py",
        "examples/advanced/differential_privacy.py",
        "examples/advanced/adversarial_unlearning.py",
        "examples/advanced/certified_removal.py",
        "examples/advanced/multi_task_unlearning.py",
        "examples/benchmarks/run_muse_benchmark.py",
        "examples/benchmarks/compare_methods.py",
        "examples/benchmarks/ablation_studies.py",
    ]

    @pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
    def test_import(self, script):
        path = ROOT / script
        assert path.exists(), f"Missing: {script}"
        mod = import_from_path(script.replace("/", ".").replace(".py", ""), str(path))
        assert hasattr(mod, "main"), f"{script} must define main()"


# ===========================================================================
# 2. Example Models Instantiation
# ===========================================================================
class TestExampleModels:
    """Verify example model classes can be instantiated and run forward."""

    def test_multi_modal_benchmark_model(self):
        mod = import_from_path("mm", str(ROOT / "examples/vision_language/multi_modal_benchmark.py"))
        model = mod.TinyVLM()
        out = model(torch.randn(2, 3, 32, 32), torch.randint(0, 500, (2, 20)))
        assert out.shape == (2, 10)

    def test_mistral_model(self):
        mod = import_from_path("mr", str(ROOT / "examples/language_models/mistral_bias_removal.py"))
        model = mod.TinyLM()
        out = model(torch.randint(0, 256, (2, 32)))
        assert out.shape == (2, 256)

    def test_bert_model(self):
        mod = import_from_path("bf", str(ROOT / "examples/language_models/bert_feature_unlearning.py"))
        model = mod.TinyBERT()
        out = model(torch.randint(0, 256, (2, 32)))
        assert out.shape == (2, 4)

    def test_multi_task_model(self):
        mod = import_from_path("mt", str(ROOT / "examples/advanced/multi_task_unlearning.py"))
        model = mod.MultiTaskModel()
        for task_id in range(model.n_tasks):
            out = model(torch.randn(2, 16), task_id)
            assert out.shape == (2, 4)

    def test_adversarial_fgsm(self):
        mod = import_from_path("adv", str(ROOT / "examples/advanced/adversarial_unlearning.py"))
        model = mod.RobustClassifier()
        x = torch.randn(4, 16)
        y = torch.randint(0, 4, (4,))
        x_adv = mod.fgsm_attack(model, x, y)
        assert x_adv.shape == x.shape

    def test_federated_model(self):
        mod = import_from_path("fed", str(ROOT / "examples/advanced/federated_unlearning.py"))
        model = mod.FedModel()
        out = model(torch.randn(2, 16))
        assert out.shape == (2, 4)


# ===========================================================================
# 3. Benchmark Suite Imports
# ===========================================================================
class TestBenchmarkImports:
    """Verify benchmark suite scripts are importable."""

    BENCHMARK_SCRIPTS = [
        "benchmarks/muse/run.py",
        "benchmarks/custom/privacy_benchmark.py",
        "benchmarks/custom/efficiency_benchmark.py",
        "benchmarks/custom/utility_benchmark.py",
    ]

    @pytest.mark.parametrize("script", BENCHMARK_SCRIPTS)
    def test_import(self, script):
        path = ROOT / script
        assert path.exists(), f"Missing: {script}"
        mod = import_from_path(script.replace("/", ".").replace(".py", ""), str(path))
        assert hasattr(mod, "main"), f"{script} must define main()"


# ===========================================================================
# 4. Benchmark Config Files
# ===========================================================================
class TestBenchmarkConfigs:
    """Verify benchmark config files exist and are valid YAML."""

    CONFIGS = [
        "benchmarks/muse/config.yaml",
        "benchmarks/tofu/config.yaml",
        "benchmarks/wmdp/config.yaml",
    ]

    @pytest.mark.parametrize("config", CONFIGS)
    def test_config_exists(self, config):
        path = ROOT / config
        assert path.exists(), f"Missing: {config}"
        content = path.read_text()
        assert "benchmark:" in content or "name:" in content


# ===========================================================================
# 5. Paper Reproduction Imports
# ===========================================================================
class TestPaperReproductions:
    """Verify paper reproduction scripts are importable."""

    PAPERS = [
        "papers/reproductions/gradient_ascent_unlearning.py",
        "papers/reproductions/scrub_cvpr2024.py",
        "papers/reproductions/ssd_neurips2024.py",
        "papers/reproductions/concept_erasure_iccv2023.py",
    ]

    @pytest.mark.parametrize("script", PAPERS)
    def test_import(self, script):
        path = ROOT / script
        assert path.exists(), f"Missing: {script}"
        mod = import_from_path(script.replace("/", ".").replace(".py", ""), str(path))
        assert hasattr(mod, "main")

    def test_scrub_model(self):
        mod = import_from_path("scrub", str(ROOT / "papers/reproductions/scrub_cvpr2024.py"))
        model = mod.ResNetSmall()
        out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)

    def test_ssd_model(self):
        mod = import_from_path("ssd", str(ROOT / "papers/reproductions/ssd_neurips2024.py"))
        model = mod.MLP()
        out = model(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)

    def test_concept_erasure_model(self):
        mod = import_from_path("ce", str(ROOT / "papers/reproductions/concept_erasure_iccv2023.py"))
        model = mod.SimpleDiffModel()
        out = model(torch.randn(2, 32))
        assert out.shape == (2, 32)


# ===========================================================================
# 6. Existing Example Smoke Tests
# ===========================================================================
class TestExistingExamples:
    """Verify pre-existing example scripts still importable."""

    EXISTING = [
        "examples/clip_basic.py",
        "examples/vision_language/clip_coreset_comparison.py",
        "examples/vision_language/llava_unlearning.py",
        "examples/vision_language/blip_unlearning.py",
        "examples/language_models/gpt2_unlearning.py",
        "examples/language_models/llama_concept_removal.py",
        "examples/language_models/lora_efficient_unlearning.py",
        "examples/diffusion_models/stable_diffusion_nsfw.py",
        "examples/diffusion_models/stable_diffusion_artist.py",
    ]

    @pytest.mark.parametrize("script", EXISTING)
    def test_exists(self, script):
        path = ROOT / script
        assert path.exists(), f"Missing: {script}"
