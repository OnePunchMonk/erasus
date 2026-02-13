---
description: Next set of agendas for completing the Erasus framework based on the specification gap analysis
---

# Erasus Framework — Next Agenda (Gap Analysis vs. Specification)

**Date:** 2026-02-14
**Source:** `erasus_complete_comprehensive_specification.txt`

---

## ✅ What's DONE (Phase 1 Complete)

### Core Infrastructure
- `core/` — `base_unlearner.py`, `base_selector.py`, `base_strategy.py`, `base_metric.py`, `config.py`, `registry.py`, `exceptions.py`, `types.py` — **All implemented**

### Models (15+ Architectures target → 8 implemented)
- `models/vlm/clip.py` ✅ | `llava.py` ✅ | `blip.py` ✅
- `models/llm/llama.py` ✅ | `mistral.py` ✅ | `gpt.py` ✅ | `bert.py` ✅
- `models/diffusion/stable_diffusion.py` ✅
- `models/audio/whisper.py` ✅
- `models/video/videomae.py` ✅

### Selectors (25+ target → 15+ implemented)
- `gradient_based/` — `influence.py` ✅ | `tracin.py` ✅ | `gradient_norm.py` ✅ | `grad_match.py` ✅ | `el2n.py` ✅ | `representer.py` ✅
- `geometry_based/` — `kcenter.py` ✅ | `herding.py` ✅ | `craig.py` ✅ | `glister.py` ✅ | `submodular.py` ✅ | `kmeans_coreset.py` ✅
- `learning_based/` — `forgetting_events.py` ✅ | `data_shapley.py` ✅ | `valuation_network.py` ✅
- `ensemble/voting.py` ✅ | `auto_selector.py` ✅ | `random_selector.py` ✅ | `full_selector.py` ✅

### Strategies (20+ target → 17+ implemented)
- `gradient_methods/` — `gradient_ascent.py` ✅ | `scrub.py` ✅ | `modality_decoupling.py` ✅ | `fisher_forgetting.py` ✅ | `negative_gradient.py` ✅
- `parameter_methods/` — `lora_unlearning.py` ✅ | `sparse_aware.py` ✅ | `mask_based.py` ✅ | `neuron_pruning.py` ✅
- `data_methods/` — `amnesiac.py` ✅ | `sisa.py` ✅ | `certified_removal.py` ✅
- `llm_specific/` — `ssd.py` ✅ | `token_masking.py` ✅ | `embedding_alignment.py` ✅ | `causal_tracing.py` ✅
- `diffusion_specific/` — `concept_erasure.py` ✅ | `noise_injection.py` ✅ | `unet_surgery.py` ✅
- `vlm_specific/` — `contrastive_unlearning.py` ✅

### Losses
- `retain_anchor.py` ✅ | `contrastive.py` ✅ | `kl_divergence.py` ✅ | `mmd.py` ✅ | `custom_losses.py` ✅

### Metrics
- `accuracy.py` ✅ | `membership_inference.py` ✅ | `perplexity.py` ✅ | `retrieval.py` ✅ | `fid.py` ✅

### Privacy
- `accountant.py` ✅ | `dp_mechanisms.py` ✅ | `certificates.py` ✅ | `influence_bounds.py` ✅

### Utils / Data / Config
- `checkpointing.py` ✅ | `logging.py` ✅ | `seed.py` ✅
- `loaders.py` ✅ | `datasets.py` ✅ | `multimodal.py` ✅ | `splits.py` ✅ | `transforms.py` ✅

---

## 🔴 GAPS — What's MISSING vs. Specification

### PRIORITY 1 — High-Level Unlearner API (Critical for usability)
The spec (Section 9, 10) shows a high-level `ErasusUnlearner.fit()` API. Currently:
- `erasus/unlearners/erasus_unlearner.py` exists (2.4KB) — **needs audit**
- **MISSING**: `vlm_unlearner.py`, `llm_unlearner.py`, `diffusion_unlearner.py`, `audio_unlearner.py`, `video_unlearner.py`, `multimodal_unlearner.py`
- These are the user-facing orchestration classes that tie selector → strategy → metric together

### PRIORITY 2 — Visualization Module (Spec Section: visualization/)
Currently only 3 files exist:
- `loss_curves.py` ✅ | `feature_plots.py` ✅ | `mia_plots.py` ✅
- **MISSING**: `embeddings.py`, `surfaces.py`, `gradients.py`, `reports.py`, `interactive.py`
- These should provide t-SNE/PCA embedding plots, loss landscape surfaces, gradient flow visualization, HTML report generation, and interactive dashboards

### PRIORITY 3 — Metrics Module Restructuring (Spec Section 6)
The specification defines a much richer metrics hierarchy:
- **MISSING** `metrics/metric_suite.py` — Unified metric runner
- **MISSING** `metrics/forgetting/` directory:
  - `mia.py` — Full blown MIA with ROC curves (current `membership_inference.py` is flat, not in subfolder)
  - `mia_variants.py` — LiRA and other advanced attacks
  - `confidence.py` — Confidence-based forgetting measures
  - `feature_distance.py` — Embedding distance metrics
- **MISSING** `metrics/utility/` directory (currently flat)
- **MISSING** `metrics/efficiency/`:
  - `time_complexity.py` — Wall-clock and FLOPs tracking
  - `memory_usage.py` — Peak memory, GPU utilization
- **MISSING** `metrics/privacy/`:
  - `differential_privacy.py` — DP-specific evaluation metrics

### PRIORITY 4 — Benchmark Datasets (Spec Section 5)
- **MISSING** `data/datasets/` directory entirely:
  - `coco.py` — COCO Captions dataset wrapper
  - `conceptual_captions.py` — CC3M/CC12M wrapper
  - `tofu.py` — TOFU benchmark (critical for LLM eval)
  - `wmdp.py` — WMDP benchmark
  - `i2p.py` — Inappropriate Image Prompts for diffusion
- **MISSING** `data/synthetic/backdoor_generator.py`
- **MISSING** `data/preprocessing.py`, `data/partitioning.py`, `data/samplers.py`

### PRIORITY 5 — CLI Module (Spec Section: cli/)
- `cli/main.py` exists (1.5KB) — **needs audit**
- **MISSING**: `cli/unlearn.py`, `cli/evaluate.py`
- These enable `erasus unlearn --config config.yaml` and `erasus evaluate` commands

### PRIORITY 6 — VLM-Specific Strategy Gap
- `vlm_specific/cross_modal_decoupling.py` is only 214 bytes — **likely a stub/alias**

### PRIORITY 7 — Certification Module (Spec Section 7.2)
The specification defines `certification/` as separate from `privacy/`:
- **MISSING**: `certification/` directory:
  - `certified_removal.py`
  - `verification.py`
- Note: the privacy folder has `certificates.py` and `influence_bounds.py` but these may not cover formal verification

### PRIORITY 8 — Experiments Module
- **MISSING** `experiments/` directory:
  - `experiment_tracker.py` — W&B / MLflow integration for tracking runs

### PRIORITY 9 — Project Infrastructure
- **MISSING** `configs/` directory with YAML presets:
  - `models/clip.yaml`, `llama.yaml`, `stable_diffusion.yaml`
  - `selectors/influence.yaml`, `craig.yaml`, `auto.yaml`
  - `strategies/gradient_ascent.yaml`, `modality_decoupling.yaml`, `scrub.yaml`
  - `default.yaml`
- **MISSING** `scripts/` — `setup_env.sh`, `download_datasets.py`, `run_benchmarks.sh`
- **MISSING** `benchmarks/` — `tofu/run.py`, `muse/run.py`, `wmdp/run.py`
- **MISSING** `examples/` — Only `clip_basic.py` exists. Need:
  - `vision_language/clip_coreset_comparison.py`, `llava_unlearning.py`, `blip_unlearning.py`
  - `language_models/llama_concept_removal.py`, `gpt2_unlearning.py`, `lora_efficient_unlearning.py`
  - `diffusion_models/stable_diffusion_artist.py`, `stable_diffusion_nsfw.py`
  - `benchmarks/run_tofu_benchmark.py`
- **MISSING** `docs/` — Documentation with Sphinx/RST
- **MISSING** `.github/workflows/` — CI/CD
- **MISSING** `docker/` — Dockerfiles
- **MISSING** `papers/reproductions/` — Paper reproduction scripts
- **MISSING** `utils/distributed.py`, `utils/helpers.py`

### PRIORITY 10 — Testing
Current tests are minimal (5 files, ~12KB). Spec requires:
- **MISSING** `tests/conftest.py` — Shared fixtures
- **MISSING** `tests/unit/test_selectors.py`, `test_strategies.py`, `test_metrics.py`
- **MISSING** `tests/integration/` — `test_clip_pipeline.py`, `test_llm_pipeline.py`, `test_diffusion_pipeline.py`
- **MISSING** `tests/benchmarks/test_performance.py`

---

## 📋 RECOMMENDED SPRINT PLAN

### Sprint 1 — Unlearners + CLI (User-Facing API) ⚡
**Goal:** Make the framework usable end-to-end from a single entry point
1. Audit and complete `erasus_unlearner.py`
2. Create `vlm_unlearner.py`, `llm_unlearner.py`, `diffusion_unlearner.py`
3. Complete `cli/unlearn.py` and `cli/evaluate.py`
4. Create `configs/default.yaml` and model-specific YAML configs
5. Write an end-to-end integration test

### Sprint 2 — Visualization Module 📊
**Goal:** Complete all visualization capabilities
1. Implement `embeddings.py` — t-SNE/PCA plots of forget/retain embeddings
2. Implement `surfaces.py` — Loss landscape visualization
3. Implement `gradients.py` — Gradient flow and magnitude plots
4. Implement `reports.py` — HTML report generator with all metrics/plots
5. Implement `interactive.py` — Plotly/Dash interactive dashboard

### Sprint 3 — Metrics Restructuring + New Metrics 📐
**Goal:** Match the 50+ metrics target from the spec
1. Create `metrics/metric_suite.py` — Unified runner
2. Create `metrics/forgetting/` — `mia.py`, `mia_variants.py`, `confidence.py`, `feature_distance.py`
3. Create `metrics/efficiency/` — `time_complexity.py`, `memory_usage.py`
4. Create `metrics/privacy/differential_privacy.py`
5. Refactor existing flat metrics into the hierarchy

### Sprint 4 — Benchmark Datasets 📦
**Goal:** Make TOFU, WMDP, I2P, COCO usable out of the box
1. Create `data/datasets/tofu.py` — TOFU benchmark loader
2. Create `data/datasets/wmdp.py` — WMDP benchmark loader
3. Create `data/datasets/coco.py` — COCO Captions loader
4. Create `data/datasets/i2p.py` — I2P prompts loader
5. Create `data/preprocessing.py`, `data/samplers.py`

### Sprint 5 — Examples + Benchmarks + Documentation 📚
**Goal:** Make the framework approachable
1. Write 8+ example scripts (CLIP, LLaVA, LLaMA, GPT-2, Stable Diffusion)
2. Create benchmark runners (`benchmarks/tofu/`, `benchmarks/wmdp/`)
3. Create `papers/reproductions/` scripts
4. Add comprehensive docstrings across all modules
5. Set up Sphinx documentation skeleton

### Sprint 6 — Testing + CI/CD + Docker 🧪
**Goal:** Production-readiness
1. Create `tests/conftest.py` with shared fixtures
2. Write unit tests for all selectors, strategies, metrics
3. Write integration tests for CLIP, LLM, Diffusion pipelines
4. Set up `.github/workflows/ci.yml`
5. Create `docker/Dockerfile` and `docker-compose.yml`

---

## 🎯 IMMEDIATE NEXT SESSION AGENDA

For the next coding session, focus on **Sprint 1** (most impactful):

1. **Audit `erasus_unlearner.py`** — Verify the high-level `.fit()` API works
2. **Create `vlm_unlearner.py`** — CLIP/LLaVA/BLIP orchestration
3. **Create `llm_unlearner.py`** — LLaMA/GPT/Mistral orchestration
4. **Create `diffusion_unlearner.py`** — Stable Diffusion concept erasure orchestration
5. **Complete `cli/unlearn.py`** — CLI command to run unlearning from terminal
6. **Complete `cli/evaluate.py`** — CLI command to evaluate results
7. **Create YAML configs** — `configs/default.yaml`, `configs/models/*.yaml`
8. **Write integration test** — End-to-end test with a tiny model
