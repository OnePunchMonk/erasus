---
description: Comprehensive agenda for the Erasus framework — gap analysis vs. full specification + future enhancements
---

# Erasus Framework — Comprehensive Agenda

**Last Updated:** 2026-02-14 (Sprint B complete)
**Test Status:** 206 / 206 passing ✅

---

## ✅ CURRENTLY IMPLEMENTED

### Core Infrastructure ✅
- `core/base_unlearner.py`, `core/base_selector.py`, `core/base_strategy.py`, `core/base_metric.py`
- `core/config.py`, `core/registry.py`, `core/exceptions.py`, `core/types.py`

### Models (10 architectures) ✅
- **VLM:** `clip.py`, `llava.py`, `blip.py`
- **LLM:** `llama.py`, `mistral.py`, `gpt.py`, `bert.py`
- **Diffusion:** `stable_diffusion.py`
- **Audio:** `whisper.py`
- **Video:** `videomae.py`
- **Infra:** `model_wrapper.py`, `registry.py`

### Selectors (22 implementations) ✅
- **Gradient-based:** `influence.py`, `tracin.py`, `gradient_norm.py`, `grad_match.py`, `el2n.py`, `representer.py`, `forgetting_score.py`
- **Geometry-based:** `kcenter.py`, `herding.py`, `craig.py`, `glister.py`, `submodular.py`, `kmeans_coreset.py`, `k_center.py`
- **Learning-based:** `forgetting_events.py`, `data_shapley.py`, `valuation_network.py`, `loss_accum.py`, `active_learning.py` ← NEW
- **Ensemble:** `voting.py`, `weighted_fusion.py` ← NEW
- **Quality analysis:** `quality_metrics.py` ← NEW
- **Utility:** `auto_selector.py`, `random_selector.py`, `full_selector.py`

### Strategies (28 implementations) ✅
- **Gradient methods:** `gradient_ascent.py`, `scrub.py`, `modality_decoupling.py`, `fisher_forgetting.py`, `negative_gradient.py`, `saliency_unlearning.py` ← Sprint A
- **Parameter methods:** `lora_unlearning.py`, `sparse_aware.py`, `mask_based.py`, `neuron_pruning.py`, `layer_freezing.py` ← Sprint A
- **Data methods:** `amnesiac.py`, `sisa.py`, `certified_removal.py`, `knowledge_distillation.py` ← Sprint A
- **LLM-specific:** `ssd.py`, `token_masking.py`, `embedding_alignment.py`, `causal_tracing.py`, `attention_surgery.py` ← Sprint A
- **Diffusion-specific:** `concept_erasure.py`, `noise_injection.py`, `unet_surgery.py`, `timestep_masking.py` ← Sprint A, `safe_latents.py` ← Sprint A
- **VLM-specific:** `contrastive_unlearning.py`, `cross_modal_decoupling.py`, `attention_unlearning.py` ← Sprint A, `vision_text_split.py` ← NEW Sprint F
- **Ensemble:** `ensemble_strategy.py` ← Sprint A

### Losses (8 implementations) ✅
- **Original (5):** `retain_anchor.py`, `contrastive.py`, `kl_divergence.py`, `mmd.py`, `custom_losses.py`
- **New (4):** `fisher_regularization.py` ← NEW, `adversarial_loss.py` ← NEW, `triplet_loss.py` ← NEW, `l2_regularization.py` ← NEW

### Unlearner API (8 classes) ✅
- `erasus_unlearner.py`, `vlm_unlearner.py`, `llm_unlearner.py`, `diffusion_unlearner.py`
- `audio_unlearner.py`, `video_unlearner.py`, `multimodal_unlearner.py`
- `federated_unlearner.py` ← NEW Sprint F

### Metrics (26+ implementations) ✅
- **Flat:** `accuracy.py`, `membership_inference.py`, `perplexity.py`, `retrieval.py`, `fid.py`, `retrieval_metrics.py`
- **Suite:** `metric_suite.py`
- **Forgetting:** `mia.py`, `mia_variants.py`, `confidence.py`, `feature_distance.py`, `activation_analysis.py` ← Sprint A, `backdoor_activation.py` ← Sprint A, `extraction_attack.py` ← NEW Sprint F
- **Efficiency:** `time_complexity.py`, `memory_usage.py`, `speedup.py` ← Sprint A, `flops.py` ← Sprint A
- **Utility:** `clip_score.py` ← NEW, `bleu.py` ← NEW, `rouge.py` ← NEW, `inception_score.py` ← NEW, `downstream_tasks.py` ← NEW (all Sprint F)
- **Privacy:** `differential_privacy.py`, `epsilon_delta.py` ← NEW Sprint F, `privacy_audit.py` ← NEW Sprint F
- **Benchmark:** `benchmarks.py` ← NEW Sprint F (unified runner with LaTeX, radar plots, statistical tests)

### Visualization (13 files) ✅
- **Original (8):** `embeddings.py`, `surfaces.py`, `gradients.py`, `reports.py`, `interactive.py`, `loss_curves.py`, `feature_plots.py`, `mia_plots.py`
- **Sprint A (2):** `attention.py`, `comparisons.py`
- **Sprint F (3):** `activation.py` ← NEW, `influence_maps.py` ← NEW, `cross_modal.py` ← NEW

### Data Module ✅
- **Datasets:** `tofu.py`, `wmdp.py`, `coco.py`, `i2p.py`, `conceptual_captions.py`, `muse.py` ← NEW Sprint B, `imagenet.py` ← NEW Sprint B
- **Utils:** `preprocessing.py`, `partitioning.py`, `samplers.py`, `loaders.py`, `transforms.py`, `splits.py`, `datasets.py`, `multimodal.py`
- **Augmentation:** `augmentation.py` ← NEW Sprint B
- **Synthetic:** `backdoor_generator.py`, `bias_generator.py` ← NEW Sprint B, `privacy_generator.py` ← NEW Sprint B

### Privacy Module ✅
- `accountant.py`, `dp_mechanisms.py`, `certificates.py`, `influence_bounds.py`
- `gradient_clipping.py` ← NEW Sprint B, `secure_aggregation.py` ← NEW Sprint B

### Certification Module ✅
- `certified_removal.py`, `verification.py`, `bounds.py` ← NEW (PAC bounds, influence bounds, certified radius)

### CLI (4 commands) ✅
- `main.py`, `unlearn.py`, `evaluate.py`, `benchmark.py` ← NEW, `visualize.py` ← NEW

### Utils (9 modules) ✅
- **Original (5):** `checkpointing.py`, `distributed.py`, `helpers.py`, `logging.py`, `seed.py`
- **Sprint A (2):** `callbacks.py`, `early_stopping.py`
- **Sprint B (2):** `profiling.py` ← NEW, `reproducibility.py` ← NEW

### Experiments (3 modules) ✅
- `experiment_tracker.py` (local/W&B/MLflow)
- `hyperparameter_search.py` ← NEW (Optuna + random search fallback)
- `ablation_studies.py` ← NEW (automated ablation runner)

### Configs ✅
- `default.yaml` + model/strategy/selector configs

### Examples (9 scripts) ✅
- VLM: `clip_basic.py`, `clip_coreset_comparison.py`, `llava_unlearning.py`, `blip_unlearning.py`
- LLM: `llama_concept_removal.py`, `gpt2_unlearning.py`, `lora_efficient_unlearning.py`
- Diffusion: `stable_diffusion_nsfw.py`, `stable_diffusion_artist.py`
- Benchmark: `run_tofu_benchmark.py`

### Benchmarks ✅
- `benchmarks/tofu/run.py`, `benchmarks/wmdp/run.py`

### Paper Reproductions ✅
- `papers/reproductions/gradient_ascent_unlearning.py`

### CI/CD + Docker ✅
- `.github/workflows/ci.yml`, `docker/Dockerfile`, `docker/docker-compose.yml`

### Docs ✅
- `docs/conf.py`, `docs/index.rst`, `docs/quickstart.rst`, `docs/installation.rst`
- `docs/api/core.rst`, `docs/api/unlearners.rst`

### Tests (206 passing) ✅

---

## ✅ SPRINT A — COMPLETED (2026-02-14)

**22 new files implemented:**

| Category | New Modules | Count |
|----------|------------|:-----:|
| **Strategies** | `saliency_unlearning`, `layer_freezing`, `knowledge_distillation`, `attention_surgery`, `timestep_masking`, `safe_latents`, `attention_unlearning`, `ensemble_strategy` | 8 |
| **Losses** | `fisher_regularization`, `adversarial_loss`, `triplet_loss`, `l2_regularization` | 4 |
| **Metrics** | `activation_analysis`, `backdoor_activation`, `speedup`, `flops` | 4 |
| **Visualization** | `attention`, `comparisons` | 2 |
| **Utils** | `callbacks`, `early_stopping` | 2 |
| **CLI** | `benchmark`, `visualize` | 2 |
| **Certification** | `bounds` (PAC, influence, certified radius) | 1 |
| **Experiments** | `hyperparameter_search`, `ablation_studies` | 2 |

**Updated files:**
- `strategies/__init__.py` — registers all 27 strategies
- `experiments/__init__.py` — exports new experiment tools
- `cli/main.py` — adds `benchmark` and `visualize` sub-commands
- `README.md` — fully updated to reflect expanded framework

---

## 🔮 FUTURE ENHANCEMENTS — COMPREHENSIVE ROADMAP

### Priority 1: Remaining Missing Modules from Specification

#### 1.1 Missing Model Architectures
| Module | Description | Priority |
|--------|-------------|----------|
| ~~`models/vlm/flamingo.py`~~ | ~~Flamingo VLM adapter~~ | ✅ Done (Sprint B) |
| ~~`models/vlm/vision_transformer.py`~~ | ~~ViT helper utilities~~ | ✅ Done (Sprint B) |
| ~~`models/llm/t5.py`~~ | ~~T5 encoder-decoder model wrapper~~ | ✅ Done (Sprint B) |
| ~~`models/diffusion/dalle.py`~~ | ~~DALL-E 2/3 model wrapper~~ | ✅ Done (Sprint B) |
| ~~`models/diffusion/imagen.py`~~ | ~~Imagen model wrapper~~ | ✅ Done (Sprint B) |
| ~~`models/diffusion/diffusion_utils.py`~~ | ~~Noise schedulers, diffusion helpers~~ | ✅ Done (Sprint B) |
| ~~`models/audio/wav2vec.py`~~ | ~~Wav2Vec 2.0 model wrapper~~ | ✅ Done (Sprint B) |
| ~~`models/audio/clap.py`~~ | ~~CLAP audio-text model~~ | ✅ Done (Sprint B) |
| ~~`models/video/video_clip.py`~~ | ~~VideoCLIP model wrapper~~ | ✅ Done (Sprint B) |
| DINOv2 | Self-supervised vision model | 🟢 Low |
| Segment Anything (SAM) | Foundation segmentation model | 🟢 Low |
| Gemma / Phi | Smaller LLMs for efficient unlearning | 🟡 Medium |

#### 1.2 Remaining Missing Strategies
| Module | Description | Priority |
|--------|-------------|----------|
| ~~`strategies/gradient_methods/saliency_unlearning.py`~~ | ~~Saliency-guided gradient unlearning~~ | ✅ Done |
| ~~`strategies/parameter_methods/layer_freezing.py`~~ | ~~Selective layer freezing~~ | ✅ Done |
| ~~`strategies/data_methods/knowledge_distillation.py`~~ | ~~Teacher-student unlearning via KD~~ | ✅ Done |
| ~~`strategies/llm_specific/attention_surgery.py`~~ | ~~Direct attention weight modification~~ | ✅ Done |
| ~~`strategies/diffusion_specific/timestep_masking.py`~~ | ~~Selective timestep training~~ | ✅ Done |
| ~~`strategies/diffusion_specific/safe_latents.py`~~ | ~~Safe Latent Diffusion (SLD)~~ | ✅ Done |
| ~~`strategies/vlm_specific/attention_unlearning.py`~~ | ~~Cross-attention modification~~ | ✅ Done |
| ~~`strategies/vlm_specific/vision_text_split.py`~~ | ~~Separate encoder update strategy~~ | ✅ Done |
| ~~`strategies/ensemble_strategy.py`~~ | ~~Combine multiple strategies~~ | ✅ Done |

#### 1.3 Remaining Missing Selectors
All planned selectors are now implemented. ✅

#### 1.4 Remaining Missing Losses
All originally planned losses are now implemented. ✅

#### 1.5 Remaining Missing Metrics
All planned metrics are now implemented. ✅

#### 1.6 Remaining Missing Visualization
All planned visualization modules are now implemented. ✅

#### 1.7 Missing Data Components
| Module | Description | Priority |
|--------|-------------|----------|
| ~~`data/augmentation.py`~~ | ~~Data augmentation strategies for unlearning~~ | ✅ Done (Sprint B) |
| ~~`data/datasets/imagenet.py`~~ | ~~ImageNet variants loader~~ | ✅ Done (Sprint B) |
| `data/datasets/laion.py` | LAION subset loaders | 🟢 Low |
| ~~`data/datasets/muse.py`~~ | ~~MUSE benchmark dataset~~ | ✅ Done (Sprint B) |
| ~~`data/synthetic/bias_generator.py`~~ | ~~Synthetic bias injection for fairness~~ | ✅ Done (Sprint B) |
| ~~`data/synthetic/privacy_generator.py`~~ | ~~Privacy-sensitive synthetic data~~ | ✅ Done (Sprint B) |

#### 1.8 Missing Privacy Components
All planned privacy modules are now implemented. ✅

#### 1.9 Remaining Missing Certification
All planned certification modules are now implemented. ✅

#### 1.10 Remaining Missing Utils
All planned utils modules are now implemented. ✅

#### 1.11 Remaining Missing CLI Commands
All planned CLI commands are now implemented. ✅

#### 1.12 Remaining Missing Experiment Tools
All planned experiment tools are now implemented. ✅

#### 1.13 Missing Unlearners
All planned unlearners are now implemented. ✅

---

### Priority 2: Missing Examples & Benchmarks from Specification

#### 2.1 Missing Example Scripts
| Script | Description |
|--------|-------------|
| `examples/vision_language/multi_modal_benchmark.py` | Multi-modal comparison benchmark |
| `examples/language_models/mistral_bias_removal.py` | Mistral bias removal example |
| `examples/language_models/bert_feature_unlearning.py` | BERT feature unlearning |
| `examples/language_models/continual_unlearning.py` | Continual/sequential unlearning |
| `examples/diffusion_models/dalle_concept_removal.py` | DALL-E concept removal |
| `examples/diffusion_models/diffusion_backdoor_removal.py` | Backdoor removal from diffusion |
| `examples/audio_models/whisper_unlearning.py` | Whisper unlearning example |
| `examples/audio_models/wav2vec_unlearning.py` | Wav2Vec unlearning example |
| `examples/video_models/videomae_unlearning.py` | VideoMAE unlearning example |
| `examples/video_models/video_clip_unlearning.py` | VideoCLIP unlearning example |
| `examples/advanced/federated_unlearning.py` | Federated unlearning demo |
| `examples/advanced/differential_privacy.py` | DP-enabled unlearning demo |
| `examples/advanced/adversarial_unlearning.py` | Adversarial robustness in unlearning |
| `examples/advanced/certified_removal.py` | Certified removal end-to-end demo |
| `examples/advanced/multi_task_unlearning.py` | Multi-task unlearning scenario |
| `examples/benchmarks/run_muse_benchmark.py` | MUSE benchmark runner |
| `examples/benchmarks/compare_methods.py` | Side-by-side method comparison |
| `examples/benchmarks/ablation_studies.py` | Ablation study example |

#### 2.2 Missing Notebooks
| Notebook | Description |
|----------|-------------|
| `notebooks/01_introduction.ipynb` | Interactive introduction |
| `notebooks/02_coreset_analysis.ipynb` | Coreset selection theory & practice |
| `notebooks/03_multimodal_unlearning.ipynb` | Multimodal walkthrough |
| `notebooks/04_privacy_guarantees.ipynb` | Privacy analysis notebook |
| `notebooks/05_custom_research.ipynb` | Extending Erasus for research |
| `examples/notebooks/interactive_demo.ipynb` | Interactive demo notebook |
| `examples/notebooks/visualization_gallery.ipynb` | Visualization showcase |
| `examples/notebooks/research_reproducibility.ipynb` | Research reproducibility |
| `docs/source/tutorials/01_basic_unlearning.ipynb` | Tutorial: basics |
| `docs/source/tutorials/02_clip_multimodal.ipynb` | Tutorial: CLIP multimodal |
| `docs/source/tutorials/03_llm_concept_removal.ipynb` | Tutorial: LLM concept removal |
| `docs/source/tutorials/04_diffusion_artist_removal.ipynb` | Tutorial: Diffusion artist removal |
| `docs/source/tutorials/05_custom_coreset.ipynb` | Tutorial: Custom coreset |
| `docs/source/tutorials/06_distributed_unlearning.ipynb` | Tutorial: Distributed |
| `docs/source/tutorials/07_privacy_guarantees.ipynb` | Tutorial: Privacy guarantees |

#### 2.3 Missing Benchmark Suites
| Benchmark | Description |
|-----------|-------------|
| `benchmarks/muse/run.py` | MUSE benchmark runner |
| `benchmarks/muse/config.yaml` | MUSE benchmark config |
| `benchmarks/custom/privacy_benchmark.py` | Privacy-focused benchmark |
| `benchmarks/custom/efficiency_benchmark.py` | Efficiency-focused benchmark |
| `benchmarks/custom/utility_benchmark.py` | Utility preservation benchmark |
| `benchmarks/tofu/config.yaml` | TOFU benchmark config |
| `benchmarks/wmdp/config.yaml` | WMDP benchmark config |

#### 2.4 Missing Paper Reproductions
| Script | Paper | Venue |
|--------|-------|-------|
| `papers/reproductions/scrub_cvpr2024.py` | Kurmanji et al. | CVPR 2024 |
| `papers/reproductions/ssd_neurips2024.py` | Foster et al. | NeurIPS 2024 |
| `papers/reproductions/concept_erasure_iccv2023.py` | Gandikota et al. | ICCV 2023 |

---

### Priority 3: Missing Documentation from Specification

#### 3.1 API Reference Docs
| File | Description |
|------|-------------|
| `docs/api/strategies.rst` | Strategies API reference |
| `docs/api/selectors.rst` | Selectors API reference |
| `docs/api/metrics.rst` | Metrics API reference |
| `docs/api/data.rst` | Data module API reference |
| `docs/api/visualization.rst` | Visualization API reference |
| `docs/api/certification.rst` | Certification API reference |
| `docs/api/privacy.rst` | Privacy API reference |
| `docs/api/utils.rst` | Utils API reference |

#### 3.2 User Guide
| File | Description |
|------|-------------|
| `docs/guide/overview.rst` | Architecture overview |
| `docs/guide/unlearning_pipeline.rst` | Pipeline walkthrough |
| `docs/guide/strategies.rst` | Strategy selection guide |
| `docs/guide/selectors.rst` | Selector selection guide |
| `docs/guide/metrics.rst` | Metrics user guide |
| `docs/guide/visualization.rst` | Visualization user guide |
| `docs/user_guide/configuration.rst` | Configuration deep-dive |
| `docs/user_guide/custom_strategies.rst` | Writing custom strategies |
| `docs/user_guide/debugging.rst` | Debugging guide |
| `docs/user_guide/faq.rst` | FAQ |

#### 3.3 Developer Guide
| File | Description |
|------|-------------|
| `docs/developer_guide/architecture.md` | Internal architecture |
| `docs/developer_guide/adding_models.md` | How to add new models |
| `docs/developer_guide/adding_selectors.md` | How to add new selectors |
| `docs/developer_guide/testing.md` | Testing guide |

#### 3.4 Research Documentation
| File | Description |
|------|-------------|
| `docs/research/theory.md` | Theoretical foundations of machine unlearning |
| `docs/research/coreset_analysis.md` | Coreset selection theory & formal analysis |
| `docs/research/utility_bounds.md` | Formal utility preservation guarantees |
| `docs/research/benchmarks.md` | Experimental results & benchmark comparisons |
| `docs/research/paper_reproductions.md` | How to reproduce SOTA papers |

#### 3.5 Project Metadata Files
| File | Description |
|------|-------------|
| `CITATION.cff` | Citation file for academic use |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CODE_OF_CONDUCT.md` | Community code of conduct |
| `LICENSE` | Apache 2.0 license file |
| `requirements-dev.txt` | Development-only dependencies |
| `setup.py` | Legacy setuptools config (in addition to pyproject.toml) |
| `docs/Makefile` | Sphinx build Makefile |
| `docs/requirements.txt` | Docs build dependencies |
| `docs/changelog.rst` | Version changelog |
| `docs/contributing.rst` | Sphinx-formatted contribution guide |

---

### Priority 4: Missing CI/CD & Infrastructure from Specification

#### 4.1 Additional GitHub Workflows
| File | Description |
|------|-------------|
| `.github/workflows/benchmarks.yml` | Automated benchmarking on commits |
| `.github/workflows/publish-pypi.yml` | PyPI release automation |
| `.github/workflows/publish-docs.yml` | GitHub Pages documentation deployment |
| `.github/workflows/security-scan.yml` | Dependency vulnerability scanning |

#### 4.2 GitHub Templates
| File | Description |
|------|-------------|
| `.github/ISSUE_TEMPLATE/bug_report.md` | Bug report template |
| `.github/ISSUE_TEMPLATE/feature_request.md` | Feature request template |
| `.github/ISSUE_TEMPLATE/research_idea.md` | Research idea template |
| `.github/pull_request_template.md` | PR template |

#### 4.3 Docker Enhancements
| File | Description |
|------|-------------|
| `docker/Dockerfile.gpu` | Dedicated GPU Dockerfile with CUDA base image |
| `docker/requirements.txt` | Docker-specific requirements |

#### 4.4 Scripts
| Script | Description |
|--------|-------------|
| `scripts/download_models.py` | Model downloading utility |
| `scripts/generate_docs.sh` | Documentation builder script |
| `scripts/profile_memory.py` | Memory profiling utility |
| `scripts/distributed_launch.sh` | Multi-GPU DDP launcher |

---

### Priority 5: Missing Tests from Specification

#### 5.1 Unit Tests
| File | Description |
|------|-------------|
| `tests/unit/test_losses.py` | Unit tests for all loss functions |
| `tests/unit/test_models.py` | Unit tests for model wrappers |
| `tests/unit/test_utils.py` | Unit tests for utility modules |

#### 5.2 Integration Tests (per-modality)
| File | Description |
|------|-------------|
| `tests/integration/test_clip_pipeline.py` | Dedicated CLIP pipeline integration test |
| `tests/integration/test_llm_pipeline.py` | Dedicated LLM pipeline integration test |
| `tests/integration/test_diffusion_pipeline.py` | Dedicated diffusion pipeline integration test |
| `tests/integration/test_audio_pipeline.py` | Dedicated audio pipeline integration test |

#### 5.3 Benchmark Tests
| File | Description |
|------|-------------|
| `tests/benchmarks/test_tofu.py` | TOFU dataset loading and benchmark tests |
| `tests/benchmarks/test_muse.py` | MUSE dataset loading and benchmark tests |
| `tests/benchmarks/test_performance.py` | Performance regression tests |
| `tests/benchmarks/test_memory.py` | Memory usage regression tests |

#### 5.4 Regression Tests
| File | Description |
|------|-------------|
| `tests/regression/test_accuracy.py` | Ensure no accuracy degradation across versions |
| `tests/regression/test_reproducibility.py` | Deterministic output verification |

---

### Priority 6: Novel Research Contributions (Framework-Level Innovations)

#### 6.1 Coreset Selection Research
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **Coreset Quality Analyzer** | `selectors/quality_metrics.py` — Compute coverage, diversity, influence concentration of coresets. Novel research tool. | ICLR/NeurIPS |
| **Support Vectors of Forgetting** | Formal proof that unlearning k% influential samples ≈ 100% (bounded utility loss). Core innovation. | ICML |
| **Automated Meta-Selector** | ML-based selector that learns which coreset method works best for which modality + dataset | NeurIPS |

#### 6.2 Cross-Modal Forgetting Research
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **Cross-Modal Interference Analysis** | `visualization/cross_modal.py` — Quantify and visualize how unlearning in one modality affects another | CVPR/ICCV |
| **Decoupled Gradient Flow** | Formal analysis of gradient flow in multi-encoder architectures during unlearning | ICML |
| **Modal Drift Measurement** | Measure encoder drift between vision and text models during unlearning | NeurIPS |

#### 6.3 Utility-Preserving Guarantees
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| ~~**PAC-Learning Bounds**~~ | ~~`certification/bounds.py` — PAC-learning style guarantees~~ | ✅ Implemented |
| ~~**Influence-Based Utility Bounds**~~ | ~~Certified utility bounds via influence function analysis~~ | ✅ Implemented |
| ~~**Certified Unlearning Radius**~~ | ~~Compute unlearning radius~~ | ✅ Implemented |

#### 6.4 Unified Benchmark Framework
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **ErasusBenchmark** | `metrics/benchmarks.py` — Unified benchmark across 5 dimensions with LaTeX tables, radar plots, and statistical tests | NeurIPS D&B |
| **Publication-Ready Output** | Auto-generation of LaTeX tables, radar plots, and statistical significance tests | — |

---

### Priority 7: Configuration & Ecosystem Enhancements

#### 7.1 Hydra Integration
- Integrating `hydra-core` for hierarchical config composition
- Supporting config groups (model, strategy, selector, experiment)
- Enabling command-line config overrides (`model.lr=1e-4`)

#### 7.2 Experiment Configs
| File | Description |
|------|-------------|
| `configs/experiments/clip_unlearning.yaml` | Full CLIP unlearning experiment config |
| `configs/experiments/llm_bias_removal.yaml` | LLM bias removal config |
| `configs/experiments/diffusion_artist.yaml` | Diffusion artist removal config |
| `configs/experiments/federated.yaml` | Federated unlearning config |

#### 7.3 `erasus/version.py`
Should contain version string auto-incremented by CI.

#### 7.4 `erasus/__init__.py` Top-Level API
Enrich the top-level `__init__.py` to expose a clean public API:
```python
from erasus import unlearn, evaluate, benchmark
from erasus.unlearners import ErasusUnlearner, MultimodalUnlearner
```

#### 7.5 PyPI Publishing
- `setup.py` (for backwards compat)
- Proper `pyproject.toml` with `[project.optional-dependencies]` for `gpu`, `dev`, `docs`, `all`

---

### Priority 8: Research Ecosystem & Community Goals

#### 8.1 HuggingFace Integration
- Publish unlearned model checkpoints to HuggingFace Hub
- Support `from_pretrained()` for unlearned models
- Integration with `datasets` library for all benchmark loaders

#### 8.2 Community Infrastructure
| Artifact | Description |
|----------|-------------|
| Published PyPI package | `pip install erasus` |
| Hosted documentation | GitHub Pages / ReadTheDocs |
| Academic paper | `papers/erasus_neurips2025.pdf` |
| Supplementary material | `papers/arxiv_supplementary.pdf` |
| Workshop submission | NeurIPS/ICML workshop on unlearning |

#### 8.3 Success Metrics (from Specification)
| Category | Target |
|----------|--------|
| Test coverage | 90%+ |
| Utility drop vs. retraining | <5% |
| Speedup over full retraining | 10x |
| Model architectures | 15+ (currently 10) |
| Papers at top-tier venues | 3+ |
| GitHub stars | 1000+ |
| Academic citations | 10+ |
| PyPI downloads/month | 100+ |
| Community contributions | 20+ |
| Downstream projects | 5+ |

---

## 📊 GAP SUMMARY BY CATEGORY (Updated Post-Sprint F)

| Category | Implemented | In Spec | Gap | Δ from Sprint A |
|----------|:-----------:|:-------:|:---:|:-----------:|
| **Model Architectures** | 10 | 17+ | 7+ | — |
| **Strategies** | **28** | 29 | **1** | ↓1 |
| **Selectors** | **22** | 22 | **0** | ↓3 ✅ |
| **Losses** | **8** | 9 | **1** | — |
| **Metrics** | **26+** | 26+ | **0** | ↓9 ✅ |
| **Visualization** | **13** | 13 | **0** | ↓3 ✅ |
| **Data Loaders** | 5 | 7 | 2 | — |
| **Synthetic Data** | 1 | 3 | 2 | — |
| **Examples** | 9 | 27+ | 18+ | — |
| **Notebooks** | 0 | 15+ | 15 | — |
| **Benchmark Suites** | 2 | 7+ | 5 | — |
| **Paper Reproductions** | 1 | 4 | 3 | — |
| **CI/CD Workflows** | 1 | 5 | 4 | — |
| **CLI Commands** | **5** | 5 | **0** | — ✅ |
| **Docs Pages** | 6 | 30+ | 24+ | — |
| **Test Files** | **14** | 20+ | 6+ | ↓2 |
| **Utils Modules** | **9** | 9 | **0** | ↓2 ✅ |
| **Privacy Modules** | **6** | 6 | **0** | ↓2 ✅ |
| **Certification** | **3** | 3 | **0** | — ✅ |
| **Experiment Tools** | **3** | 3 | **0** | — ✅ |
| **Unlearners** | **8** | 8 | **0** | ↓1 ✅ |
| **Models** | **18+** | 21 | **3** | ↓9 |
| **Data Modules** | **18+** | 19+ | **1** | ↓5 |

**Total files implemented:** ~196+
**Remaining gap:** ~62 files/modules (down from ~78)
**Sprint B reduced gap by:** ~16 core files + 1 test file

---

## 🗓️ REMAINING IMPLEMENTATION SPRINTS

### ✅ Sprint B: Missing Models & Data — COMPLETED

**16 new files implemented:**

| Category | New Modules | Count |
|----------|------------|:-----:|
| **Models - VLM** | `flamingo.py`, `vision_transformer.py` | 2 |
| **Models - LLM** | `t5.py` | 1 |
| **Models - Diffusion** | `dalle.py`, `imagen.py`, `diffusion_utils.py` | 3 |
| **Models - Audio** | `wav2vec.py`, `clap.py` | 2 |
| **Models - Video** | `video_clip.py` | 1 |
| **Data - Datasets** | `muse.py`, `imagenet.py` | 2 |
| **Data - Augmentation** | `augmentation.py` | 1 |
| **Data - Synthetic** | `bias_generator.py`, `privacy_generator.py` | 2 |
| **Privacy** | `gradient_clipping.py`, `secure_aggregation.py` | 2 |
| **Utils** | `profiling.py`, `reproducibility.py` | 2 |

**Updated `__init__.py` files:** 10 (all model/data/privacy/utils packages)
**New test file:** `tests/unit/test_sprint_b.py` — 85 tests
**Total tests:** 206 passing ✅

### Sprint C: Examples, Benchmarks & Reproductions (Est. 2 days)
- 18+ missing example scripts
- Benchmark: MUSE runner, custom benchmarks (privacy, efficiency, utility)
- Reproductions: SCRUB, SSD, Concept Erasure

### Sprint D: Documentation & Tutorials (Est. 2 days)
- 24+ missing doc pages (API refs, user guide, developer guide, research docs)
- 15+ missing tutorial notebooks

### Sprint E: Testing & CI/CD (Est. 2 days)
- 8+ missing test files
- 4 additional CI/CD workflows
- GitHub templates
- Docker GPU Dockerfile

### ✅ Sprint F: Research Innovations & Ecosystem — COMPLETED

**17 new files implemented:**

| Category | New Modules | Count |
|----------|------------|:-----:|
| **Selectors** | `quality_metrics`, `active_learning`, `weighted_fusion` | 3 |
| **Strategies** | `vision_text_split` | 1 |
| **Metrics** | `benchmarks`, `clip_score`, `extraction_attack`, `bleu`, `rouge`, `inception_score`, `downstream_tasks`, `epsilon_delta`, `privacy_audit` | 9 |
| **Visualization** | `activation`, `influence_maps`, `cross_modal` | 3 |
| **Unlearners** | `federated_unlearner` | 1 |

**Updated files:**
- `strategies/__init__.py` — registers 28 strategies
- `selectors/__init__.py` — registers 22 selectors
- `metrics/__init__.py` — registers 26+ metrics
- `visualization/__init__.py` — registers 13 visualization tools
- `unlearners/__init__.py` — registers 8 unlearner classes
- `selectors/ensemble/__init__.py` — updated
- `metrics/utility/__init__.py` — new sub-package init

**New test file:** `tests/unit/test_sprint_f.py` — 34 tests

### Sprint G: Publishing & Community (Est. 2 days)
- CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, LICENSE
- version.py, enriched __init__.py
- PyPI publishing workflow
- ReadTheDocs deployment
- HuggingFace integration
