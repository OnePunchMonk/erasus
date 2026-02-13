---
description: Comprehensive agenda for the Erasus framework — gap analysis vs. full specification + future enhancements
---

# Erasus Framework — Comprehensive Agenda

**Last Updated:** 2026-02-14
**Test Status:** 87 / 87 passing ✅

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

### Selectors (19 implementations) ✅
- **Gradient-based:** `influence.py`, `tracin.py`, `gradient_norm.py`, `grad_match.py`, `el2n.py`, `representer.py`, `forgetting_score.py`
- **Geometry-based:** `kcenter.py`, `herding.py`, `craig.py`, `glister.py`, `submodular.py`, `kmeans_coreset.py`, `k_center.py`
- **Learning-based:** `forgetting_events.py`, `data_shapley.py`, `valuation_network.py`, `loss_accum.py`
- **Ensemble:** `voting.py`
- **Utility:** `auto_selector.py`, `random_selector.py`, `full_selector.py`

### Strategies (20 implementations) ✅
- **Gradient methods:** `gradient_ascent.py`, `scrub.py`, `modality_decoupling.py`, `fisher_forgetting.py`, `negative_gradient.py`
- **Parameter methods:** `lora_unlearning.py`, `sparse_aware.py`, `mask_based.py`, `neuron_pruning.py`
- **Data methods:** `amnesiac.py`, `sisa.py`, `certified_removal.py`
- **LLM-specific:** `ssd.py`, `token_masking.py`, `embedding_alignment.py`, `causal_tracing.py`
- **Diffusion-specific:** `concept_erasure.py`, `noise_injection.py`, `unet_surgery.py`
- **VLM-specific:** `contrastive_unlearning.py`, `cross_modal_decoupling.py`

### Losses (5 implementations) ✅
- `retain_anchor.py`, `contrastive.py`, `kl_divergence.py`, `mmd.py`, `custom_losses.py`

### Unlearner API (7 classes) ✅
- `erasus_unlearner.py`, `vlm_unlearner.py`, `llm_unlearner.py`, `diffusion_unlearner.py`
- `audio_unlearner.py`, `video_unlearner.py`, `multimodal_unlearner.py`

### Metrics (13 implementations) ✅
- **Flat:** `accuracy.py`, `membership_inference.py`, `perplexity.py`, `retrieval.py`, `fid.py`, `retrieval_metrics.py`
- **Suite:** `metric_suite.py`
- **Forgetting:** `mia.py`, `mia_variants.py`, `confidence.py`, `feature_distance.py`
- **Efficiency:** `time_complexity.py`, `memory_usage.py`
- **Privacy:** `differential_privacy.py`

### Visualization (8 files) ✅
- `embeddings.py`, `surfaces.py`, `gradients.py`, `reports.py`, `interactive.py`
- `loss_curves.py`, `feature_plots.py`, `mia_plots.py`

### Data Module ✅
- **Datasets:** `tofu.py`, `wmdp.py`, `coco.py`, `i2p.py`, `conceptual_captions.py`
- **Utils:** `preprocessing.py`, `partitioning.py`, `samplers.py`, `loaders.py`, `transforms.py`, `splits.py`, `datasets.py`, `multimodal.py`
- **Synthetic:** `backdoor_generator.py`

### Privacy Module ✅
- `accountant.py`, `dp_mechanisms.py`, `certificates.py`, `influence_bounds.py`

### Certification Module ✅
- `certified_removal.py`, `verification.py`

### CLI ✅
- `main.py`, `unlearn.py`, `evaluate.py`

### Utils ✅
- `checkpointing.py`, `distributed.py`, `helpers.py`, `logging.py`, `seed.py`

### Experiments ✅
- `experiment_tracker.py` (local/W&B/MLflow)

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

### Tests (87 passing) ✅

---

## 🔮 FUTURE ENHANCEMENTS — COMPREHENSIVE ROADMAP

### Priority 1: Missing Modules from Specification (High Impact)

#### 1.1 Missing Model Architectures
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `models/vlm/flamingo.py` | `models/vlm/` | Flamingo VLM adapter | 🔴 High |
| `models/vlm/vision_transformer.py` | `models/vlm/` | ViT helper utilities | 🟡 Medium |
| `models/llm/t5.py` | `models/llm/` | T5 encoder-decoder model wrapper | 🔴 High |
| `models/diffusion/dalle.py` | `models/diffusion/` | DALL-E 2/3 model wrapper | 🟡 Medium |
| `models/diffusion/imagen.py` | `models/diffusion/` | Imagen model wrapper | 🟡 Medium |
| `models/diffusion/diffusion_utils.py` | `models/diffusion/` | Noise schedulers, diffusion helpers | 🔴 High |
| `models/audio/wav2vec.py` | `models/audio/` | Wav2Vec 2.0 model wrapper | 🟡 Medium |
| `models/audio/clap.py` | `models/audio/` | CLAP audio-text model | 🟢 Low |
| `models/video/video_clip.py` | `models/video/` | VideoCLIP model wrapper | 🟡 Medium |
| DINOv2 | Not in spec | Self-supervised vision model | 🟢 Low |
| Segment Anything (SAM) | Not in spec | Foundation segmentation model | 🟢 Low |
| Gemma / Phi | Not in spec | Smaller LLMs for efficient unlearning | 🟡 Medium |

#### 1.2 Missing Strategies
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `strategies/gradient_methods/saliency_unlearning.py` | `gradient_methods/` | Saliency-guided gradient unlearning | 🔴 High |
| `strategies/parameter_methods/layer_freezing.py` | `parameter_methods/` | Selective layer freezing during unlearning | 🔴 High |
| `strategies/data_methods/knowledge_distillation.py` | `data_methods/` | Teacher-student unlearning via KD | 🔴 High |
| `strategies/llm_specific/attention_surgery.py` | `llm_specific/` | Direct attention weight modification | 🟡 Medium |
| `strategies/diffusion_specific/timestep_masking.py` | `diffusion_specific/` | Selective timestep training for diffusion | 🟡 Medium |
| `strategies/diffusion_specific/safe_latents.py` | `diffusion_specific/` | Safe Latent Diffusion (SLD) | 🟡 Medium |
| `strategies/vlm_specific/attention_unlearning.py` | `vlm_specific/` | Cross-attention modification for VLMs | 🔴 High |
| `strategies/vlm_specific/vision_text_split.py` | `vlm_specific/` | Separate encoder update strategy | 🟡 Medium |
| `strategies/ensemble_strategy.py` | `strategies/` | Combine multiple unlearning strategies | 🔴 High |

#### 1.3 Missing Selectors
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `selectors/learning_based/active_learning.py` | `learning_based/` | Uncertainty-based active selection | 🟡 Medium |
| `selectors/ensemble/weighted_fusion.py` | `ensemble/` | Weighted combination of selectors | 🟡 Medium |
| Coreset quality metrics | `selectors/quality_metrics.py` | Coverage, diversity, influence concentration analysis | 🔴 High |

#### 1.4 Missing Losses
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `losses/fisher_regularization.py` | `losses/` | Fisher information regularization penalty | 🔴 High |
| `losses/l2_regularization.py` | `losses/` | Simple weight decay / L2 penalty loss | 🟢 Low |
| `losses/adversarial_loss.py` | `losses/` | GAN-style adversarial unlearning loss | 🟡 Medium |
| `losses/triplet_loss.py` | `losses/` | Triplet-based separation in embedding space | 🟡 Medium |

#### 1.5 Missing Metrics
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `metrics/forgetting/activation_analysis.py` | `forgetting/` | Internal activation analysis for forgetting verification | 🔴 High |
| `metrics/forgetting/backdoor_activation.py` | `forgetting/` | Backdoor success rate metric (Liu et al., ICLR 2022) | 🔴 High |
| `metrics/forgetting/extraction_attack.py` | `forgetting/` | Data extraction attack metric | 🟡 Medium |
| `metrics/utility/bleu.py` | `utility/` | BLEU translation quality metric | 🟡 Medium |
| `metrics/utility/rouge.py` | `utility/` | ROUGE summarization metric | 🟡 Medium |
| `metrics/utility/clip_score.py` | `utility/` | CLIP similarity score | 🔴 High |
| `metrics/utility/inception_score.py` | `utility/` | Inception Score for generation quality | 🟡 Medium |
| `metrics/utility/downstream_tasks.py` | `utility/` | Task-specific downstream evaluation | 🟢 Low |
| `metrics/efficiency/flops.py` | `efficiency/` | FLOPs estimation for unlearning cost | 🟡 Medium |
| `metrics/efficiency/speedup.py` | `efficiency/` | Speedup ratio vs. full retraining | 🔴 High |
| `metrics/privacy/epsilon_delta.py` | `privacy/` | (ε, δ)-DP computation module | 🟡 Medium |
| `metrics/privacy/privacy_audit.py` | `privacy/` | Privacy auditing framework | 🟡 Medium |
| `metrics/benchmarks.py` | `metrics/` | Unified benchmark runner (publication-ready output) | 🔴 High |
| Metrics `utility/` sub-package init | `metrics/utility/` | Reorganize flat accuracy/perplexity/fid into `utility/` sub-package | 🟡 Medium |

#### 1.6 Missing Visualization
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `visualization/attention.py` | `visualization/` | Attention heatmap visualization | 🔴 High |
| `visualization/activation.py` | `visualization/` | Layer activation visualization | 🔴 High |
| `visualization/influence_maps.py` | `visualization/` | Influence attribution visualization | 🟡 Medium |
| `visualization/comparisons.py` | `visualization/` | Before/after comparison plots | 🔴 High |
| `visualization/cross_modal.py` | Novel research | Cross-modal interference visualization | 🔴 High |

#### 1.7 Missing Data Components
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `data/augmentation.py` | `data/` | Data augmentation strategies for unlearning | 🟡 Medium |
| `data/datasets/imagenet.py` | `datasets/` | ImageNet variants loader | 🟡 Medium |
| `data/datasets/laion.py` | `datasets/` | LAION subset loaders | 🟢 Low |
| `data/datasets/muse.py` | `datasets/` | MUSE benchmark dataset | 🔴 High |
| `data/synthetic/bias_generator.py` | `synthetic/` | Synthetic bias injection for fairness | 🟡 Medium |
| `data/synthetic/privacy_generator.py` | `synthetic/` | Privacy-sensitive synthetic data | 🟡 Medium |

#### 1.8 Missing Privacy Components
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `privacy/gradient_clipping.py` | `privacy/` | Per-sample gradient clipping for DP-SGD | 🔴 High |
| `privacy/secure_aggregation.py` | `privacy/` | Secure aggregation for federated privacy | 🟡 Medium |

#### 1.9 Missing Certification
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `certification/bounds.py` | `certification/` | Theoretical PAC-learning style guarantees and utility bounds | 🔴 High |

#### 1.10 Missing Utils
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `utils/profiling.py` | `utils/` | Performance profiling (GPU utilization, bottleneck analysis) | 🟡 Medium |
| `utils/reproducibility.py` | `utils/` | Extended reproducibility utilities (beyond seed.py) | 🟢 Low |
| `utils/callbacks.py` | `utils/` | Training callbacks (LR scheduling, logging, etc.) | 🔴 High |
| `utils/early_stopping.py` | `utils/` | Early stopping logic for unlearning convergence | 🔴 High |

#### 1.11 Missing CLI Commands
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `cli/benchmark.py` | `cli/` | `erasus benchmark` command | 🔴 High |
| `cli/visualize.py` | `cli/` | `erasus visualize` command | 🟡 Medium |

#### 1.12 Missing Experiment Tools
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `experiments/hyperparameter_search.py` | `experiments/` | Optuna / Ray Tune integration for HPO | 🔴 High |
| `experiments/ablation_studies.py` | `experiments/` | Automated ablation study runner | 🔴 High |

#### 1.13 Missing Unlearners
| Module | Spec Path | Description | Priority |
|--------|-----------|-------------|----------|
| `unlearners/federated_unlearner.py` | `unlearners/` | Federated unlearning orchestrator | 🟡 Medium |

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

These are research contributions described in the specification that should be **built into the framework** as first-class features:

#### 6.1 Coreset Selection Research
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **Coreset Quality Analyzer** | `selectors/quality_metrics.py` — Compute coverage, diversity, influence concentration of coresets. Novel research tool for analyzing geometric properties. | ICLR/NeurIPS |
| **Support Vectors of Forgetting** | Formal proof that unlearning k% influential samples ≈ 100% (bounded utility loss). Core innovation of the framework. | ICML |
| **Automated Meta-Selector** | ML-based selector that learns which coreset method works best for which modality + dataset automatically | NeurIPS |

#### 6.2 Cross-Modal Forgetting Research
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **Cross-Modal Interference Analysis** | `visualization/cross_modal.py` — Quantify and visualize how unlearning in one modality affects another (vision ↔ text leakage) | CVPR/ICCV |
| **Decoupled Gradient Flow** | Formal analysis of gradient flow in multi-encoder architectures during unlearning | ICML |
| **Modal Drift Measurement** | Measure encoder drift between vision and text models during unlearning | NeurIPS |

#### 6.3 Utility-Preserving Guarantees
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **PAC-Learning Bounds** | `certification/bounds.py` — PAC-learning style guarantees for post-unlearning utility preservation | COLT/ALT |
| **Influence-Based Utility Bounds** | Certified utility bounds via influence function analysis | ICML |
| **Certified Unlearning Radius** | Extend certified training concepts to compute unlearning radius | S&P/USENIX |

#### 6.4 Unified Benchmark Framework
| Contribution | Description | Target Venue |
|-------------|-------------|-------------|
| **ErasusBenchmark** | `metrics/benchmarks.py` — Unified benchmark across 5 dimensions (forgetting efficacy, utility, efficiency, privacy, scalability) with LaTeX table generation, radar plots, and statistical tests | NeurIPS Datasets & Benchmarks |
| **Publication-Ready Output** | Auto-generation of LaTeX tables, radar plots, and running statistical significance tests (Wilcoxon, paired t-test) | — |

---

### Priority 7: Configuration & Ecosystem Enhancements

#### 7.1 Hydra Integration
The spec mentions using Hydra for configuration management. Current implementation uses simple YAML loading via `ErasusConfig`. Consider:
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
The spec calls for a `version.py` module. Currently absent — should contain version string auto-incremented by CI.

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

## 📊 GAP SUMMARY BY CATEGORY

| Category | Implemented | In Spec | Gap |
|----------|:-----------:|:-------:|:---:|
| **Model Architectures** | 10 | 17+ | 7+ |
| **Strategies** | 20 | 29 | 9 |
| **Selectors** | 19 | 22 | 3 |
| **Losses** | 5 | 9 | 4 |
| **Metrics** | 13 | 26+ | 13+ |
| **Visualization** | 8 | 13 | 5 |
| **Data Loaders** | 5 | 7 | 2 |
| **Synthetic Data** | 1 | 3 | 2 |
| **Examples** | 9 | 27+ | 18+ |
| **Notebooks** | 0 | 15+ | 15 |
| **Benchmark Suites** | 2 | 7+ | 5 |
| **Paper Reproductions** | 1 | 4 | 3 |
| **CI/CD Workflows** | 1 | 5 | 4 |
| **CLI Commands** | 3 | 5 | 2 |
| **Docs Pages** | 6 | 30+ | 24+ |
| **Test Files** | 12 | 20+ | 8+ |
| **Utils Modules** | 5 | 8 | 3 |
| **Privacy Modules** | 4 | 6 | 2 |
| **Certification** | 2 | 3 | 1 |
| **Experiment Tools** | 1 | 3 | 2 |

**Total Implementation Gap: ~120+ files/modules**

---

## 🗓️ RECOMMENDED IMPLEMENTATION SPRINTS

### Sprint A: Critical Missing Modules (Est. 3 days)
- Missing strategies: `saliency_unlearning`, `layer_freezing`, `knowledge_distillation`, `attention_surgery`, `ensemble_strategy`, `attention_unlearning`
- Missing metrics: `activation_analysis`, `backdoor_activation`, `clip_score`, `speedup`, `benchmarks.py`
- Missing visualization: `attention.py`, `activation.py`, `comparisons.py`, `cross_modal.py`
- Missing utils: `callbacks.py`, `early_stopping.py`
- Missing CLI: `benchmark.py`, `visualize.py`
- Missing certification: `bounds.py`

### Sprint B: Missing Models & Data (Est. 2 days)
- Models: `flamingo`, `t5`, `dalle`, `imagen`, `diffusion_utils`, `wav2vec`, `clap`, `video_clip`
- Data: `muse.py`, `imagenet.py`, `bias_generator.py`, `privacy_generator.py`, `augmentation.py`

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

### Sprint F: Research Innovations & Ecosystem (Est. 3 days)
- Coreset quality analyzer
- Cross-modal interference tools
- PAC-learning bounds
- ErasusBenchmark unified runner
- Hyperparameter search (Optuna)
- Ablation study automation
- Federated unlearner

### Sprint G: Publishing & Community (Est. 2 days)
- CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, LICENSE
- version.py, enriched __init__.py
- PyPI publishing workflow
- ReadTheDocs deployment
- HuggingFace integration
