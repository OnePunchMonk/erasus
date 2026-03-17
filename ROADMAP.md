# Erasus Roadmap: From Alpha to Adopted

> Deep research analysis of the machine unlearning landscape (2024–2026) and a concrete plan to make Erasus the standard framework for the field.
>
> Last updated: 2026-03-17

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Where Erasus Stands Today](#2-where-erasus-stands-today)
3. [The 2024–2025 Research Landscape](#3-the-20242025-research-landscape)
4. [Competitive Analysis](#4-competitive-analysis)
5. [The Evaluation Crisis](#5-the-evaluation-crisis)
6. [Missing SOTA Methods](#6-missing-sota-methods)
7. [Missing Capabilities](#7-missing-capabilities)
8. [Architecture & Engineering Improvements](#8-architecture--engineering-improvements)
9. [Visibility & Community Strategy](#9-visibility--community-strategy)
10. [Prioritised Action Plan](#10-prioritised-action-plan)
11. [Positioning & Narrative](#11-positioning--narrative)
12. [Key References](#12-key-references)

---

## 1. Executive Summary

Erasus is the **only framework attempting unified cross-modality unlearning** (LLMs, VLMs, Diffusion, Audio, Video) with an intelligent coreset selection layer. No competitor covers this breadth. However:

- **The field has moved fast.** Preference-based methods (NPO, SimNPO, AltPO) are now SOTA for LLM unlearning. Erasus has none of them.
- **Evaluation is in crisis.** ICLR 2025 and CVPR 2025 papers showed that existing unlearning methods are fragile — benchmarks give false confidence, erased concepts revive under fine-tuning, and benign data can "jog" model memory. The framework that solves evaluation wins.
- **Nobody knows Erasus exists.** Zero public presence, no academic paper, not listed anywhere. The technical work is ~60% there; the community work is 0% there.

The path forward has three pillars:

1. **Visibility** — publish a paper, go public, get listed
2. **Evaluation rigour** — adversarial stress-tests, relearning attacks, 6-MIA suite
3. **SOTA methods** — preference-based unlearning, inference-time unlearning, continual unlearning

---

## 2. Where Erasus Stands Today

### What exists (v0.1.1)

| Component | Count | Details |
|-----------|-------|---------|
| Unlearning strategies | 27 | Gradient (6), Parameter (5), Data (4), LLM-specific (5), Diffusion-specific (5), VLM-specific (3), Ensemble (1) |
| Coreset selectors | 24 | Gradient-based (7), Geometry-based (6), Learning-based (4), Ensemble (3), Auto/Random/Full (3) |
| Evaluation metrics | 25+ | Forgetting (8), Utility (7), Efficiency (4), Privacy (4+) |
| Loss functions | 8 | Retain-anchor, Contrastive, KL, MMD, Fisher, Adversarial, Triplet, L2 |
| Model wrappers | 10+ | CLIP, LLaVA, BLIP, Flamingo, LLaMA, GPT, T5, StableDiffusion, DALL-E, Whisper, VideoMAE |
| Modality unlearners | 7 | Generic, VLM, LLM, Diffusion, Audio, Video, Federated + MultimodalDispatcher |
| Benchmarks | 3 | TOFU, MUSE, WMDP (synthetic data runners) |
| Tests | 340+ | 17 test files, unit + integration + end-to-end |
| Notebooks | 7 | Introduction, coreset analysis, CLIP, GPT-2, Stable Diffusion demos |
| Examples | 30+ | All modalities covered |
| CLI | 4 commands | `unlearn`, `evaluate`, `benchmark`, `visualize` |
| Docs | Sphinx | Quickstart, API reference, user guides, developer guide |

### Genuine differentiators

1. **Cross-modality architecture** — LLM + VLM + Diffusion + Audio + Video under one API. No competitor does this.
2. **Coreset selection focus** — 24 selectors with quality analysis. Unique in the ecosystem.
3. **Certification module** — Formal (ε, δ)-removal verification with PAC-style bounds.
4. **Federated unlearning** — Dedicated `FederatedUnlearner` with FedAvg aggregation. No competitor has this.
5. **Registry-based plugin system** — Add strategies/selectors/metrics without modifying core code.

### Known gaps (from `project_ideas.md` and code inspection)

- No preference-based unlearning methods (NPO, SimNPO, etc.)
- Benchmarks use synthetic data and a `BenchmarkModel`, not real models/datasets
- 7+ selectors silently fall back to random selection on failure
- No `.pre-commit-config.yaml` despite pre-commit in dev deps
- No coverage threshold in CI
- Tests only use synthetic `TinyClassifier` / `BenchmarkModel`
- No lm-evaluation-harness integration
- Zero public presence

---

## 3. The 2024–2025 Research Landscape

### Paradigm shifts

The field has undergone three major shifts since Erasus's last update:

#### Shift 1: Preference-based unlearning replaces gradient ascent as SOTA

Gradient ascent on the forget set was the baseline for years. In 2024–2025, preference optimisation methods took over:

| Method | Venue | Innovation |
|--------|-------|------------|
| NPO (Negative Preference Optimization) | 2024 | Reference model constrains parameter drift |
| SimNPO | NeurIPS 2025 | Removes reference model dependency; identifies "reference model bias" in NPO |
| AltPO (Alternate Preference Optimization) | 2024 | Positive feedback for alternative answers + negative for forget set |
| FLAT (Loss Adjustment) | ICLR 2025 | Forget-data-only; no retain data or reference model needed |
| RMU (Representation Misdirection) | 2024 | Manipulates internal representations; strong on WMDP |
| UNDIAL | 2024 | Self-distillation with logit adjustment for stability |

These methods produce coherent alternative responses instead of the nonsensical output that gradient ascent creates.

#### Shift 2: Fundamental scepticism about approximate unlearning

Multiple 2025 papers challenged whether current methods work at all:

- **"Machine Unlearning Fails to Remove Data Poisoning Attacks"** (ICLR 2025) — existing methods fail across all poison types for both image classifiers and LLMs. Conclusion: "not yet ready for prime time."

- **"Unlearning or Obfuscating?"** (ICLR 2025) — finetuning-based approximate unlearning merely *obfuscates* outputs. A small amount of benign auxiliary data (e.g. public medical articles) can "jog" model memory to output harmful knowledge about bioweapons.

- **"The Illusion of Unlearning"** (CVPR 2025) — ALL concept erasure methods for diffusion models (SalUn, ESD, EDiff, CA, MACE, SPM, SA, Receler, UCE) are unstable. Erased concepts revive under downstream fine-tuning.

- **"LLM Unlearning Benchmarks are Weak Measures of Progress"** (CMU, April 2025) — TOFU and WMDP are fundamentally flawed. Combining forget + retain queries in one prompt resurfaces "unlearned" info. Inserting a forget keyword into a wrong MCQ option causes 28% accuracy drop. Models are fragile, not forgetful.

#### Shift 3: Inference-time and continual unlearning emerge

- **DExperts**: Expert/anti-expert ensemble at decode time. No gradient computation, no model modification. Works on black-box APIs.
- **delta-UNLEARNING**: Computes logit offsets using small white-box proxy models, applied to black-box LLMs.
- **FIT (2025)**: Addresses catastrophic forgetting during sequential deletion requests. First serious treatment of continual unlearning.
- **Meta-Unlearning (ICCV 2025)**: Prevents relearning of erased diffusion concepts via meta-learning.

### NeurIPS 2023 Machine Unlearning Challenge: Lessons

- 1,338 participants, 72 countries, 1,923 submissions
- Task: unlearn face images from an age predictor
- Winning pattern: **"erase then repair"** — erase via layer reinitialization/noise injection/gradient modification, then repair via retain-set finetuning with KL-divergence or entropy regularisation
- Critical finding: simple metrics (accuracy gap) poorly correlate with principled evaluation. Generalization across datasets was poor.

### Regulatory pressure (active)

- **GDPR Article 17** ("Right to Erasure"): Applies but lacks clear guidelines for model-embedded data
- **EU AI Act** (phased enforcement): Feb 2025 — prohibited practices; Aug 2025 — GPAI obligations; Aug 2027 — high-risk system rules. Unlearning is proposed as a mandated safeguard.
- No universally accepted verification method for unlearning effectiveness yet — whoever standardises this wins.

---

## 4. Competitive Analysis

### Direct competitors

| Framework | Stars | Scope | Venue | Strengths | Weaknesses |
|-----------|-------|-------|-------|-----------|------------|
| **OpenUnlearning** (CMU/Locuslab) | 504 | LLM only | NeurIPS D&B 2025 | 13 methods (incl. NPO, SimNPO, AltPO, RMU), 16 metrics (6 MIA types), 3 benchmarks, 450+ HF checkpoints, Hydra configs | LLM only; no coreset selection; no diffusion/VLM/audio/video |
| **EasyEdit** (ZJU NLP) | 2,700 | Knowledge editing | ACL 2024 | ROME, MEMIT, 15+ methods, HF Spaces demo, massive community | Not unlearning (editing ≠ forgetting); different paradigm |
| **OpenUnlearn** (Pawelczyk, CMU) | ~100 | Evaluation | ICLR 2025 | Rigorous poison-based evaluation; 8 algorithms, 3 metrics | Evaluation-only tool |
| **ZJUNLP Unlearn** | ~50 | LLM | ACL 2025 | ReLearn method, KnowUnDo benchmark, MemFlex | Narrow scope; single research group |
| **ESD/erasing** (Baulab) | ~500 | Diffusion only | ICCV 2023 | SD, SDXL, FLUX concept erasure | Diffusion only; shown unstable by CVPR 2025 |
| **FLAT** (UCSC-REAL) | ~50 | LLM | ICLR 2025 | Forget-data-only; no retain data needed | Single method; no framework |
| **Erasus** | 0 (private) | All modalities | None | Breadth, coreset selection, certification, federated | Zero public presence; missing SOTA methods |

### What OpenUnlearning has that Erasus doesn't

This is the primary competitor and the gap analysis matters:

| Feature | OpenUnlearning | Erasus |
|---------|---------------|--------|
| NPO / SimNPO / AltPO | Yes | No |
| RMU | Yes | No |
| UNDIAL | Yes | No |
| WGA (weighted gradient ascent) | Yes | No |
| 6 MIA attack types (LOSS, ZLib, Reference, GradNorm, MinK, MinK++) | Yes | Only basic MIA + LiRA |
| Extraction Strength (ES) metric | Yes | No |
| Exact Memorization (EM) metric | Yes | No |
| PrivLeak metric | Yes | No |
| lm-evaluation-harness integration (MMLU, GSM8K, etc.) | Yes | No |
| Hydra config-driven experiments | Yes | YAML configs exist but not Hydra-level |
| 450+ public HF checkpoints | Yes | No |
| Real TOFU/MUSE/WMDP benchmark runs | Yes | Synthetic data only |
| Published leaderboards with real results | Yes | Leaderboards exist but on synthetic data |
| System paper at top venue | NeurIPS D&B 2025 | None |

### What Erasus has that nobody else does

| Feature | Erasus | Competitors |
|---------|--------|-------------|
| Cross-modality support (LLM + VLM + Diffusion + Audio + Video) | Yes | All are single-modality |
| 24 coreset selectors with quality analysis | Yes | None focus on this |
| Certification module with (ε, δ)-removal verification | Yes | None |
| Federated unlearning | Yes | None |
| Ensemble unlearning (strategy combination) | Yes | None |
| 8 loss functions with modular composition | Yes | Hardcoded in others |
| Visualization suite (16 modules) | Yes | Basic in OpenUnlearning |
| Privacy module (DP accounting, secure aggregation) | Yes | None |

---

## 5. The Evaluation Crisis

This deserves its own section because **evaluation is both the biggest unsolved problem and Erasus's biggest opportunity**.

### What's wrong with current evaluation

| Problem | Source | Impact |
|---------|--------|--------|
| TOFU/WMDP test forget and retain independently | CMU Blog, April 2025 | Combining them in one prompt resurfaces "unlearned" info |
| Keyword injection breaks WMDP | CMU Blog, April 2025 | Inserting a forget keyword into wrong MCQ option → 28% accuracy drop |
| Basic MIA (threshold-based) is too weak | OpenUnlearning, NeurIPS D&B 2025 | Passes even when info is still extractable |
| Concept erasure is unstable | CVPR 2025 | Downstream fine-tuning revives erased concepts |
| Benign relearning reverses unlearning | ICLR 2025 | Small amounts of public auxiliary data restore harmful knowledge |
| Quantization breaks unlearning | Emerging 2025 | 4-bit/8-bit quantization can revive erased knowledge |
| Simple metrics don't correlate with principled evaluation | NeurIPS 2023 Challenge | Accuracy gap ≠ actual forgetting quality |

### What Erasus should build

#### 5.1 Adversarial evaluation suite

A new module — `erasus/evaluation/adversarial/` — with:

- **Cross-prompt leakage test**: Combine forget and retain queries in a single prompt. If unlearned info resurfaces, the method failed.
- **Keyword injection test**: Insert forget-set keywords into incorrect MCQ options. Measure accuracy degradation.
- **Paraphrase robustness**: Rephrase forget-set queries. Measure if the model still recalls info under semantic variation.
- **Multilingual leakage**: If model is multilingual, test if unlearned English knowledge is still accessible in other languages.

#### 5.2 Relearning robustness module

A new module — `erasus/evaluation/relearning/` — with:

- **Benign fine-tuning attack**: Fine-tune the unlearned model on a small amount of benign, publicly available data related to the forget domain. Measure if unlearned knowledge returns.
- **Quantization attack**: Quantize the unlearned model to 4-bit and 8-bit. Measure knowledge retention via MIA and generation probing.
- **LoRA relearning**: Attach a LoRA adapter and fine-tune on tangentially related data. Measure concept revival.
- **Prompt engineering attack**: Use chain-of-thought, role-playing, and jailbreak prompts to extract unlearned info.

#### 5.3 Upgraded MIA suite

Expand from basic MIA to the 6-attack standard:

| MIA Type | Description | Status in Erasus |
|----------|-------------|------------------|
| LOSS | Threshold on per-sample loss | Exists (basic MIA) |
| ZLib | Loss normalised by zlib compression ratio | Missing |
| Reference | Loss ratio between target and reference model | Missing |
| GradNorm | Gradient norm of the loss w.r.t. inputs | Missing |
| MinK | Minimum-k% token probabilities | Missing |
| MinK++ | Improved MinK with better calibration | Missing |
| LiRA | Likelihood Ratio Attack | Exists (mia_variants.py) |

#### 5.4 New metrics

| Metric | Description | Status |
|--------|-------------|--------|
| Extraction Strength (ES) | How much of the forget data can be extracted via prompting | Missing |
| Exact Memorization (EM) | Exact string match of forget data in model output | Missing |
| PrivLeak (from MUSE) | Privacy leakage score | Missing |
| ROUGE-L (for unlearning) | Overlap between model output and forget data | Exists but not used for unlearning eval |
| KnowMem (from TOFU) | Knowledge memorisation score | Missing |
| Verbatim Memorisation | Verbatim reproduction of training data | Missing |

#### 5.5 General capability preservation

Integrate **lm-evaluation-harness** to measure retained model capabilities:

- MMLU (general knowledge)
- GSM8K (mathematical reasoning)
- TruthfulQA (truthfulness)
- HellaSwag (commonsense reasoning)
- ARC-Challenge (science QA)

This answers the question: "After unlearning, does the model still work?"

---

## 6. Missing SOTA Methods

### Tier 1: Must-add (SOTA on standard benchmarks)

#### 6.1 NPO — Negative Preference Optimization

- **Paper**: Zhang et al., 2024
- **Key idea**: Treat unlearning as preference optimisation. The forget set provides negative examples. A reference model (the original model before unlearning) constrains how much parameters can drift, preventing catastrophic forgetting.
- **Why it matters**: First method to frame unlearning as a preference problem rather than a loss maximisation problem. Produces coherent alternative responses instead of nonsensical output.
- **Where it fits in Erasus**: `erasus/strategies/llm_specific/npo.py`

#### 6.2 SimNPO — Simplified NPO (NeurIPS 2025)

- **Paper**: Fan et al., 2025
- **Key idea**: Identifies "reference model bias" in NPO — the reference model gives a misleading impression of unlearning effectiveness. Removes reference model dependency entirely. Simpler, faster, and outperforms NPO.
- **Why it matters**: Current SOTA on multiple benchmarks. The name "Simplicity Prevails" captures the insight that simpler is better.
- **Where it fits in Erasus**: `erasus/strategies/llm_specific/simnpo.py`

#### 6.3 AltPO — Alternate Preference Optimization

- **Paper**: Choi et al., 2024
- **Key idea**: Combines negative feedback (forget set) with in-domain positive feedback. The model learns to produce coherent alternative answers to forget-set queries instead of refusing or producing garbage.
- **Why it matters**: Solves the "utility collapse" problem where aggressive unlearning makes the model useless for related (but non-forget) queries.
- **Where it fits in Erasus**: `erasus/strategies/llm_specific/altpo.py`

#### 6.4 FLAT — LLM Unlearning via Loss Adjustment (ICLR 2025)

- **Paper**: Li et al., 2025
- **Key idea**: Eliminates the need for retain data or a reference model. Uses only the forget data to guide unlearning: teaches the model *what not to respond to* and *how to respond* instead. Two components: IDK loss (respond with "I don't know") and Maintain loss (preserve general capabilities via self-distillation).
- **Why it matters**: Extremely practical. In real scenarios, you often don't have access to the original retain data or a reference model.
- **Where it fits in Erasus**: `erasus/strategies/llm_specific/flat.py`

#### 6.5 RMU — Representation Misdirection for Unlearning

- **Paper**: Li et al., 2024
- **Key idea**: Instead of modifying outputs, manipulate internal representations. For forget-set inputs, steer hidden representations toward random vectors. For retain-set inputs, preserve original representations via a contrastive objective.
- **Why it matters**: Strong performer on WMDP. Operates at a deeper level than output-based methods.
- **Where it fits in Erasus**: `erasus/strategies/llm_specific/rmu.py`

### Tier 2: Should-add (important for completeness and production scenarios)

#### 6.6 UNDIAL — Self-Distillation Logit Adjustment

- **Paper**: Dong et al., 2024
- **Key idea**: Uses self-distillation to reduce target-token confidence via logit adjustment. Avoids the instability of gradient ascent by operating in logit space.
- **Where it fits**: `erasus/strategies/llm_specific/undial.py`

#### 6.7 DExperts — Inference-Time Unlearning

- **Paper**: Liu et al., 2024
- **Key idea**: No gradient computation. Combine an "expert" model (retain capabilities) and an "anti-expert" model (trained on forget data) at inference time. Recalculate token probabilities at each decoding step: P(token) = P_base(token) + α(P_expert(token) - P_anti-expert(token)).
- **Why it matters**: Works on black-box models. No model modification needed. Instantly reversible.
- **Where it fits**: `erasus/strategies/llm_specific/dexperts.py` (or a new `inference_time/` category)

#### 6.8 WGA/FPGA — Token-Weighted Gradient Ascent

- **Paper**: Various, 2024
- **Key idea**: Instead of uniform gradient ascent on all tokens, weight the gradient by per-token importance. Tokens more associated with the forget concept get higher weight. Fine-grained control over what gets unlearned.
- **Where it fits**: `erasus/strategies/gradient_methods/weighted_gradient_ascent.py`

#### 6.9 delta-UNLEARNING — Logit Offsets for Black-Box LLMs

- **Paper**: Goel et al., 2024
- **Key idea**: Train a small white-box proxy model to compute logit offsets. Apply these offsets to the black-box model's outputs at inference time. No access to the target model's parameters needed.
- **Where it fits**: `erasus/strategies/llm_specific/delta_unlearning.py`

#### 6.10 Meta-Unlearning on Diffusion Models (ICCV 2025)

- **Paper**: Gao et al., 2025
- **Key idea**: Addresses the relearning vulnerability in diffusion models. Uses meta-learning to make unlearning robust — the model learns to resist concept relearning even when fine-tuned on related data.
- **Why it matters**: CVPR 2025 showed ALL existing diffusion erasure methods are unstable. This is the first defence.
- **Where it fits**: `erasus/strategies/diffusion_specific/meta_unlearning.py`

### Tier 3: Nice-to-have (emerging or niche)

| Method | Type | Innovation |
|--------|------|------------|
| Sparsity-Aware Unlearning (SAU) | Parameter | Decouples unlearning from sparsification via gradient masking |
| FIT (continual unlearning) | Framework | Anti-catastrophic forgetting for sequential deletion |
| MLLMEraser | Multimodal | Test-time activation steering for multimodal LLMs |
| AdvUnlearn | Diffusion | Adversarial training for robust concept erasure |
| Adaptive Guided Erasure (ICLR 2025) | Diffusion | Adaptive guidance for concept erasure |

---

## 7. Missing Capabilities

### 7.1 Continual unlearning

**Problem**: Real-world deployment means sequential deletion requests over time (user A requests deletion on Monday, user B on Tuesday, ...). Current methods assume a single forget set. Repeated application causes catastrophic forgetting of general capabilities.

**What to build**:
- `erasus/unlearners/continual_unlearner.py` — orchestrates sequential unlearning requests
- Incremental coreset update (don't recompute coresets from scratch each time)
- Strategy scheduling (adapt learning rate, epochs per deletion request)
- Catastrophic forgetting detection (monitor general capability metrics between requests)
- Implements ideas from FIT (2025) — the first serious treatment of this problem

**Why it matters**: First-mover advantage. No framework handles this well.

### 7.2 Inference-time unlearning

**Problem**: Many production models are served via APIs (OpenAI, Anthropic, etc.). You can't modify weights. You need unlearning at inference time.

**What to build**:
- `erasus/strategies/inference_time/` — new strategy category
- DExperts-style expert/anti-expert ensembling
- delta-UNLEARNING logit offset approach
- Activation steering (manipulation of hidden states during forward pass)
- Compatible with the existing `BaseStrategy` interface but with a `requires_training = False` flag

**Why it matters**: Bridges the gap between research (full model access) and production (API access only).

### 7.3 Real benchmark integration

**Problem**: Current benchmarks use synthetic data and a `BenchmarkModel` (32→64→64→10 FC network). Published results on synthetic data are not comparable to the field.

**What to build**:
- Real TOFU integration: `locuslab/TOFU` dataset, Llama-2-7B / Phi-1.5 / GPT-2 models
- Real WMDP integration: `cais/wmdp` dataset with Zephyr-7B
- Real MUSE integration: proper 6-way evaluation
- Maintain synthetic benchmarks for CI (fast, deterministic) but add real-model benchmarks for paper results
- Automated leaderboard generation with reproducibility hashes

### 7.4 lm-evaluation-harness integration

**Problem**: After unlearning, you need to verify the model still works on general tasks. No way to measure this currently.

**What to build**:
- `erasus/integrations/lm_eval.py` — wrapper around EleutherAI's lm-evaluation-harness
- Default evaluation suite: MMLU, GSM8K, TruthfulQA, HellaSwag, ARC-Challenge
- Pre/post unlearning comparison with delta tracking
- Integration with `MetricSuite` so it runs automatically during `evaluate()`

### 7.5 HuggingFace Hub integration (checkpoints)

**Problem**: Researchers need pre-trained and unlearned model pairs to reproduce results. OpenUnlearning has 450+ checkpoints.

**What to build**:
- Push unlearned model checkpoints to HuggingFace Hub for each benchmark × strategy combination
- Standardised model cards with unlearning metadata (strategy, selector, prune ratio, metrics)
- One-line loading: `model = erasus.from_pretrained("erasus/tofu-llama2-7b-npo")`
- Version-controlled with dataset + config hashes for exact reproducibility

---

## 8. Architecture & Engineering Improvements

### 8.1 Silent selector fallbacks

**Problem**: 7+ selectors (`ForgettingEventsSelector`, `ValuationNetworkSelector`, `GlisterSelector`, `DataShapleySelector`, etc.) silently fall back to random selection with just a `warnings.warn()`. This masks real failures.

**Fix**: Default to raising an error. Add an explicit `fallback="random"` parameter that users must opt into. Log fallback events to experiment tracker.

### 8.2 Test coverage

**Problem**: No coverage threshold in CI. All 340+ tests use only synthetic models. Coverage can silently decline.

**Fix**:
- Add `--cov-fail-under=80` to pytest config
- Add integration tests with real tiny models (`gpt2`, `openai/clip-vit-base-patch16`) — even one test per modality
- Add a `tests/real_models/` directory with `@pytest.mark.slow` markers for CI-optional real model tests

### 8.3 Pre-commit hooks

**Problem**: `pre-commit>=3.4` is in dev deps but there's no `.pre-commit-config.yaml`.

**Fix**: Add `.pre-commit-config.yaml` with ruff, mypy, and a minimal test run.

### 8.4 Config system upgrade

**Problem**: YAML configs exist but lack the composition and override capabilities that Hydra provides. OpenUnlearning uses Hydra.

**Options**:
- **Option A**: Adopt Hydra (full compatibility with OpenUnlearning's config patterns, but heavy dependency)
- **Option B**: Extend the existing YAML system with composition and CLI overrides (lighter, but less ecosystem compatibility)
- **Recommendation**: Option A for paper reproducibility; researchers expect Hydra in 2025

### 8.5 Error handling consistency

**Problem**: Mixed patterns — some modules raise, some warn, some silently fall back. No standardised error hierarchy.

**Fix**: Establish a clear policy:
- **Hard errors**: Invalid configuration, missing required parameters, incompatible model/strategy combinations
- **Warnings with fallback**: Optional features unavailable (e.g. wandb not installed), selector degradation
- **Silent**: Debug-level information (loss values, timing)

### 8.6 API stabilisation

**Problem**: `pyproject.toml` says "Development Status :: 3 - Alpha". The API may change.

**Plan**:
- Audit the public API surface. Document what is stable vs. experimental.
- Add `@experimental` decorator for unstable APIs that logs a deprecation warning.
- Bump to 0.2.0 before paper submission with a stable core API.

---

## 9. Visibility & Community Strategy

### 9.1 Academic paper (highest priority)

**Target venues** (in order of impact):
1. **NeurIPS 2025 Datasets & Benchmarks** — deadline ~June 2025; perfect for a framework paper
2. **ICML 2026** — if NeurIPS D&B is missed
3. **MUGen Workshop @ ICML 2025** (July 18, Vancouver) — workshop paper as a stepping stone
4. **EMNLP 2025 System Demonstrations** — if framing as an NLP tool

**Paper angle**: "Erasus: Unified Cross-Modality Machine Unlearning with Coreset-Driven Forgetting and Adversarial Evaluation"
- Contribution 1: First framework supporting unlearning across LLMs, VLMs, diffusion, audio, video
- Contribution 2: Coreset selection for efficient unlearning (24 methods, 90% speedup)
- Contribution 3: Adversarial evaluation suite exposing failures in standard benchmarks
- Contribution 4: Comprehensive empirical comparison of 30+ strategies across modalities

### 9.2 Public GitHub launch

**Checklist before going public**:
- [ ] Clean git history (squash WIP commits)
- [ ] Ensure no hardcoded paths, tokens, or credentials
- [ ] Add GitHub topics: `machine-unlearning`, `coreset-selection`, `llm`, `diffusion-models`, `privacy`, `pytorch`
- [ ] Add social preview image
- [ ] Create GitHub Releases with changelog
- [ ] Set up GitHub Discussions for community Q&A
- [ ] Write a clear CONTRIBUTING.md (already exists — verify it's complete)

### 9.3 Awesome-list submissions

Submit PRs to:
- [awesome-machine-unlearning](https://github.com/tamlhp/awesome-machine-unlearning) — 1.4k stars
- [awesome-llm-unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Awesome-GenAI-Unlearning](https://github.com/franciscoliu/Awesome-GenAI-Unlearning)
- [Awesome-Diffusion-Model-Unlearning](https://github.com/hxxdtd/Awesome-Diffusion-Model-Unlearning)

### 9.4 HuggingFace presence

- Push 20+ model checkpoints (unlearned models for TOFU, MUSE, WMDP × top 5 strategies)
- Create an Erasus organisation on HuggingFace
- Add a HuggingFace Spaces demo (Gradio-based — already in optional deps)
- Create dataset cards for any custom evaluation datasets

### 9.5 Blog posts

Write 3–4 blog posts for launch:
1. **"Introducing Erasus: Unified Machine Unlearning Across Modalities"** — overview, key features, quick start
2. **"Why Machine Unlearning Evaluation is Broken (And What We're Doing About It)"** — adversarial evaluation suite, relearning attacks
3. **"Coreset Selection: 90% Faster Unlearning Without Losing Quality"** — the core innovation
4. **"Unlearning Harry Potter from GPT-2 in 5 Minutes"** — tutorial walkthrough

### 9.6 Community channels

- **Discord server** — preferred over Slack for open-source (better retention, easier invites)
- **Twitter/X**: Announce the launch, tag relevant researchers (Locuslab, Ken Ziyu Liu, Pawelczyk, etc.)
- **Reddit**: Post to r/MachineLearning, r/LanguageTechnology

---

## 10. Prioritised Action Plan

### Phase 1: Foundation (Weeks 1–4)

> Goal: Make Erasus publishable and competitive with OpenUnlearning on LLM benchmarks.

| # | Task | Impact | Effort | Details |
|---|------|--------|--------|---------|
| 1.1 | Add NPO, SimNPO, AltPO strategies | Critical | Medium | 3 new strategies in `strategies/llm_specific/` |
| 1.2 | Add FLAT strategy | Critical | Medium | Forget-data-only method; highly practical |
| 1.3 | Add RMU strategy | Critical | Medium | Representation-based; strong on WMDP |
| 1.4 | Upgrade MIA suite to 6 attacks | Critical | Medium | ZLib, Reference, GradNorm, MinK, MinK++ |
| 1.5 | Add ES and EM metrics | High | Low | Extraction Strength, Exact Memorisation |
| 1.6 | Real TOFU benchmark integration | Critical | High | Real dataset, real models (Phi-1.5 or GPT-2) |
| 1.7 | Fix silent selector fallbacks | Medium | Low | Default to error, opt-in fallback |
| 1.8 | Add coverage threshold to CI | Medium | Low | `--cov-fail-under=80` |

### Phase 2: Differentiation (Weeks 5–8)

> Goal: Build features no competitor has. This is the "why Erasus?" story.

| # | Task | Impact | Effort | Details |
|---|------|--------|--------|---------|
| 2.1 | Adversarial evaluation suite | High | High | Cross-prompt leakage, keyword injection, paraphrase robustness |
| 2.2 | Relearning robustness module | High | High | Benign fine-tuning, quantization, LoRA relearning, prompt attacks |
| 2.3 | lm-evaluation-harness integration | High | Medium | MMLU, GSM8K, TruthfulQA post-unlearning |
| 2.4 | Real WMDP + MUSE benchmarks | High | Medium | Real datasets, real models |
| 2.5 | Continual unlearning pipeline | High | High | Sequential deletion without catastrophic forgetting |
| 2.6 | Inference-time unlearning (DExperts) | Medium | Medium | No-gradient unlearning for production |
| 2.7 | Meta-unlearning for diffusion | Medium | Medium | Prevents concept relearning (ICCV 2025) |

### Phase 3: Launch (Weeks 9–12)

> Goal: Public launch with academic paper submission.

| # | Task | Impact | Effort | Details |
|---|------|--------|--------|---------|
| 3.1 | Write system paper | Critical | High | Target NeurIPS D&B 2025 or ICML 2026 |
| 3.2 | Public GitHub launch | Critical | Low | Clean history, topics, releases, discussions |
| 3.3 | HuggingFace checkpoints (20+) | High | Medium | Unlearned models for TOFU/WMDP/MUSE × top strategies |
| 3.4 | HuggingFace Spaces demo | High | Medium | Gradio-based interactive demo |
| 3.5 | Awesome-list submissions (4 lists) | High | Low | PRs to 4 curated lists |
| 3.6 | Blog post series (3–4 posts) | High | Medium | Introduction, evaluation, coreset, tutorial |
| 3.7 | Discord community setup | Medium | Low | Community channel for users and contributors |
| 3.8 | API stabilisation + v0.2.0 | Medium | Medium | Audit, document, bump version |

### Phase 4: Growth (Weeks 13+)

> Goal: Sustain momentum and build community.

| # | Task | Impact | Effort | Details |
|---|------|--------|--------|---------|
| 4.1 | UNDIAL, WGA, delta-UNLEARNING strategies | Medium | Medium | Complete the method inventory |
| 4.2 | Hydra config system | Medium | Medium | Reproducible experiment management |
| 4.3 | Continual unlearning benchmarks | Medium | High | New benchmark for sequential deletion |
| 4.4 | GNN unlearning support | Medium | High | Graph neural network modality |
| 4.5 | REST API service | Medium | Medium | FastAPI endpoints for unlearning-as-a-service |
| 4.6 | RL unlearning support | Low | High | Remove trajectories from RL policies |
| 4.7 | Competition hosting | High | High | Community benchmark challenge |

---

## 11. Positioning & Narrative

### The wrong narrative

> "Erasus has 27 strategies, 24 selectors, and supports 5 modalities."

This is feature-counting. It doesn't answer "why should I use this?" — especially when 2025 research shows that many strategies don't actually work.

### The right narrative

> "Other frameworks claim to unlearn. Erasus helps you verify it."

The 2025 research reveals a fundamental tension: approximate unlearning may not work. ICLR 2025 showed it fails against data poisoning. CVPR 2025 showed it's unstable for diffusion models. CMU showed benchmarks give false confidence.

Erasus should position itself as the framework that takes this seriously:

1. **Verification-first**: Certification module, adversarial evaluation, relearning robustness testing
2. **Coreset intelligence**: Not just "apply gradient ascent" but "identify the minimal set that matters, then surgically remove it"
3. **Cross-modality**: The only framework where you can unlearn from CLIP, GPT-2, Stable Diffusion, and Whisper with the same API
4. **Honest evaluation**: Report adversarial metrics alongside standard ones. Show where methods fail, not just where they succeed.

### Tagline options

- "Efficient Representative And Surgical Unlearning Selection" (current — accurate but not memorable)
- "Verify, then forget." (emphasises evaluation)
- "The machine unlearning framework that doesn't lie to you." (provocative but differentiated)
- "Unlearn across any modality. Verify it actually worked." (complete positioning)

---

## 12. Key References

### Foundational critiques (must-read)

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| Pawelczyk et al., "Machine Unlearning Fails to Remove Data Poisoning Attacks" | ICLR 2025 | Existing methods fail across all poison types |
| Shumailov et al., "Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs" | ICLR 2025 | Benign relearning reverses approximate unlearning |
| George et al., "The Illusion of Unlearning" | CVPR 2025 | All diffusion erasure methods are unstable |
| Thaker et al., "LLM Unlearning Benchmarks are Weak Measures of Progress" | CMU ML Blog, April 2025 | TOFU/WMDP benchmarks are fundamentally flawed |

### SOTA methods to implement

| Paper | Venue | Method |
|-------|-------|--------|
| Zhang et al., "Negative Preference Optimization" | 2024 | NPO |
| Fan et al., "SimNPO: Simplicity Prevails" | NeurIPS 2025 | SimNPO |
| Choi et al., "Alternate Preference Optimization" | 2024 | AltPO |
| Li et al., "LLM Unlearning via Loss Adjustment" | ICLR 2025 | FLAT |
| Li et al., "Representation Misdirection for Unlearning" | 2024 | RMU |
| Dong et al., "UNDIAL" | 2024 | UNDIAL |
| Liu et al., "DExperts" | 2024 | Inference-time unlearning |
| Gao et al., "Meta-Unlearning on Diffusion Models" | ICCV 2025 | Anti-relearning for diffusion |

### Competing frameworks

| Framework | Repo | Venue |
|-----------|------|-------|
| OpenUnlearning | [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning) | NeurIPS D&B 2025 |
| EasyEdit | [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) | ACL 2024 |
| OpenUnlearn | [MartinPawelczyk/OpenUnlearn](https://github.com/MartinPawelczyk/OpenUnlearn) | ICLR 2025 |
| ZJUNLP Unlearn | [zjunlp/unlearn](https://github.com/zjunlp/unlearn) | ACL 2025 |
| ESD/erasing | [rohitgandikota/erasing](https://github.com/rohitgandikota/erasing) | ICCV 2023 |
| FLAT | [UCSC-REAL/FLAT](https://github.com/UCSC-REAL/FLAT) | ICLR 2025 |

### Curated paper lists

- [awesome-machine-unlearning](https://github.com/tamlhp/awesome-machine-unlearning)
- [awesome-llm-unlearning](https://github.com/chrisliu298/awesome-llm-unlearning)
- [Awesome-GenAI-Unlearning](https://github.com/franciscoliu/Awesome-GenAI-Unlearning)
- [Awesome-Diffusion-Model-Unlearning](https://github.com/hxxdtd/Awesome-Diffusion-Model-Unlearning)

### Surveys

- "A Survey on Machine Unlearning in LLMs" (2025) — [arXiv:2503.01854](https://arxiv.org/abs/2503.01854)
- "Machine Unlearning in 2024" — Ken Ziyu Liu, Stanford — [blog](https://ai.stanford.edu/~kzliu/blog/unlearning/)
- ACM Computing Surveys, "Machine Unlearning" (2025) — [doi:10.1145/3749987](https://doi.org/10.1145/3749987)

### Competitions and workshops

- NeurIPS 2023 Machine Unlearning Challenge — [site](https://unlearning-challenge.github.io)
- SemEval 2025 Task 4: Unlearning Sensitive Content from LLMs — [site](https://llmunlearningsemeval2025.github.io)
- MUGen Workshop @ ICML 2025 (July 18, Vancouver) — [site](https://mugenworkshop.github.io)

---

*This document should be updated as the landscape evolves. The machine unlearning field is moving fast — what's SOTA today may be obsolete in 6 months.*
