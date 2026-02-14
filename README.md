<p align="center">
  <h1 align="center">👻 Erasus</h1>
  <p align="center">
    <strong>Efficient Representative And Surgical Unlearning Selection</strong><br>
    Universal Machine Unlearning via Coreset Selection
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="#-installation"><img src="https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="#-test-status"><img src="https://img.shields.io/badge/tests-340%20passed-brightgreen.svg" alt="Tests"></a>
    <a href="#-supported-models"><img src="https://img.shields.io/badge/models-18%20architectures-purple.svg" alt="Models"></a>
    <a href="#-strategies--selectors"><img src="https://img.shields.io/badge/strategies-30%20methods-orange.svg" alt="Strategies"></a>
  </p>
</p>

---

**Erasus** is a research-grade Python framework for **Machine Unlearning** across all major foundation model types. It surgically removes specific data, concepts, or behaviors from trained models — without the computational cost of full retraining.

It supports **Vision-Language Models**, **Large Language Models**, **Diffusion Models**, **Audio Models**, and **Video Models** through a unified API backed by 27 unlearning strategies, 19 coreset selectors, 7 loss functions, and a comprehensive evaluation suite with 15+ metrics.

---

## 🧠 How It Works

Erasus operates in a three-stage pipeline:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  1. CORESET SELECTION │────▶│  2. TARGETED          │────▶│  3. EVALUATION &     │
│                      │     │     UNLEARNING         │     │     CERTIFICATION    │
│  Pick the minimal    │     │                        │     │                      │
│  set of samples that │     │  Apply gradient ascent,│     │  MIA, accuracy,      │
│  define forgetting   │     │  Fisher, SCRUB, LoRA,  │     │  perplexity, FID,    │
│  "support vectors"   │     │  or 16+ other methods  │     │  certified removal   │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

**Key Innovation:** Geometry-aware coreset selection identifies the *"support vectors of forgetting"* — proving that unlearning k% of the most influential samples approximates unlearning 100% with bounded utility loss.

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Coreset-Driven Forgetting** | 24 coreset selectors (influence functions, CRAIG, herding, k-center, EL2N, TracIn, Data Shapley, Active Learning) reduce compute by up to 90% |
| 🧩 **Ensemble Unlearning** | Combine strategies sequentially or via weight averaging for robust forgetting |
| 📷📝 **Multimodal Decoupling** | Unlearn image-text associations without breaking visual or textual generalization |
| 🌐 **Federated Unlearning** | Decentralized unlearning across clients with FedAvg aggregation and client-side forgetting |
| 🛡️ **Utility Preservation** | Retain-Anchor loss + Fisher regularization constrain model drift on safe data |
| 🔐 **Certified Removal** | Formal (ε, δ)-removal verification with PAC-style guarantees |
| 📊 **Integrated Evaluation** | MIA, confidence, feature distance, perplexity, FID, activation analysis, backdoor detection, 25+ metrics |
| 📈 **Visualization Suite** | Loss landscapes, embedding plots, gradient flow, interactive Plotly dashboards, HTML reports |
| 🔌 **Model Agnostic** | Works with any PyTorch model + HuggingFace Transformers (BERT, LLaMA, T5, CLIP, DALL-E) |
| 🖥️ **CLI + Python API** | `erasus unlearn`, `erasus benchmark`, `erasus visualize`, or full Python API |
| 🧪 **Experiment Tracking** | Built-in W&B, MLflow, local JSON tracking + HPO with Optuna |
| 📐 **Theoretical Bounds** | PAC-learning utility bounds, influence bounds, certified unlearning radius |

---

## 🏗️ Supported Models

| Modality | Models | Unlearner |
|----------|--------|-----------|
| **Vision-Language** | CLIP, LLaVA, BLIP-2, Flamingo, VisionTransformer | `VLMUnlearner` |
| **Language** | LLaMA, Mistral, GPT-2/J, BERT, T5 | `LLMUnlearner` |
| **Diffusion** | Stable Diffusion 1.x/2.x/XL, DALL-E, Imagen | `DiffusionUnlearner` |
| **Audio** | Whisper, CLAP, Wav2Vec | `AudioUnlearner` |
| **Video** | VideoMAE, VideoCLIP | `VideoUnlearner` |
| **Federated** | Any Architecture | `FederatedUnlearner` |
| **Any** | Auto-detect | `MultimodalUnlearner` |

---

## 📦 Installation

```bash
# From PyPI (once published)
pip install erasus
pip install erasus[full]   # with diffusers, datasets, wandb, etc.
pip install erasus[hub]    # Hugging Face Hub push/pull

# From source (development)
git clone https://github.com/OnePunchMonk/erasus.git
cd erasus
pip install -e .

# With all optional dependencies
pip install -e ".[full]"

# Hugging Face Hub (push/pull unlearned models)
pip install -e ".[hub]"

# Development
pip install -e ".[dev]"
```

### Quick Setup Script
```bash
bash scripts/setup_env.sh          # CPU
bash scripts/setup_env.sh --gpu    # CUDA 12.1
```

### Docker
```bash
docker compose -f docker/docker-compose.yml up test       # Run tests
docker compose -f docker/docker-compose.yml run dev        # Dev shell
docker compose -f docker/docker-compose.yml up benchmark   # GPU benchmarks
```

---

## 🚀 Quick Start

### Python API

```python
from erasus.unlearners import ErasusUnlearner

# 1. Load your model
model = ...  # Any PyTorch model

# 2. Create unlearner
unlearner = ErasusUnlearner(
    model=model,
    strategy="gradient_ascent",    # 27 strategies available
    selector="influence",          # 19 selectors available
    device="cuda",
)

# 3. Unlearn
result = unlearner.fit(
    forget_data=forget_loader,     # Data to remove
    retain_data=retain_loader,     # Data to preserve
    prune_ratio=0.1,               # Use top 10% coreset
    epochs=5,
)

# 4. Evaluate
metrics = unlearner.evaluate(
    forget_data=forget_loader,
    retain_data=retain_loader,
)
print(f"MIA AUC: {metrics['mia_auc']:.4f}")  # Should → 0.5
```

### Modality-Specific Unlearners

```python
from erasus.unlearners import VLMUnlearner, LLMUnlearner, DiffusionUnlearner

# CLIP: Remove NSFW concepts
vlm = VLMUnlearner(model=clip_model, strategy="modality_decoupling")
vlm.fit(forget_data=nsfw_loader, retain_data=safe_loader)

# LLaMA: Remove hazardous knowledge
llm = LLMUnlearner(model=llama_model, strategy="gradient_ascent")
llm.fit(forget_data=harmful_loader, retain_data=benign_loader)

# Stable Diffusion: Remove artist styles
diff = DiffusionUnlearner(model=sd_model, strategy="concept_erasure")
diff.fit(forget_data=artist_loader, retain_data=general_loader)
```

### Auto-Detect Model Type

```python
from erasus.unlearners import MultimodalUnlearner

# Automatically picks the right unlearner
unlearner = MultimodalUnlearner.from_model(your_model)
```

### CLI

```bash
# Run unlearning
erasus unlearn --config configs/default.yaml

# Evaluate results
erasus evaluate --config configs/default.yaml --checkpoint model.pt

# Run benchmarks
erasus benchmark --strategies gradient_ascent,scrub --selectors random,influence

# Generate visualizations
erasus visualize --type embeddings --method tsne --output embeddings.png
erasus visualize --type comparison --output comparison.png
erasus visualize --type report --output report.html
```

---

## 🔧 Strategies & Selectors

### Unlearning Strategies (30)

| Category | Strategies |
|----------|------------|
| **Gradient Methods** | Gradient Ascent, SCRUB (CVPR 2024), Fisher Forgetting, Negative Gradient, Modality Decoupling, **Saliency Unlearning** |
| **Parameter Methods** | LoRA Unlearning, Sparse-Aware, Mask-Based, Neuron Pruning, **Layer Freezing** |
| **Data Methods** | Amnesiac ML, SISA, Certified Removal, **Knowledge Distillation** |
| **LLM-Specific** | SSD (NeurIPS 2024), Token Masking, Embedding Alignment, Causal Tracing, **Attention Surgery** |
| **Diffusion-Specific** | Concept Erasure (ICCV 2023), Noise Injection, U-Net Surgery, **Timestep Masking**, **Safe Latents** |
| **VLM-Specific** | Contrastive Unlearning, Cross-Modal Decoupling, **Attention Unlearning**, Vision-Text Split |
| **Ensemble** | Sequential / Averaged multi-strategy combination |

### Coreset Selectors (24)

| Category | Selectors |
|----------|-----------|
| **Gradient-Based** | Influence Functions, TracIn, Gradient Norm, GradMatch/CRAIG, EL2N, Representer, Forgetting Score |
| **Geometry-Based** | k-Center, Herding, GLISTER, Submodular, k-Means++, Farthest First |
| **Learning-Based** | Forgetting Events, Data Shapley, Valuation Network, Active Learning, Loss Accumulation |
| **Ensemble** | Voting Selector, Auto-Selector, Weighted Fusion |

---

## 📊 Evaluation & Metrics

```python
from erasus.metrics import MetricSuite

suite = MetricSuite(["accuracy", "mia", "perplexity"])
results = suite.run(model, forget_loader, retain_loader)
```

| Category | Metrics |
|----------|---------|
| **Forgetting** | MIA (+ LiRA, LOSS variants), Confidence, Feature Distance, **Activation Analysis**, **Backdoor ASR**, Extraction Attack |
| **Utility** | Accuracy, Perplexity, Retrieval (R@1/5/10), FID, BLEU, ROUGE, CLIP Score, Inception Score |
| **Efficiency** | Time Complexity, Memory Usage, **Speedup Ratio**, **FLOPs Estimation** |
| **Privacy** | Differential Privacy (ε, δ), Privacy Audit |

---

## 📈 Visualization

```python
from erasus.visualization import (
    EmbeddingVisualizer,
    LossLandscapeVisualizer,
    GradientVisualizer,
    ReportGenerator,
)
from erasus.visualization.attention import AttentionVisualizer
from erasus.visualization.comparisons import ComparisonVisualizer

# t-SNE / PCA embeddings
viz = EmbeddingVisualizer(model)
viz.plot(data_loader, method="tsne")

# Loss landscape
landscape = LossLandscapeVisualizer(model)
landscape.plot_2d_contour(data_loader)

# Attention heatmaps (before vs. after)
attn_viz = AttentionVisualizer(model_after)
attn_viz.plot_attention_comparison(inputs, model_before)

# Before/after comparisons
comp = ComparisonVisualizer()
comp.plot_prediction_shift(model_before, model_after, forget_loader)
comp.plot_metric_comparison(metrics_before, metrics_after)

# HTML report
report = ReportGenerator("Unlearning Report")
report.add_metrics(metrics)
report.save("report.html")
```

---

## 🔐 Certification & Privacy

```python
from erasus.certification import CertifiedRemovalVerifier, UnlearningVerifier

# Formal (ε, δ)-removal verification
verifier = CertifiedRemovalVerifier(epsilon=1.0, delta=1e-5)
result = verifier.verify(unlearned_model, retrained_model, n_total=10000, n_forget=500)
print(f"Certified: {result['certified']}")

# Statistical verification
stat_verifier = UnlearningVerifier(significance=0.05)
tests = stat_verifier.verify_all(model, forget_loader, retain_loader)
```

### Theoretical Bounds

```python
from erasus.certification.bounds import TheoreticalBounds

# PAC-learning utility bound
bounds = TheoreticalBounds.pac_utility_bound(
    n_total=50000, n_forget=500, n_retain=49500, delta=0.05, model=model,
)
print(f"Utility drop bound: {bounds['pac_utility_drop_bound']:.4f}")

# Certified unlearning radius
radius = TheoreticalBounds.unlearning_radius(
    epsilon=1.0, delta=1e-5, n_forget=500,
)
print(f"Certified radius: {radius['certified_radius']:.4f}")
```

---

## 📉 Loss Functions

| Loss | Description |
|------|-------------|
| **Retain Anchor** | Cross-entropy on retain data to preserve utility |
| **Contrastive** | CLIP-style contrastive loss for VLM alignment |
| **KL Divergence** | Distribution matching between models |
| **MMD** | Maximum Mean Discrepancy for distribution comparison |
| **Fisher Regularization** | Fisher information-weighted parameter penalty |
| **Adversarial** | GAN-style loss for indistinguishable forget/retain outputs |
| **Triplet** | Push forget embeddings away from retain-set anchors |
| **L2 Regularization** | Simple weight-drift penalty |

---

## 🧪 Experiment Tracking

```python
from erasus.experiments import ExperimentTracker, HyperparameterSearch, AblationStudy

# Supports: "local", "wandb", "mlflow"
with ExperimentTracker("clip_unlearning", backend="wandb") as tracker:
    tracker.log_config({"strategy": "gradient_ascent", "lr": 1e-4})
    result = unlearner.fit(...)
    tracker.log_metrics({"mia_auc": 0.52, "accuracy": 0.94})
    tracker.log_model(model)

# Hyperparameter search (Optuna or random fallback)
search = HyperparameterSearch(
    objective_fn=my_objective,
    param_space={"lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    n_trials=50,
)
best = search.run()

# Ablation studies
ablation = AblationStudy(base_config={...}, run_fn=run_trial)
ablation.run_full_ablation({"lr": [1e-3, 1e-4, 1e-5], "strategy": ["ga", "scrub"]})
print(ablation.summary())
```

---

## 📁 Project Structure

```
erasus/
├── core/           # Base classes, registry, config, types
├── unlearners/     # High-level API (7 modality-specific unlearners)
├── strategies/     # 27 unlearning algorithms (gradient, parameter, data, LLM, diffusion, VLM, ensemble)
├── selectors/      # 19 coreset selection methods (gradient, geometry, learning, ensemble)
├── metrics/        # 15+ evaluation metrics (forgetting, utility, efficiency, privacy)
├── losses/         # 8 loss functions (retain-anchor, Fisher, adversarial, triplet, KL, MMD, L2)
├── visualization/  # Embeddings, loss surfaces, gradients, attention heatmaps, comparisons, reports
├── data/           # Dataset loaders (TOFU, WMDP, COCO, I2P, CC), preprocessing, partitioning
├── models/         # 10 model wrappers (VLM, LLM, diffusion, audio, video)
├── privacy/        # DP mechanisms, privacy accountant, certificates
├── certification/  # Certified removal, statistical verification, theoretical bounds
├── experiments/    # W&B / MLflow / local tracking, HPO, ablation studies
├── cli/            # Command-line interface (unlearn, evaluate, benchmark, visualize)
└── utils/          # Checkpointing, distributed, helpers, logging, callbacks, early stopping
```

---

## 🏆 Benchmarks

Run standardized benchmarks:

```bash
# TOFU Benchmark (LLM unlearning)
python benchmarks/tofu/run.py --strategies gradient_ascent,scrub --epochs 5

# WMDP Benchmark (hazardous knowledge)
python benchmarks/wmdp/run.py --subsets bio,cyber

# Full suite
bash scripts/run_benchmarks.sh
```

---

## 🧑‍💻 Examples

| Example | Description |
|---------|-------------|
| [CLIP Coreset Comparison](examples/vision_language/clip_coreset_comparison.py) | Compare random vs. gradient_norm selectors |
| [LLaVA Unlearning](examples/vision_language/llava_unlearning.py) | VLM unlearning with gradient ascent |
| [LLaMA Concept Removal](examples/language_models/llama_concept_removal.py) | Remove concepts from LLaMA |
| [GPT-2 Strategy Comparison](examples/language_models/gpt2_unlearning.py) | Compare gradient_ascent vs. negative_gradient |
| [LoRA Efficient Unlearning](examples/language_models/lora_efficient_unlearning.py) | Parameter-efficient unlearning |
| [SD NSFW Removal](examples/diffusion_models/stable_diffusion_nsfw.py) | Remove NSFW concepts |
| [SD Artist Removal](examples/diffusion_models/stable_diffusion_artist.py) | Remove artist styles |
| [TOFU Benchmark](examples/benchmarks/run_tofu_benchmark.py) | End-to-end benchmark |

---

## ✅ Test Status

```
340 tests passed ✅  |  0 failed  |  54s
```

```bash
python -m pytest tests/ -v --tb=short
```

| Test Suite | Status |
|-----------|:------:|
| Integration (pipelines) | ✅ |
| End-to-end | ✅ |
| Unit (selectors) | ✅ |
| Unit (strategies) | ✅ |
| Unit (metrics) | ✅ |
| Core / imports / components | ✅ |

---

## 📚 Research References

Erasus integrates and builds upon these key works:

| Method | Paper | Venue |
|--------|-------|-------|
| SCRUB | Kurmanji et al. | CVPR 2024 |
| Selective Synaptic Dampening | Foster et al. | NeurIPS 2024 |
| Concept Erasure (ESD) | Gandikota et al. | ICCV 2023 |
| Gradient Ascent | Golatkar et al. | NeurIPS 2020 |
| Fisher Forgetting | Golatkar et al. | NeurIPS 2020 |
| CRAIG | Mirzasoleiman et al. | NeurIPS 2020 |
| GLISTER | Killamsetty et al. | ICLR 2021 |
| Influence Functions | Koh & Liang | ICML 2017 |
| TracIn | Pruthi et al. | NeurIPS 2020 |
| Data Shapley | Ghorbani & Zou | ICML 2019 |
| Forgetting Events | Toneva et al. | ICLR 2019 |
| EL2N | Paul et al. | ICML 2021 |
| Amnesiac ML | Graves et al. | S&P 2021 |

---

## 🗺️ Roadmap

- [x] Core framework (base classes, registry, config)
- [x] 10 model architectures
- [x] 27 unlearning strategies (gradient, parameter, data, LLM, diffusion, VLM, ensemble)
- [x] 19 coreset selectors
- [x] 15+ evaluation metrics (forgetting, utility, efficiency, privacy)
- [x] 8 loss functions (Fisher, adversarial, triplet, L2, retain-anchor, KL, MMD, contrastive)
- [x] Visualization suite (embeddings, landscapes, gradients, attention, comparisons, reports)
- [x] CLI (`erasus unlearn`, `erasus evaluate`, `erasus benchmark`, `erasus visualize`)
- [x] Certification & privacy modules + theoretical bounds (PAC, influence, certified radius)
- [x] Experiment tracking (W&B, MLflow, local) + HPO + ablation studies
- [x] Benchmark runners (TOFU, WMDP)
- [x] Callbacks & early stopping
- [x] 340+ passing tests
- [x] Additional model architectures (Flamingo, T5, DALL-E, Wav2Vec)
- [ ] HuggingFace Hub integration
- [ ] Interactive Gradio/Streamlit dashboard
- [ ] Tutorial notebooks
- [ ] PyPI release

---

## 🤝 Contributing

Contributions are welcome! Whether it's new unlearning strategies, coreset selectors, model support, or documentation.

```bash
# Setup development environment
git clone https://github.com/OnePunchMonk/erasus.git
cd erasus
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📖 Citation

```bibtex
@software{erasus2026,
  title={Erasus: Universal Machine Unlearning via Coreset Selection},
  author={Aggarwal, Avaya},
  year={2026},
  url={https://github.com/OnePunchMonk/erasus}
}
```

---

<p align="center">
  <b>Built with ❤️ for the machine unlearning research community</b>
</p>
