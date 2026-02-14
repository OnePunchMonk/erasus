# Erasus — Project Ideas

Ideas to extend and improve the Erasus machine unlearning package.

---

## 1. Algorithms & Strategies

- **Certified data deletion with exact Newton step** — Implement the full Newton-step certified removal (e.g. Guo et al.) for convex models and approximate for deep nets.
- **Unlearning for graph neural networks** — Support GNNs (node/edge deletion) with message-passing unlearning strategies.
- **Continual unlearning** — Sequential forget requests without full retrain; incremental coreset update and strategy scheduling.
- **Differential privacy–guaranteed unlearning** — Tight (ε, δ) accounting when unlearning is composed with DP training.
- **Sparse unlearning** — Only update a small subset of parameters (e.g. mask or low-rank) to reduce cost and improve stability.
- **Bayesian unlearning** — Prior/posterior updates to “remove” data from Bayesian models or Laplace approximations.

---

## 2. Coresets & Selectors

- **Learned meta-selector** — Train a small model to predict which selector (or prune ratio) works best for a given dataset/model/strategy.
- **Cross-modal influence** — For VLMs, compute influence that accounts for both image and text branches.
- **Uncertainty-based selection** — Use epistemic or aleatoric uncertainty to pick forget samples that are most “informative” to unlearn.
- **Coreset for diffusion** — Define and implement forget-set coresets for diffusion models (e.g. by prompt or latent importance).

---

## 3. Evaluation & Benchmarks

- **Standardised benchmark suite** — One script/CLI that runs TOFU, MUSE, WMDP, and custom benchmarks and outputs a single report (JSON + Markdown).
- **LiRA and advanced MIA** — Add Likelihood Ratio Attack and other strong MIA baselines to the metrics module.
- **Utility–forgetting Pareto front** — Tools to sweep hyperparameters and plot utility vs forgetting trade-offs.
- **Temporal unlearning** — Benchmarks for “forget by time window” (e.g. forget last 30 days of data).

---

## 4. Integrations & Deployment

- **Hugging Face Trainer integration** — Callback or custom Trainer that performs unlearning steps after training.
- **ONNX / TorchScript export** — Export unlearned models to ONNX or TorchScript for deployment.
- **REST API** — Small FastAPI service: “unlearn these indices” and “evaluate” endpoints.
- **Weights & Biases / MLflow** — Deeper integration: log unlearning runs, curves, and model versions.

---

## 5. Documentation & Outreach

- **Video tutorials** — Short walkthroughs: installation, first unlearning run, running a paper reproduction.
- **Comparison table** — Maintain a table of “Erasus vs other libraries” (e.g. vs SISA codebases, commercial tools).
- **Blog posts** — “How to unlearn in 5 minutes” and “Choosing a strategy and selector”.
- **Academic survey** — Keep a `papers/` summary of key unlearning papers and how they map to Erasus strategies.

---

## 6. Research Directions

- **Theoretical bounds** — Tighten PAC-style or regret bounds for specific strategy/selector combinations.
- **Unlearning for RL** — Remove trajectories or reward sources from trained RL policies.
- **Federated unlearning** — Extend federated unlearner with secure aggregation and client-level forget requests.
- **Unlearning in retrieval systems** — Remove documents or queries from embedding-based retrievers (e.g. dual-encoder unlearning).

---

## 7. Engineering & DX

- **Config-driven runs** — Single YAML that defines model, data paths, strategy, selector, and metrics; run via CLI.
- **Resumable unlearning** — Checkpoint mid-unlearning and resume (e.g. for long-running jobs).
- **Profiling** — Time and memory profiler for “where does unlearning spend time?” (forward, backward, selector, etc.).
- **Pre-commit hooks** — Lint and run a minimal test suite on commit.

---

## 8. Datasets & Data

- **Synthetic forget sets** — Helpers to generate synthetic “sensitive” data for testing (e.g. by class, by attribute).
- **LAION/OBELICS subset loaders** — Stream or sample subsets for large-scale unlearning experiments.
- **Data versioning** — Integration with DVC or similar to track “forget set v1” vs “v2” in experiments.

---

*Contributions welcome. Open an issue or PR on [GitHub](https://github.com/OnePunchMonk/erasus) with the label `project-idea`.*
