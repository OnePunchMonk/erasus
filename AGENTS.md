# Erasus — Codex Guide

## What is this project?

Erasus (**Efficient Representative And Surgical Unlearning Selection**) is a Python framework for machine unlearning across all major foundation model types (LLMs, VLMs, Diffusion, Audio, Video). It surgically removes data, concepts, or behaviors from trained models without full retraining.

## Quick reference

- **Language**: Python 3.9+
- **Framework**: PyTorch
- **Package manager**: pip (pyproject.toml)
- **Entry point**: `erasus` CLI (`erasus/cli/main.py`)
- **License**: MIT

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run all tests (exclude optional-dep-only tests)
python3 -m pytest tests/ -v --tb=short --ignore=tests/test_components.py --ignore=tests/unit/test_sprint_b.py --ignore=tests/unit/test_sprint_f.py

# Run a specific test file
python3 -m pytest tests/unit/test_verification.py -v

# Lint
ruff check erasus/ tests/

# Format check
ruff format --check erasus/ tests/
```

## Architecture

### Core pattern: Registry + Strategy + Selector

Everything is built on a plugin registry:

```
erasus/core/registry.py     → strategy_registry, selector_registry, model_registry, metric_registry
erasus/core/base_strategy.py → BaseStrategy (abstract)
erasus/core/base_selector.py → BaseSelector (abstract)
erasus/core/base_metric.py   → BaseMetric (abstract)
erasus/core/base_unlearner.py → BaseUnlearner (abstract)
```

New strategies/selectors/metrics are registered via `@registry.register("name")` decorator.

### Key directories

```
erasus/
├── core/           # Base classes, registry, config, types
├── unlearners/     # High-level orchestrators (Generic, VLM, LLM, Diffusion, Audio, Video, Federated)
├── strategies/     # 27 unlearning algorithms (gradient/, parameter/, data/, llm_specific/, diffusion_specific/, vlm_specific/)
├── selectors/      # 24 coreset selection methods (gradient_based/, geometry_based/, learning_based/, ensemble/)
├── losses/         # 8 loss functions
├── metrics/        # 25+ evaluation metrics (forgetting/, utility/, efficiency/, privacy/)
├── evaluation/     # Adversarial + relearning robustness tests, unified verification suite
├── models/         # Model wrappers (vlm/, llm/, diffusion/, audio/, video/)
├── data/           # Dataset loaders, preprocessing, partitioning
├── privacy/        # DP mechanisms, certificates
├── certification/  # Formal (epsilon, delta)-removal verification
├── visualization/  # 16 visualization modules
├── experiments/    # Experiment tracking, HPO, ablation
├── cli/            # CLI commands (unlearn, evaluate, benchmark, visualize)
└── utils/          # Helpers, checkpointing, distributed, profiling
```

### Data flow

```
forget_data + retain_data → Selector (picks coreset) → Strategy.unlearn() → Modified Model → Metrics
```

### Typical usage

```python
from erasus import ErasusUnlearner
unlearner = ErasusUnlearner(model=model, strategy="gradient_ascent", selector="influence")
result = unlearner.fit(forget_data=forget_loader, retain_data=retain_loader, epochs=5)
metrics = unlearner.evaluate(forget_data=forget_loader, retain_data=retain_loader)
```

## Code conventions

- **Style**: PEP 8, enforced by ruff (line length 100)
- **Type hints**: Required on all public functions
- **Docstrings**: NumPy-style docstrings on public classes and methods
- **Imports**: Use `from __future__ import annotations` at the top of every module
- **Base classes**: All strategies extend `BaseStrategy`, all selectors extend `BaseSelector`, all metrics extend `BaseMetric`
- **Registration**: Use `@registry.register("name")` decorator or register in `__init__.py`
- **Error handling**: Raise errors for invalid config. Use `warnings.warn()` only with explicit opt-in fallbacks.
- **No silent fallbacks**: Selectors should raise errors if required data is missing, not silently fall back to random selection (this is a known issue being fixed).

## Testing

- Tests live in `tests/` with `unit/` and `integration/` subdirs
- Fixtures in `tests/conftest.py` — `TinyClassifier`, `TinyCNN`, `_make_loader()`
- All tests use synthetic tiny models (input_dim=16, 4 classes) for speed
- 3 test files require optional deps (`matplotlib`, etc.) and may fail in minimal environments: `test_components.py`, `test_sprint_b.py`, `test_sprint_f.py`
- Pre-existing failure: `test_imports.py::test_visualization_imports` fails without matplotlib

## Key modules for verification (new)

The `erasus/evaluation/` package implements adversarial unlearning verification:

- **`mia_suite.py`**: 6-attack MIA battery (LOSS, ZLib, Reference, GradNorm, MinK, MinK++)
- **`memorization.py`**: Extraction Strength, Exact Memorization, Verbatim Memorization metrics
- **`adversarial.py`**: Cross-prompt leakage, keyword injection, paraphrase robustness tests
- **`relearning.py`**: Benign fine-tuning, quantization, LoRA relearning, prompt extraction attacks
- **`verification_suite.py`**: Unified runner that combines all of the above into a single PASS/PARTIAL/FAIL verdict

## Benchmarks

- `benchmarks/tofu/` — TOFU (LLM unlearning, fictitious author QA)
- `benchmarks/muse/` — MUSE (6-way evaluation)
- `benchmarks/wmdp/` — WMDP (hazardous knowledge removal)
- Currently use synthetic data + `BenchmarkModel` for CI. Real model benchmarks planned.

## Important context

- This is a research framework, not a production service
- The project is pre-v1 (currently 0.1.1, alpha status)
- Main competitor is OpenUnlearning (CMU, LLM-only). Erasus differentiates via cross-modality support, coreset selection, and verification.
- See `ROADMAP.md` for the detailed improvement plan
- See `project_ideas.md` for future extension ideas
