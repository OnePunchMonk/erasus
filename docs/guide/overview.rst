Architecture Overview
=====================

Erasus is designed as a modular, extensible framework for machine
unlearning research. This guide explains the core architecture.

Design Philosophy
-----------------

Erasus follows three key design principles:

1. **Modular Architecture** — Every component (strategy, selector,
   metric, model wrapper) is a pluggable module registered via a
   global registry. Swap components without changing pipeline code.

2. **Multi-Modal Support** — Unified API across vision, language,
   diffusion, audio, and video modalities. Each modality has a
   specialised unlearner that inherits from ``BaseUnlearner``.

3. **Research-First** — Comprehensive metrics, visualisation, and
   experiment tracking built-in. Reproduce papers, run benchmarks,
   and compare strategies with minimal code.

Core Components
---------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────┐
   │                    ERASUS FRAMEWORK                       │
   ├──────────────────────────────────────────────────────────┤
   │                                                          │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
   │  │ Strategy │  │ Selector │  │ Metrics  │               │
   │  │ Registry │  │ Registry │  │ Registry │               │
   │  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
   │       │             │             │                      │
   │  ┌────▼─────────────▼─────────────▼─────┐               │
   │  │          BaseUnlearner                │               │
   │  │   ┌──────────────────────────────┐   │               │
   │  │   │ fit(forget, retain, epochs)  │   │               │
   │  │   │ evaluate(forget, retain)     │   │               │
   │  │   └──────────────────────────────┘   │               │
   │  └────┬──────────┬──────────┬───────────┘               │
   │       │          │          │                            │
   │  ┌────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌─────────┐          │
   │  │  VLM   │ │  LLM   │ │Diffusn │ │ Audio/  │          │
   │  │Unlearn │ │Unlearn │ │Unlearn │ │ Video   │          │
   │  └────────┘ └────────┘ └────────┘ └─────────┘          │
   │                                                          │
   └──────────────────────────────────────────────────────────┘

Registry System
---------------

All components use a decorator-based registration pattern:

.. code-block:: python

   from erasus.core.registry import strategy_registry

   @strategy_registry.register("my_custom_strategy")
   class MyStrategy(BaseStrategy):
       def unlearn(self, model, forget_loader, retain_loader, **kwargs):
           # Your custom unlearning logic
           pass

Components can then be looked up by name throughout the framework.

Data Flow
---------

1. **Input**: Model + forget set + retain set
2. **Selection**: Selector chooses a coreset from the forget set
3. **Unlearning**: Strategy modifies the model to forget selected data
4. **Evaluation**: Metrics measure forgetting quality, utility, privacy
5. **Output**: ``UnlearningResult`` with loss history, timing, metrics

Module Organisation
-------------------

.. code-block:: text

   erasus/
   ├── core/           # Base classes, registry, config, types
   ├── strategies/     # 28 unlearning strategies
   ├── selectors/      # 22 coreset selectors
   ├── metrics/        # 26+ evaluation metrics
   ├── models/         # 18+ model wrappers (VLM, LLM, diffusion, audio, video)
   ├── unlearners/     # 8 modality-specific unlearners
   ├── data/           # Dataset loaders, augmentation, synthetic data
   ├── visualization/  # 13 plotting and analysis tools
   ├── privacy/        # DP, gradient clipping, secure aggregation
   ├── certification/  # Certified removal, verification, bounds
   ├── experiments/    # Tracking, hyperparameter search, ablation
   ├── losses/         # 8 loss functions
   ├── utils/          # Helpers, profiling, reproducibility
   └── cli/            # Command-line interface
