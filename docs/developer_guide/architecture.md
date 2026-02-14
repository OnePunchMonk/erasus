# Internal Architecture

## Overview

Erasus follows a plugin-based architecture where every major component
(strategy, selector, metric, model wrapper) is registered in a global
registry and resolved by name at runtime.

## Core Abstractions

### BaseUnlearner
The central orchestrator. It:
1. Resolves strategy and selector by name from registries
2. Calls `selector.select()` to build a coreset (if specified)
3. Calls `strategy.unlearn()` with forget/retain data
4. Returns an `UnlearningResult` with timing, loss history, and metadata

### BaseStrategy
Defines the `unlearn(model, forget_loader, retain_loader, **kwargs)` interface.
All strategies return a list of per-epoch loss values.

### BaseSelector
Defines `select(dataset, n, model=None) -> indices`. Returns indices of
selected data points for the coreset.

### BaseMetric
Defines `compute(model, forget_data, retain_data) -> dict`. Returns
a dictionary of metric name → value pairs.

## Registry System

```python
# Registration (in module file)
@strategy_registry.register("gradient_ascent")
class GradientAscent(BaseStrategy): ...

# Resolution (at runtime)
strategy_cls = strategy_registry.get("gradient_ascent")
strategy = strategy_cls(**kwargs)
```

Registries are populated lazily when modules are imported. The
`__init__.py` files in each package import all submodules to trigger
registration.

## Module Dependency Graph

```
core (base classes, registry, config) ← no dependencies
  ↑
strategies, selectors, metrics ← depends on core
  ↑
models, data ← depends on core (optional torch, transformers)
  ↑
unlearners ← depends on strategies, selectors, metrics
  ↑
cli, experiments ← depends on unlearners (top-level entry points)
```

## Extending Erasus

To add a new component:
1. Create a module in the appropriate package
2. Subclass the relevant base class
3. Register with the decorator
4. Import in the package's `__init__.py`
5. Add tests
