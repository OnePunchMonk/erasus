# Coreset Selection Analysis

## What Are Coresets?

A coreset is a small, representative subset of a dataset that
approximates the full dataset for a given objective function. In
machine unlearning, coresets are used to reduce the forget set size
while maintaining unlearning quality.

## Selector Comparison

| Selector | Complexity | Quality | Use Case |
|----------|-----------|---------|----------|
| Random | O(n) | Low | Baselines |
| Herding | O(n·k) | High | General purpose |
| K-Center | O(n·k) | High | Diverse coverage |
| Facility Location | O(n²) | Highest | Small datasets |
| CRAIG | O(n·k·d) | High | Gradient matching |
| Influence Functions | O(n·p²) | Highest | Small models |
| TracIn | O(n·T) | High | With checkpoints |
| Forgetting Score | O(n·E) | Medium | Classification |
| Active Learning | O(n·k) | Medium | Uncertainty |

Where n = dataset size, k = coreset size, d = feature dim,
p = param count, T = checkpoints, E = training epochs.

## Theoretical Guarantees

### Herding (Greedy)
Selects points closest to the class centroid. Convergence:

$$\|μ_X - μ_C\| = O(1/k)$$

### K-Center
Minimizes the maximum distance from any point to its nearest
selected point (minimax criterion).

### Facility Location
Maximizes the sum of similarities between each point and its
nearest selected point (submodular maximization).

## Evaluation Metrics

Use `erasus.selectors.quality_metrics` to evaluate coreset quality:

```python
from erasus.selectors.quality_metrics import CoresetQuality

quality = CoresetQuality()
score = quality.evaluate(
    full_dataset=full_data,
    coreset_indices=selected_indices,
    model=model,
)
```

Key metrics:
- **Coverage**: Fraction of feature space covered
- **Diversity**: Average pairwise distance in coreset
- **Representativeness**: MMD between coreset and full set
