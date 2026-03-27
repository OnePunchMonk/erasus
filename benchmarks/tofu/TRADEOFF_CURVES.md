# Coreset Unlearning Tradeoff Curves

> **Generated**: 2026-03-27T23:08:25.516721
> **Strategy**: gradient_ascent
> **Data**: 200 forget / 800 retain
> **Fractions swept**: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

## Tradeoff score = (1 - forget_accuracy) x retain_accuracy

Higher is better. A perfect unlearner would have forget_accuracy = 0 and retain_accuracy = 1, giving a score of 1.0.

## Key operating point: 10% coreset

| Selector | Family | Forget Acc | Retain Acc | Tradeoff Score | Time (s) |
|----------|--------|-----------|-----------|----------------|----------|
| **influence** | Gradient-based | 0.0850 | 0.0963 | 0.0881 | 0.08 |
| **gradient_norm** | Gradient-based | 0.0850 | 0.0963 | 0.0881 | 0.04 |
| **el2n** | Gradient-based | 0.0850 | 0.0963 | 0.0881 | 0.03 |
| **tracin** | Gradient-based | 0.0850 | 0.0963 | 0.0881 | 0.04 |
| **herding** | Geometry-based | 0.0850 | 0.0963 | 0.0881 | 0.03 |
| **kcenter** | Geometry-based | 0.0850 | 0.0963 | 0.0881 | 0.03 |
| **random** (baseline) | N/A | 0.0850 | 0.0963 | 0.0881 | 0.02 |

## Full sweep results

### `influence`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0963 | 0.0881 | 0.34s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 50% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.08s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `gradient_norm`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.05s |
| 50% | 0.0850 | 0.0963 | 0.0881 | 0.05s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.05s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `el2n`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0950 | 0.0869 | 0.03s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 50% | 0.0850 | 0.0950 | 0.0869 | 0.03s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `tracin`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 50% | 0.0850 | 0.0963 | 0.0881 | 0.04s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.05s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `herding`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0950 | 0.0869 | 0.03s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 50% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `kcenter`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 1% | 0.0850 | 0.0925 | 0.0846 | 0.03s |
| 2% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 5% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 10% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 15% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 20% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 30% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 50% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 70% | 0.0850 | 0.0963 | 0.0881 | 0.03s |
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

### `forgetting_events`

| Fraction | Forget Acc | Retain Acc | Score | Time |
|----------|-----------|-----------|-------|------|
| 100% | 0.0850 | 0.0963 | 0.0881 | 0.03s |

## Interpretation

If coreset selectors (influence, gradient_norm, etc.) consistently achieve higher tradeoff scores than random selection at the same coreset fraction, the coreset thesis holds: **intelligent sample selection dominates random deletion at every operating point**.

The practical implication: practitioners can pick their desired coreset fraction based on compute budget, with the tradeoff curve showing exactly how much forgetting quality they get per unit of compute.
