# Knowledge Distillation × Coreset Selectors

> **Generated**: 2026-02-15 03:27
> **Strategy**: knowledge_distillation
> **Base Model**: BenchmarkModel (32→64→64→10)
> **Data**: 200 forget / 800 retain, prune_ratio=0.2 (coreset=40)
> **Epochs**: 3 | **LR**: 1e-3
> **Base Accuracy**: Forget 0.0950 / Retain 0.1050
> **Results**: 18 OK, 2 failed

## Ranking

Lower forget accuracy = better unlearning. Higher retain accuracy = better utility.

| Rank | Selector | Time (s) | Coreset Size | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|--------------|--------------|--------------|
| 🥇 1 | **full** | 0.350 | 200 | 0.0300 | 0.1775 |
| 🥈 2 | **random** | 0.221 | 40 | 0.0700 | 0.1875 |
| 🥉 3 | **glister** | 0.248 | 40 | 0.0750 | 0.1888 |
| 4 | **forgetting_events** | 0.383 | 40 | 0.0750 | 0.1812 |
| 5 | **kcenter** | 0.224 | 40 | 0.0750 | 0.1750 |
| 6 | **valuation_network** | 0.349 | 40 | 0.0850 | 0.1913 |
| 7 | **data_shapley** | 0.639 | 40 | 0.0850 | 0.1862 |
| 8 | **forgetting_score** | 0.686 | 40 | 0.0850 | 0.1787 |
| 9 | **kmeans** | 3.154 | 40 | 0.0900 | 0.1950 |
| 10 | **herding** | 0.254 | 40 | 0.0900 | 0.1850 |
| 11 | **submodular** | 0.240 | 40 | 0.0950 | 0.1800 |
| 12 | **representer** | 0.345 | 40 | 0.0950 | 0.1700 |
| 13 | **grad_match** | 0.276 | 40 | 0.1000 | 0.1963 |
| 14 | **gradient_norm** | 0.480 | 40 | 0.1000 | 0.1713 |
| 15 | **el2n** | 0.411 | 40 | 0.1050 | 0.1963 |
| 16 | **tracin** | 0.402 | 40 | 0.1050 | 0.1700 |
| 17 | **voting** | 0.431 | 40 | 0.1150 | 0.1688 |
| 18 | **active_learning** | 1.632 | 40 | 0.1250 | 0.1837 |

## Failed Selectors

| Selector | Error |
|----------|-------|
| **auto** | inconsistent tensor size, expected tensor [6923] and src [6922] to have the same |
| **influence** | inconsistent tensor size, expected tensor [6923] and src [6922] to have the same |

## Summary

- **Best unlearning**: `full` (Forget Acc: 0.0300)
- **Best utility preservation**: `grad_match` (Retain Acc: 0.1963)
- **Fastest**: `random` (0.221s)
