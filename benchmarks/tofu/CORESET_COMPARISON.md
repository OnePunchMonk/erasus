# Knowledge Distillation × Coreset Selectors

> **Generated**: 2026-02-15 02:28
> **Strategy**: knowledge_distillation
> **Base Model**: BenchmarkModel (32→64→64→10)
> **Data**: 200 forget / 800 retain, prune_ratio=0.2 (coreset=40)
> **Epochs**: 3 | **LR**: 1e-3
> **Base Accuracy**: Forget 0.1150 / Retain 0.0963
> **Results**: 17 OK, 3 failed

## Ranking

Lower forget accuracy = better unlearning. Higher retain accuracy = better utility.

| Rank | Selector | Time (s) | Coreset Size | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|--------------|--------------|--------------|
| 🥇 1 | **full** | 0.834 | 200 | 0.0350 | 0.1750 |
| 🥈 2 | **submodular** | 1.559 | 40 | 0.0850 | 0.1963 |
| 🥉 3 | **random** | 0.864 | 40 | 0.0850 | 0.1862 |
| 4 | **valuation_network** | 0.574 | 40 | 0.0850 | 0.1862 |
| 5 | **herding** | 0.764 | 40 | 0.0900 | 0.1925 |
| 6 | **glister** | 0.740 | 40 | 0.0900 | 0.1825 |
| 7 | **voting** | 0.960 | 40 | 0.0900 | 0.1812 |
| 8 | **kcenter** | 0.731 | 40 | 0.0900 | 0.1750 |
| 9 | **forgetting_score** | 0.883 | 40 | 0.0950 | 0.1913 |
| 10 | **data_shapley** | 1.158 | 40 | 0.0950 | 0.1800 |
| 11 | **tracin** | 1.674 | 40 | 0.0950 | 0.1787 |
| 12 | **active_learning** | 1.215 | 40 | 0.1050 | 0.1875 |
| 13 | **forgetting_events** | 1.901 | 40 | 0.1100 | 0.2062 |
| 14 | **kmeans** | 3.188 | 40 | 0.1150 | 0.1963 |
| 15 | **grad_match** | 3.082 | 40 | 0.1150 | 0.1800 |
| 16 | **gradient_norm** | 1.325 | 40 | 0.1200 | 0.1750 |
| 17 | **el2n** | 1.241 | 40 | 0.1300 | 0.1925 |

## Failed Selectors

| Selector | Error |
|----------|-------|
| **auto** | One of the differentiated Tensors appears to not have been used in the graph. Se |
| **influence** | One of the differentiated Tensors appears to not have been used in the graph. Se |
| **representer** | One of the differentiated Tensors appears to not have been used in the graph. Se |

## Summary
- **Best unlearning**: `full` (Forget Acc: 0.0350)
- **Best utility preservation**: `forgetting_events` (Retain Acc: 0.2062)
- **Fastest**: `valuation_network` (0.574s)
- **Fastest**: 
