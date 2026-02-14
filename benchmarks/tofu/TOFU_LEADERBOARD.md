# 🏆 TOFU Benchmark Leaderboard

> **Generated**: 2026-02-15 02:21  
> **Framework**: Erasus  
> **Model**: BenchmarkModel (Linear→ReLU→Linear→ReLU→Linear, in=32, hidden=64, classes=10)  
> **Data**: Synthetic (200 forget / 800 retain samples, batch=32)  
> **Epochs**: 3 | **LR**: 1e-3  
> **Base Model Accuracy**: Forget 0.1100 / Retain 0.0788  
> **Strategies Tested**: 29 succeeded, 0 errored (out of 29 total)

## Ranking Criteria

Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).

## Leaderboard

| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|----------|-------------|-------------|--------------|--------------|
| 🥇 1 | **knowledge_distillation** | Data | 0.556 | 2.3532 | 2.2710 | 0.0300 | 0.1913 |
| 🥈 2 | **attention_unlearning** | VLM | 0.569 | 2.3640 | 2.2682 | 0.0350 | 0.1938 |
| 🥉 3 | **fisher_forgetting** | Gradient | 0.169 | -2.3531 | 0.0000 | 0.0350 | 0.0788 |
| 4 | **certified_removal** | Data | 0.165 | 2.3551 | — | 0.0400 | 0.0838 |
| 5 | **concept_erasure** | Diffusion | 0.152 | 2.3551 | — | 0.0400 | 0.0838 |
| 6 | **noise_injection** | Diffusion | 0.063 | -2.3551 | — | 0.0400 | 0.0838 |
| 7 | **token_masking** | LLM | 0.061 | 2.3551 | — | 0.0400 | 0.0838 |
| 8 | **lora** | Parameter | 0.478 | 2.3332 | 2.2844 | 0.0550 | 0.1263 |
| 9 | **sparse_aware** | Parameter | 0.092 | -2.3402 | — | 0.0600 | 0.0862 |
| 10 | **timestep_masking** | Diffusion | 0.403 | 2.3532 | 2.2656 | 0.0800 | 0.1888 |
| 11 | **safe_latents** | Diffusion | 0.357 | 2.3712 | 2.4376 | 0.1000 | 0.1375 |
| 12 | **attention_surgery** | LLM | 0.616 | — | 2.2633 | 0.1050 | 0.1850 |
| 13 | **amnesiac** | Data | 0.393 | 2.3009 | 2.3098 | 0.1100 | 0.0788 |
| 14 | **layer_freezing** | Parameter | 0.264 | 2.3010 | 2.3097 | 0.1100 | 0.0788 |
| 15 | **mask_based** | Parameter | 0.205 | -2.3012 | — | 0.1100 | 0.0788 |
| 16 | **negative_gradient** | Gradient | 0.043 | 2.3013 | — | 0.1100 | 0.0788 |
| 17 | **scrub** | Gradient | 0.361 | 0.0000 | 0.0000 | 0.1100 | 0.0788 |
| 18 | **sisa** | Data | 0.000 | — | — | 0.1100 | 0.0788 |
| 19 | **ssd** | LLM | 0.007 | — | — | 0.1100 | 0.0788 |
| 20 | **unet_surgery** | Diffusion | 0.015 | 0.0000 | — | 0.1100 | 0.0788 |
| 21 | **vision_text_split** | VLM | 0.261 | — | 2.3091 | 0.1100 | 0.0788 |
| 22 | **ensemble** | Ensemble | 0.103 | 2.3011 | 2.3098 | 0.1100 | 0.0775 |
| 23 | **neuron_pruning** | Parameter | 0.009 | — | — | 0.1150 | 0.0862 |
| 24 | **gradient_ascent** | Gradient | 0.322 | 2.3013 | 2.3093 | 0.1150 | 0.0775 |
| 25 | **saliency_unlearning** | Gradient | 0.199 | 2.3012 | 2.3095 | 0.1150 | 0.0775 |
| 26 | **modality_decoupling** | Gradient | 0.535 | 0.0798 | 0.0932 | 0.1150 | 0.0750 |
| 27 | **embedding_alignment** | LLM | 0.070 | 0.1016 | — | 0.1200 | 0.0887 |
| 28 | **contrastive_unlearning** | VLM | 0.914 | 0.0053 | 0.0025 | 0.1250 | 0.0862 |
| 29 | **causal_tracing** | LLM | 0.134 | -2.2985 | — | 0.1350 | 0.0838 |

## Summary

- **Best unlearning**: `knowledge_distillation` (Forget Acc: 0.0300)
- **Best utility preservation**: `attention_unlearning` (Retain Acc: 0.1938)
- **Fastest**: `sisa` (0.000s)
- **Total strategies**: 29
- **Successful runs**: 29
- **Errored (specialized model required)**: 0
