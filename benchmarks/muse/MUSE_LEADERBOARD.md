# MUSE Benchmark Leaderboard

> **Generated**: 2026-02-15 02:38  
> **Framework**: Erasus  
> **Model**: BenchmarkModel (32->64->64->10)  
> **Data**: Synthetic MUSE-style (64 forget / 256 retain, batch=32)  
> **Base Model Accuracy**: Forget 0.0781 / Retain 0.1250  
> **Strategies Tested**: 29 succeeded, 0 errored (out of 29 total)

## Ranking Criteria

Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).

## Leaderboard

| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|----------|-------------|-------------|--------------|--------------|
| 🥇 1 | **safe_latents** | Diffusion | 0.167 | 2.3633 | 2.2679 | 0.0156 | 0.2109 |
| 🥈 2 | **attention_unlearning** | VLM | 0.099 | 2.4556 | 2.2533 | 0.0156 | 0.2070 |
| 🥉 3 | **knowledge_distillation** | Data | 0.131 | 2.4366 | 2.2540 | 0.0156 | 0.1797 |
| 4 | **lora** | Parameter | 0.273 | 2.3862 | 2.2762 | 0.0156 | 0.1758 |
| 5 | **sparse_aware** | Parameter | 0.098 | -2.4061 | — | 0.0469 | 0.1328 |
| 6 | **certified_removal** | Data | 0.043 | 2.4092 | — | 0.0469 | 0.1289 |
| 7 | **concept_erasure** | Diffusion | 0.020 | 2.4092 | — | 0.0469 | 0.1289 |
| 8 | **fisher_forgetting** | Gradient | 0.050 | -2.4090 | 0.0000 | 0.0469 | 0.1289 |
| 9 | **noise_injection** | Diffusion | 0.026 | -2.4092 | — | 0.0469 | 0.1289 |
| 10 | **token_masking** | LLM | 0.026 | 2.4092 | — | 0.0469 | 0.1289 |
| 11 | **attention_surgery** | LLM | 0.074 | — | 2.2480 | 0.0625 | 0.1875 |
| 12 | **timestep_masking** | Diffusion | 0.166 | 0.0000 | 2.2480 | 0.0625 | 0.1875 |
| 13 | **gradient_ascent** | Gradient | 0.072 | 2.3360 | 2.2994 | 0.0625 | 0.1250 |
| 14 | **modality_decoupling** | Gradient | 0.187 | 0.0601 | 0.0617 | 0.0625 | 0.1250 |
| 15 | **neuron_pruning** | Parameter | 0.010 | — | — | 0.0625 | 0.1250 |
| 16 | **saliency_unlearning** | Gradient | 0.126 | 2.3359 | 2.2995 | 0.0625 | 0.1250 |
| 17 | **vision_text_split** | VLM | 0.097 | — | 2.2992 | 0.0625 | 0.1250 |
| 18 | **amnesiac** | Data | 0.051 | 2.3352 | 2.2998 | 0.0781 | 0.1250 |
| 19 | **causal_tracing** | LLM | 0.027 | -2.2961 | — | 0.0781 | 0.1250 |
| 20 | **ensemble** | Ensemble | 0.030 | 2.3356 | 2.2997 | 0.0781 | 0.1250 |
| 21 | **layer_freezing** | Parameter | 0.093 | 2.3354 | 2.2997 | 0.0781 | 0.1250 |
| 22 | **mask_based** | Parameter | 0.049 | -2.3353 | — | 0.0781 | 0.1250 |
| 23 | **negative_gradient** | Gradient | 0.015 | 2.3358 | — | 0.0781 | 0.1250 |
| 24 | **scrub** | Gradient | 0.153 | 0.0000 | 0.0000 | 0.0781 | 0.1250 |
| 25 | **sisa** | Data | 0.000 | — | — | 0.0781 | 0.1250 |
| 26 | **ssd** | LLM | 0.009 | — | — | 0.0781 | 0.1250 |
| 27 | **unet_surgery** | Diffusion | 0.027 | 0.0000 | — | 0.0781 | 0.1250 |
| 28 | **contrastive_unlearning** | VLM | 0.138 | 0.0257 | 0.0044 | 0.0938 | 0.1250 |
| 29 | **embedding_alignment** | LLM | 0.017 | 0.4293 | — | 0.1094 | 0.1406 |

## Summary

- **Best unlearning**: `safe_latents` (Forget Acc: 0.0156)
- **Best utility preservation**: `safe_latents` (Retain Acc: 0.2109)
- **Fastest**: `sisa` (0.000s)
- **Total strategies**: 29
- **Successful runs**: 29
- **Errored**: 0
