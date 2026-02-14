# WMDP Benchmark Leaderboard (CYBER)

> **Generated**: 2026-02-15 02:40  
> **Framework**: Erasus  
> **Model**: BenchmarkModel (32→64→64→4)  
> **Data**: Synthetic WMDP-style (48 forget / 128 retain, 4-way MC)  
> **Base Model Accuracy**: Forget 0.3333 / Retain 0.2578  
> **Strategies Tested**: 29 succeeded, 0 errored (out of 29 total)

## Ranking Criteria

Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).

## Leaderboard

| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|----------|-------------|-------------|--------------|--------------|
| 🥇 1 | **attention_unlearning** | VLM | 0.066 | 1.4560 | 1.3563 | 0.0833 | 0.3672 |
| 🥈 2 | **knowledge_distillation** | Data | 0.109 | 1.4550 | 1.3567 | 0.0833 | 0.3672 |
| 🥉 3 | **sparse_aware** | Parameter | 0.062 | -1.4254 | — | 0.1250 | 0.2891 |
| 4 | **certified_removal** | Data | 0.028 | 1.4277 | — | 0.1250 | 0.2812 |
| 5 | **concept_erasure** | Diffusion | 0.018 | 1.4277 | — | 0.1250 | 0.2812 |
| 6 | **fisher_forgetting** | Gradient | 0.051 | -1.4276 | 0.0000 | 0.1250 | 0.2812 |
| 7 | **noise_injection** | Diffusion | 0.024 | -1.4277 | — | 0.1250 | 0.2812 |
| 8 | **token_masking** | LLM | 0.017 | 1.4277 | — | 0.1250 | 0.2812 |
| 9 | **lora** | Parameter | 0.085 | 1.4095 | 1.3719 | 0.2083 | 0.3281 |
| 10 | **safe_latents** | Diffusion | 0.068 | 1.4355 | 1.3800 | 0.2500 | 0.3281 |
| 11 | **attention_surgery** | LLM | 0.049 | — | 1.3545 | 0.3333 | 0.3203 |
| 12 | **timestep_masking** | Diffusion | 0.162 | 0.0000 | 1.3551 | 0.3333 | 0.3203 |
| 13 | **contrastive_unlearning** | VLM | 0.099 | 0.0545 | 0.0107 | 0.3333 | 0.2891 |
| 14 | **causal_tracing** | LLM | 0.023 | -1.3812 | — | 0.3333 | 0.2734 |
| 15 | **scrub** | Gradient | 0.298 | 0.0000 | 0.0001 | 0.3333 | 0.2734 |
| 16 | **vision_text_split** | VLM | 0.054 | — | 1.3891 | 0.3333 | 0.2656 |
| 17 | **amnesiac** | Data | 0.030 | 1.3793 | 1.3894 | 0.3333 | 0.2578 |
| 18 | **ensemble** | Ensemble | 0.027 | 1.3794 | 1.3894 | 0.3333 | 0.2578 |
| 19 | **gradient_ascent** | Gradient | 0.170 | 1.3796 | 1.3892 | 0.3333 | 0.2578 |
| 20 | **layer_freezing** | Parameter | 0.062 | 1.3793 | 1.3894 | 0.3333 | 0.2578 |
| 21 | **mask_based** | Parameter | 0.033 | -1.3795 | — | 0.3333 | 0.2578 |
| 22 | **negative_gradient** | Gradient | 0.082 | 1.3796 | — | 0.3333 | 0.2578 |
| 23 | **saliency_unlearning** | Gradient | 0.094 | 1.3795 | 1.3893 | 0.3333 | 0.2578 |
| 24 | **sisa** | Data | 0.000 | — | — | 0.3333 | 0.2578 |
| 25 | **ssd** | LLM | 0.003 | — | — | 0.3333 | 0.2578 |
| 26 | **unet_surgery** | Diffusion | 0.004 | 0.0000 | — | 0.3333 | 0.2578 |
| 27 | **embedding_alignment** | LLM | 0.013 | 0.3153 | — | 0.3333 | 0.2500 |
| 28 | **modality_decoupling** | Gradient | 0.188 | 0.0334 | 0.0522 | 0.3333 | 0.2500 |
| 29 | **neuron_pruning** | Parameter | 0.025 | — | — | 0.3542 | 0.2734 |

## Summary

- **Best unlearning**: `attention_unlearning` (Forget Acc: 0.0833)
- **Best utility preservation**: `attention_unlearning` (Retain Acc: 0.3672)
- **Fastest**: `sisa` (0.000s)
- **Total strategies**: 29
- **Successful runs**: 29
- **Errored**: 0
