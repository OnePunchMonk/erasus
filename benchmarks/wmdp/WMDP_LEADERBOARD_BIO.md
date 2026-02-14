# WMDP Benchmark Leaderboard (BIO)

> **Generated**: 2026-02-15 02:40  
> **Framework**: Erasus  
> **Model**: BenchmarkModel (32→64→64→4)  
> **Data**: Synthetic WMDP-style (48 forget / 128 retain, 4-way MC)  
> **Base Model Accuracy**: Forget 0.2917 / Retain 0.2656  
> **Strategies Tested**: 29 succeeded, 0 errored (out of 29 total)

## Ranking Criteria

Strategies are ranked by **forget accuracy** (lower = better unlearning), with ties broken by **retain accuracy** (higher = better utility preservation).

## Leaderboard

| Rank | Strategy | Category | Time (s) | Forget Loss | Retain Loss | Forget Acc ↓ | Retain Acc ↑ |
|------|----------|----------|----------|-------------|-------------|--------------|--------------|
| 🥇 1 | **attention_unlearning** | VLM | 0.054 | 1.4259 | 1.3682 | 0.1250 | 0.3984 |
| 🥈 2 | **knowledge_distillation** | Data | 0.184 | 1.4253 | 1.3683 | 0.1458 | 0.3984 |
| 🥉 3 | **certified_removal** | Data | 0.060 | 1.4102 | — | 0.1667 | 0.2578 |
| 4 | **concept_erasure** | Diffusion | 0.019 | 1.4102 | — | 0.1667 | 0.2578 |
| 5 | **fisher_forgetting** | Gradient | 0.044 | -1.4101 | 0.0000 | 0.1667 | 0.2578 |
| 6 | **noise_injection** | Diffusion | 0.076 | -1.4102 | — | 0.1667 | 0.2578 |
| 7 | **sparse_aware** | Parameter | 0.069 | -1.4079 | — | 0.1667 | 0.2578 |
| 8 | **token_masking** | LLM | 0.019 | 1.4102 | — | 0.1667 | 0.2578 |
| 9 | **lora** | Parameter | 0.108 | 1.3957 | 1.3810 | 0.2292 | 0.3125 |
| 10 | **attention_surgery** | LLM | 0.040 | — | 1.3593 | 0.2708 | 0.3516 |
| 11 | **safe_latents** | Diffusion | 0.251 | 1.4001 | 1.3922 | 0.2917 | 0.2734 |
| 12 | **amnesiac** | Data | 0.056 | 1.3728 | 1.3955 | 0.2917 | 0.2656 |
| 13 | **ensemble** | Ensemble | 0.016 | 1.3729 | 1.3956 | 0.2917 | 0.2656 |
| 14 | **gradient_ascent** | Gradient | 0.089 | 1.3730 | 1.3954 | 0.2917 | 0.2656 |
| 15 | **layer_freezing** | Parameter | 0.080 | 1.3728 | 1.3955 | 0.2917 | 0.2656 |
| 16 | **mask_based** | Parameter | 0.021 | -1.3732 | — | 0.2917 | 0.2656 |
| 17 | **modality_decoupling** | Gradient | 0.148 | 0.0163 | 0.0235 | 0.2917 | 0.2656 |
| 18 | **negative_gradient** | Gradient | 0.039 | 1.3730 | — | 0.2917 | 0.2656 |
| 19 | **saliency_unlearning** | Gradient | 0.258 | 1.3729 | 1.3955 | 0.2917 | 0.2656 |
| 20 | **scrub** | Gradient | 0.094 | 0.0000 | 0.0001 | 0.2917 | 0.2656 |
| 21 | **sisa** | Data | 0.000 | — | — | 0.2917 | 0.2656 |
| 22 | **ssd** | LLM | 0.003 | — | — | 0.2917 | 0.2656 |
| 23 | **unet_surgery** | Diffusion | 0.006 | 0.0000 | — | 0.2917 | 0.2656 |
| 24 | **vision_text_split** | VLM | 0.042 | — | 1.3951 | 0.2917 | 0.2656 |
| 25 | **timestep_masking** | Diffusion | 0.094 | 0.0000 | 1.3708 | 0.3125 | 0.4219 |
| 26 | **causal_tracing** | LLM | 0.025 | -1.3828 | — | 0.3125 | 0.2812 |
| 27 | **neuron_pruning** | Parameter | 0.037 | — | — | 0.3125 | 0.2812 |
| 28 | **contrastive_unlearning** | VLM | 0.164 | 0.0232 | 0.0055 | 0.3125 | 0.2578 |
| 29 | **embedding_alignment** | LLM | 0.015 | 0.3445 | — | 0.3125 | 0.2500 |

## Summary

- **Best unlearning**: `attention_unlearning` (Forget Acc: 0.1250)
- **Best utility preservation**: `timestep_masking` (Retain Acc: 0.4219)
- **Fastest**: `sisa` (0.000s)
- **Total strategies**: 29
- **Successful runs**: 29
- **Errored**: 0
