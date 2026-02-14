Changelog
=========

v0.1.0 (2026-02-14)
--------------------

Initial release of the Erasus framework.

**Core Framework**

- Base abstractions: ``BaseUnlearner``, ``BaseStrategy``,
  ``BaseSelector``, ``BaseMetric``
- Registry system for pluggable components
- YAML-based configuration with ``UnlearningConfig``

**Strategies (28)**

- Gradient methods: gradient ascent, negative gradient, saliency unlearning
- Parameter methods: Fisher forgetting, layer freezing
- Data methods: SCRUB, knowledge distillation
- VLM-specific: contrastive unlearning, attention unlearning, vision-text split
- LLM-specific: attention surgery
- Diffusion-specific: concept erasure, timestep masking, safe latents
- Ensemble strategy for combining multiple approaches

**Selectors (22)**

- Influence-based, geometry-based, gradient-based, learning-based
- Ensemble: stacking, voting, weighted fusion
- Quality metrics for coreset evaluation

**Metrics (26+)**

- Forgetting: accuracy, MIA, KL divergence, extraction attack
- Utility: BLEU, ROUGE, CLIP score, inception score, downstream tasks
- Privacy: epsilon-delta, privacy audit
- Efficiency: time, memory, speedup, FLOPs
- Benchmark runner with LaTeX export and radar plots

**Model Wrappers (18+)**

- VLM: CLIP, LLaVA, BLIP, Flamingo, ViT utilities
- LLM: GPT, Mistral, LLaMA, T5
- Diffusion: Stable Diffusion, DALL-E, Imagen, diffusion utilities
- Audio: Whisper, Wav2Vec, CLAP
- Video: VideoMAE, VideoCLIP

**Unlearners (8)**

- ErasusUnlearner (generic), VLM, LLM, Diffusion, Audio, Video
- Multimodal auto-dispatcher, Federated unlearner

**Privacy & Certification**

- Privacy accountant, DP mechanisms, gradient clipping, secure aggregation
- Certified removal, verification, bounds (PAC, influence, radius)

**Visualization (13)**

- Loss curves, feature plots, MIA plots, attention maps
- Gradient analysis, surfaces, embeddings
- Activation, influence maps, cross-modal, comparisons

**Data (7 datasets + augmentation + synthetic)**

- TOFU, WMDP, COCO, I2P, Conceptual Captions, MUSE, ImageNet
- Unlearning-aware augmentation
- Synthetic: backdoor, bias, privacy generators

**Infrastructure**

- CLI: unlearn, evaluate, benchmark, visualize commands
- Experiment tracking (local/W&B/MLflow)
- Hyperparameter search, ablation studies
- Docker, CI/CD, comprehensive documentation
- 253 unit tests passing
