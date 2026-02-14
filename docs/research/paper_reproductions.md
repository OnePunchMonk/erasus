# Paper Reproductions

Erasus includes reproduction scripts for key machine unlearning papers.

## Available Reproductions

### 1. Gradient Ascent Unlearning
- **Script**: `papers/reproductions/gradient_ascent_unlearning.py`
- **Paper**: Baseline gradient ascent method
- **Key idea**: Maximize loss on forget data to reverse learning

### 2. SCRUB (Kurmanji et al., CVPR 2024)
- **Script**: `papers/reproductions/scrub_cvpr2024.py`
- **Paper**: Selective Concept Removal Using Barriers
- **Key idea**: Two-phase approach — forget step (maximize loss)
  then retain step (minimize loss with KL barrier)

### 3. SSD (Foster et al., NeurIPS 2024)
- **Script**: `papers/reproductions/ssd_neurips2024.py`
- **Paper**: Selective Synaptic Dampening
- **Key idea**: Use Fisher Information to identify and attenuate
  parameters most associated with the forget set

### 4. Concept Erasure (Gandikota et al., ICCV 2023)
- **Script**: `papers/reproductions/concept_erasure_iccv2023.py`
- **Paper**: Erasing Concepts from Diffusion Models
- **Key idea**: Fine-tune diffusion model to steer away from
  generating specific concepts

## Running Reproductions

```bash
# Gradient Ascent baseline
python papers/reproductions/gradient_ascent_unlearning.py

# SCRUB
python papers/reproductions/scrub_cvpr2024.py

# SSD / Fisher dampening
python papers/reproductions/ssd_neurips2024.py

# Concept Erasure
python papers/reproductions/concept_erasure_iccv2023.py
```

## Adding New Reproductions

1. Create a script in `papers/reproductions/`
2. Define a simplified model (for quick demo)
3. Show pre-training → unlearning → evaluation pipeline
4. Print clear before/after metrics
5. Include paper citation in docstring
