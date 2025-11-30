# 👻 Erasus  
**Efficient Representative And Surgical Unlearning Selection**  
## Universal Machine Unlearning via Coreset Selection

### 🚧 Project Status: Upcoming / Under Active Development

Erasus is currently in the alpha research phase. The features described below represent the architectural vision and are being actively implemented.

Erasus is an upcoming Python library designed to provide a unified framework for **Machine Unlearning** across all major Foundation Model types—including Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative Diffusion Models. It surgically removes specific data from trained models without the computational cost of full retraining.

It solves the “catastrophic collapse” problem in unlearning by selecting a geometric coreset of the forget set—identifying the “support vectors of forgetting”—and unlearning only those critical samples while preserving cross-modal alignment and general model utility.

---

## ⚡ Key Features (Proposed)

### 🎯 Coreset-Driven Forgetting  
Selects the top **k% most influential “outliers”** in the forget set (inspired by UPCORE, Craig, Glister).  
Reduces compute time by up to **90%**.

### 📷📝 Multimodal Decoupling  
Specifically addresses image–text coupling in VLMs.  
Unlearns associations **without breaking visual or textual generalization**.

### 🛡️ Utility Preservation  
Introduces a **Retain-Anchor loss** to constrain model drift on safe retain data.

### 📊 Integrated Benchmarking & Visualization  
Built-in tools to visualize the **forgetting surface**, plus automated reports on model utility vs. deletion efficacy.

### 🔌 Model Agnostic  
Built on PyTorch, compatible with Hugging Face Transformers.

---

## 🌟 Extended Goals & Vision

Erasus aims to be the **“AutoML of Unlearning.”** Its long-term roadmap focuses on fully automated, modality-agnostic unlearning pipelines.

### Auto-Selector Engine  
A meta-selector will analyze your model + data distribution to choose:

- best coreset technique (geometry vs. gradient)
- best unlearning framework (sparse-aware, decoupling, etc.)

### Universal Foundation Model Coverage  
Erasus will support all popular foundation models:

- **VLMs** (CLIP, LLaVA) – Multimodal Decoupling  
- **LLMs** (Llama, Mistral) – Embedding alignment & causal masking  
- **Generative Models** (Stable Diffusion) – Noise injection + parameter scrubbing  
- **Specialized Models** (Whisper, VideoMAE)

### Universal Coreset Library  
Includes Herding, k-Center, Forgetting Events, and more.

---

## 📦 Installation (Coming Soon)

### Not yet released on PyPI
```bash
pip install erasus-unlearn
```



## 🚀 Quick Start (Preview)

Erasus provides a simple, high-level API to handle the complex math of influence functions and modality decoupling.

1. The "Scrub" Workflow

import torch
from erasus import erasusUnlearner
from erasus.selectors import AutoSelector
from transformers import CLIPModel, CLIPProcessor

### 1. Load your trained multimodal model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

### 2. Define your data
retain_loader = ... (Data to keep)
forget_loader = ... (Sensitive data to remove)

### 3. Initialize the Auto-Selector (Goal)
#### Automatically chooses the best coreset strategy for CLIP models
selector = AutoSelector(model_type="multimodal", goal="balanced_utility")
forget_coreset = selector.select(
    model=model, 
    data_loader=forget_loader, 
    prune_ratio=0.10 
)

### 4. Initialize Unlearner with Multimodal Strategy
unlearner = erasusUnlearner(
    model=model,
    strategy="modality_decoupling", # Splits Image/Text loss terms
    lr=1e-4
)

### 5. Perform the Unlearning & Visualize
clean_model, stats = unlearner.fit(
    forget_data=forget_coreset,
    retain_data=retain_loader,
    epochs=5,
    visualize=True # Generates pre/post unlearning heatmaps
)




## 🧠 How It Works

Erasus operates in a three-stage pipeline:

Embedding Projection: It maps the forget_set into the model's latent space.

Coreset Selection (The "Erasus" Step): It calculates the Gradient Sensitivity of each sample. Samples that lie in the center of the data distribution are often redundant. Erasus selects the boundary cases (outliers) that define the decision boundary for that specific concept.

Theory based on "Utility-Preserving Coreset Selection for Balanced Unlearning" (2025).

Decoupled Gradient Ascent: It applies Gradient Ascent to maximize the loss on the coreset, but with a twist: it penalizes the movement of the image encoder weights differently than the text encoder weights to prevent breaking the model's general vision capabilities.

## 🗺️ Roadmap

### Phase 1: Core Framework
- Implement `modality_decoupling` for CLIP.
- Implement basic `gradient_geometry` coreset selection.

### Phase 2: Expansion
- Add support for Diffusion Models (Stable Diffusion).
- Integration with `peft` (LoRA-based unlearning).

### Phase 3: The "Auto-Unlearner"
- Develop the heuristic engine for auto-selecting strategies.
- Build the `erasusBench` evaluation suite for automated MIA (Membership Inference Attack) testing.

## 🤝 Contributing

We welcome contributions!

## 📜 License

MIT
