# Adding New Models

## Step 1: Choose the Modality

Place your wrapper in the correct package:
- `erasus/models/vlm/` — Vision-language models
- `erasus/models/llm/` — Language models
- `erasus/models/diffusion/` — Diffusion / generative models
- `erasus/models/audio/` — Audio models
- `erasus/models/video/` — Video models

## Step 2: Create the Wrapper

```python
# erasus/models/llm/my_model.py
import torch.nn as nn

class MyModelWrapper(nn.Module):
    """Wrapper for MyModel."""

    def __init__(self, model_name="my-model-base", device="cpu"):
        super().__init__()
        self.config = type("C", (), {"model_type": "my_model"})()
        # Load or initialize your model
        self.model = self._build_model()
        self.to(device)

    def _build_model(self):
        # Your model initialization
        return nn.Sequential(...)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)
```

## Step 3: Register

Add to the package `__init__.py`:

```python
# erasus/models/llm/__init__.py
from .my_model import MyModelWrapper
```

## Step 4: Use with Unlearner

```python
from erasus.models.llm import MyModelWrapper
from erasus.unlearners import LLMUnlearner

model = MyModelWrapper()
unlearner = LLMUnlearner(model=model, strategy="gradient_ascent")
```

## Step 5: Test

Add tests verifying:
- Model instantiation
- Forward pass shape
- Compatibility with unlearner `fit()`
