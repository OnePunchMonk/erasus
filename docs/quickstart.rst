Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install erasus

   # For GPU support:
   pip install erasus[gpu]

   # For development:
   pip install -e ".[dev]"

Basic Usage
-----------

.. code-block:: python

   from erasus.unlearners import ErasusUnlearner

   # Create an unlearner
   unlearner = ErasusUnlearner(
       model=your_model,
       strategy="gradient_ascent",
       selector="random",
       device="cuda",
   )

   # Run unlearning
   result = unlearner.fit(
       forget_data=forget_loader,
       retain_data=retain_loader,
       epochs=5,
   )

   # Evaluate
   metrics = unlearner.evaluate(
       forget_data=forget_loader,
       retain_data=retain_loader,
   )

Modality-Specific Unlearners
----------------------------

Erasus provides specialised unlearners for different model types:

.. code-block:: python

   from erasus.unlearners import (
       VLMUnlearner,      # Vision-Language (CLIP, LLaVA, BLIP)
       LLMUnlearner,      # Language Models (LLaMA, GPT, Mistral)
       DiffusionUnlearner, # Diffusion (Stable Diffusion)
       AudioUnlearner,    # Audio (Whisper)
       VideoUnlearner,    # Video (VideoMAE)
   )

   # Or use the auto-dispatcher:
   from erasus.unlearners import MultimodalUnlearner
   unlearner = MultimodalUnlearner.from_model(model)

CLI Usage
---------

.. code-block:: bash

   # Run unlearning from the command line
   erasus unlearn --config configs/default.yaml

   # Evaluate an unlearned model
   erasus evaluate --config configs/default.yaml --checkpoint model.pt
