Stable Diffusion NSFW Removal
=============================

Remove NSFW generation capability from a Stable Diffusion model.

.. literalinclude:: ../../examples/diffusion_models/stable_diffusion_nsfw.py
   :language: python
   :caption: examples/diffusion_models/stable_diffusion_nsfw.py

This example demonstrates:

- Setting up a ``DiffusionUnlearner`` for U-Net based models
- Using concept erasure to remove unsafe concepts
- Verifying the model can no longer generate the target concept

Key Points
----------

- Diffusion unlearning operates on the noise prediction network
- ``concept_erasure`` modifies attention weights to steer generation
  away from specific concepts
- The model should still generate high-quality images for safe prompts
