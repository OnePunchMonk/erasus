Strategy Selection Guide
========================

Erasus provides 28 unlearning strategies across six categories.
This guide helps you choose the right one.

Strategy Categories
-------------------

**Gradient Methods** — Modify gradients to reverse learning:

- ``gradient_ascent`` — Simple, fast; maximises loss on forget data
- ``negative_gradient`` — Applies negative gradients from forget data
- ``saliency_unlearning`` — Uses gradient saliency to target important parameters

**Parameter Methods** — Directly modify model weights:

- ``fisher_forgetting`` — Uses Fisher Information to identify and dampen relevant parameters
- ``layer_freezing`` — Freezes unrelated layers, unlearns only relevant ones

**Data Methods** — Leverage data relationships:

- ``scrub`` — Two-phase: forget step + retain barrier (SCRUB paper)
- ``knowledge_distillation`` — Teacher-student: retain teacher's knowledge, forget specific data

**VLM-Specific** — Vision-language models:

- ``contrastive_unlearning`` — Increases distance between forget image-text pairs
- ``attention_unlearning`` — Modifies cross-attention weights
- ``vision_text_split`` — Separate encoder updates

**LLM-Specific** — Language models:

- ``attention_surgery`` — Direct attention weight modification

**Diffusion-Specific** — Generative models:

- ``concept_erasure`` — Closed-form concept removal
- ``timestep_masking`` — Selective timestep training
- ``safe_latents`` — Safe Latent Diffusion constraint

Decision Matrix
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Strategy
     - Speed
     - Forgetting
     - Utility
     - Privacy
     - Modality
   * - gradient_ascent
     - ★★★
     - ★★
     - ★★
     - ★
     - All
   * - scrub
     - ★★
     - ★★★
     - ★★★
     - ★★
     - All
   * - fisher_forgetting
     - ★★
     - ★★★
     - ★★★
     - ★★
     - All
   * - concept_erasure
     - ★★★
     - ★★★
     - ★★★
     - ★★
     - Diffusion
   * - contrastive_unlearning
     - ★★
     - ★★★
     - ★★★
     - ★★
     - VLM

Combining Strategies
--------------------

Use ``EnsembleStrategy`` to combine multiple strategies:

.. code-block:: python

   from erasus.strategies.ensemble_strategy import EnsembleStrategy

   ensemble = EnsembleStrategy(
       strategies=["gradient_ascent", "fisher_forgetting"],
       weights=[0.7, 0.3],
   )
