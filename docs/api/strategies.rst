Strategies API
==============

Machine unlearning strategies modify a trained model to remove the influence
of specific data points while preserving model utility.

Base Strategy
-------------

.. automodule:: erasus.core.base_strategy
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Methods
----------------

.. automodule:: erasus.strategies.gradient_methods.gradient_ascent
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.gradient_methods.negative_gradient
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.gradient_methods.saliency_unlearning
   :members:
   :show-inheritance:

Parameter Methods
-----------------

.. automodule:: erasus.strategies.parameter_methods.fisher_forgetting
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.parameter_methods.layer_freezing
   :members:
   :show-inheritance:

Data Methods
------------

.. automodule:: erasus.strategies.data_methods.scrub
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.data_methods.knowledge_distillation
   :members:
   :show-inheritance:

VLM-Specific
-------------

.. automodule:: erasus.strategies.vlm_specific.contrastive_unlearning
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.vlm_specific.attention_unlearning
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.vlm_specific.vision_text_split
   :members:
   :show-inheritance:

LLM-Specific
-------------

.. automodule:: erasus.strategies.llm_specific.attention_surgery
   :members:
   :show-inheritance:

Diffusion-Specific
-------------------

.. automodule:: erasus.strategies.diffusion_specific.concept_erasure
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.diffusion_specific.timestep_masking
   :members:
   :show-inheritance:

.. automodule:: erasus.strategies.diffusion_specific.safe_latents
   :members:
   :show-inheritance:

Ensemble
--------

.. automodule:: erasus.strategies.ensemble_strategy
   :members:
   :show-inheritance:

Strategy Registry
-----------------

All strategies are registered via the global registry and can be
accessed by name:

.. code-block:: python

   from erasus.core.registry import strategy_registry

   # List all registered strategies
   print(strategy_registry.list())

   # Get a specific strategy class
   StrategyClass = strategy_registry.get("gradient_ascent")
