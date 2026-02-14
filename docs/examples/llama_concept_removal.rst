LLaMA Concept Removal
=====================

Remove specific knowledge from a LLaMA-family language model.

.. literalinclude:: ../../examples/language_models/llama_concept_removal.py
   :language: python
   :caption: examples/language_models/llama_concept_removal.py

This example demonstrates:

- Setting up an ``LLMUnlearner`` for a decoder-only language model
- Using gradient ascent to unlearn specific concepts
- Measuring forgetting quality vs utility preservation

Key Points
----------

- For LLMs, the forget set typically consists of prompts and
  completions containing the target knowledge
- ``gradient_ascent`` maximises the loss on forget data, making the
  model unable to reproduce the targeted text
- Monitor perplexity on the retain set to ensure general capabilities
  are preserved
