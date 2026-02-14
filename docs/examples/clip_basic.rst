CLIP Basic Example
==================

Basic unlearning with CLIP using contrastive unlearning strategy.

.. literalinclude:: ../../examples/clip_basic.py
   :language: python
   :caption: examples/clip_basic.py

This example demonstrates:

- Loading a CLIP-like model
- Creating synthetic image-text forget/retain data
- Applying contrastive unlearning
- Evaluating results with ``MetricSuite``

Key Points
----------

- CLIP uses dual encoders (vision + text), so unlearning must
  consider the joint embedding space
- ``contrastive_unlearning`` increases the distance between matched
  forget image-text pairs while preserving retain alignment
- Use ``VLMUnlearner`` for CLIP-family models
