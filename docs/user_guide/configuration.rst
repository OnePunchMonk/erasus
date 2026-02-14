Configuration Guide
===================

Erasus uses YAML-based configuration for reproducible experiments.

Configuration File
------------------

.. code-block:: yaml

   model:
     name: clip
     checkpoint: openai/clip-vit-base-patch32

   strategy:
     name: gradient_ascent
     lr: 1e-3
     weight_decay: 0.01

   selector:
     name: herding
     prune_ratio: 0.3

   data:
     forget_size: 500
     retain_size: 2000
     batch_size: 32

   training:
     epochs: 10
     device: cuda

   metrics:
     - accuracy
     - mia
     - kl_divergence

   output:
     dir: results/
     save_model: true
     format: json

Loading Configuration
---------------------

.. code-block:: python

   from erasus.core.config import UnlearningConfig

   # From file
   config = UnlearningConfig.from_yaml("configs/default.yaml")

   # Programmatic
   config = UnlearningConfig(
       strategy="gradient_ascent",
       selector="herding",
       epochs=10,
       lr=1e-3,
   )

CLI Usage
---------

.. code-block:: bash

   erasus unlearn --config configs/default.yaml
   erasus unlearn --config configs/default.yaml --strategy scrub --epochs 5

Configuration Precedence
------------------------

1. CLI arguments (highest priority)
2. YAML config file
3. Strategy/selector defaults (lowest priority)
