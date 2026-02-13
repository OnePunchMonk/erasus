Installation
============

Requirements
------------

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

From PyPI
---------

.. code-block:: bash

   pip install erasus

From Source
-----------

.. code-block:: bash

   git clone https://github.com/onepunchmonk/erasus.git
   cd erasus
   pip install -e .

GPU Support
-----------

.. code-block:: bash

   # Install with CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install erasus

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/onepunchmonk/erasus.git
   cd erasus
   pip install -e ".[dev]"

   # Or use the setup script:
   bash scripts/setup_env.sh       # CPU only
   bash scripts/setup_env.sh --gpu # With CUDA

Docker
------

.. code-block:: bash

   # Build and run tests
   docker compose -f docker/docker-compose.yml up test

   # Development shell
   docker compose -f docker/docker-compose.yml run dev

   # GPU benchmarks
   docker compose -f docker/docker-compose.yml up benchmark

Verifying Installation
----------------------

.. code-block:: python

   import erasus
   print("Erasus OK")

   import torch
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.cuda.is_available()}")
