.. _huggingface:

Hugging Face Hub Integration
=============================

Erasus can push unlearned models to the **Hugging Face Hub** and pull them for
further use. Install the optional dependency::

   pip install erasus[hub]
   # or
   pip install huggingface_hub datasets

Usage
-----

Push an unlearned model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from erasus.integrations import HuggingFaceHub

   hub = HuggingFaceHub(token="hf_...")  # or set HF_TOKEN env var

   hub.push_model(
       model=unlearned_model,
       repo_id="username/my-unlearned-model",
       unlearning_info={
           "strategy": "gradient_ascent",
           "selector": "herding",
           "epochs": 5,
           "forget_size": 1000,
           "elapsed_time": 120.5,
           "metrics": {"mia_auc": 0.52, "accuracy": 0.91},
       },
       create_model_card=True,
   )

This creates a repo (or updates it), uploads ``model.pt``, ``unlearning_info.json``,
and a generated README model card.

Pull a model and metadata
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   path = hub.pull_model("username/my-unlearned-model", filename="model.pt")
   info = hub.pull_unlearning_info("username/my-unlearned-model")

Load Hugging Face datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dataset = HuggingFaceHub.load_dataset("locuslab/TOFU", split="forget10")
   loader = HuggingFaceHub.dataset_to_dataloader(dataset, batch_size=32)

Deployment checklist
--------------------

* **Token**: Create a token at https://huggingface.co/settings/tokens and set
  ``HF_TOKEN`` or pass ``token=...`` to ``HuggingFaceHub()``.
* **Repo**: ``repo_id`` must be ``"org_or_user/repo_name"``. The Hub creates
  the repo on first push if it does not exist.
* **Optional deps**: Use ``pip install erasus[hub]`` or ``erasus[full]`` so
  ``huggingface_hub`` (and optionally ``datasets``) are installed.
