Unlearning Pipeline
===================

This guide walks through the complete unlearning pipeline from data
preparation to evaluation.

Step 1: Prepare Data
--------------------

Split your data into forget set (data to unlearn) and retain set
(data to preserve):

.. code-block:: python

   from torch.utils.data import DataLoader, Subset

   # Identify forget indices (e.g., user deletion request)
   forget_indices = get_deletion_request_indices()
   retain_indices = list(set(range(len(dataset))) - set(forget_indices))

   forget_set = Subset(dataset, forget_indices)
   retain_set = Subset(dataset, retain_indices)

   forget_loader = DataLoader(forget_set, batch_size=32, shuffle=True)
   retain_loader = DataLoader(retain_set, batch_size=32, shuffle=True)

Or use built-in datasets with pre-defined splits:

.. code-block:: python

   from erasus.data.datasets import TOFUDataset

   dataset = TOFUDataset(root="data/tofu")
   forget_loader, retain_loader = dataset.get_forget_retain_split()

Step 2: Choose a Strategy
-------------------------

Select an unlearning strategy based on your requirements:

.. list-table::
   :header-rows: 1

   * - Requirement
     - Recommended Strategy
   * - Fast, simple forgetting
     - ``gradient_ascent``
   * - Utility preservation
     - ``scrub``, ``fisher_forgetting``
   * - VLM concept removal
     - ``contrastive_unlearning``
   * - Diffusion concept erasure
     - ``concept_erasure``
   * - Privacy-aware
     - ``gradient_ascent`` + DP
   * - Maximum forgetting quality
     - ``saliency_unlearning``

Step 3: Configure Unlearner
---------------------------

.. code-block:: python

   from erasus.unlearners import ErasusUnlearner

   unlearner = ErasusUnlearner(
       model=your_model,
       strategy="gradient_ascent",
       selector="herding",         # Optional coreset selection
       device="cuda",
       strategy_kwargs={
           "lr": 1e-3,
           "weight_decay": 0.01,
       },
   )

Step 4: Run Unlearning
----------------------

.. code-block:: python

   result = unlearner.fit(
       forget_data=forget_loader,
       retain_data=retain_loader,
       prune_ratio=0.5,  # Keep 50% of forget set as coreset
       epochs=5,
   )

   print(f"Time: {result.elapsed_time:.2f}s")
   print(f"Coreset size: {result.coreset_size}")
   print(f"Final loss: {result.forget_loss_history[-1]:.4f}")

Step 5: Evaluate
----------------

.. code-block:: python

   from erasus.metrics.metric_suite import MetricSuite

   suite = MetricSuite(["accuracy", "mia", "kl_divergence"])
   metrics = suite.run(
       model=unlearner.model,
       forget_data=forget_loader,
       retain_data=retain_loader,
   )

   for name, value in metrics.items():
       if isinstance(value, float):
           print(f"{name}: {value:.4f}")

Step 6: Visualise (Optional)
-----------------------------

.. code-block:: python

   from erasus.visualization import loss_curves, feature_plots

   # Plot forget/retain loss over epochs
   loss_curves.plot(result.forget_loss_history, title="Forget Loss")

   # PCA/t-SNE of embeddings before vs after unlearning
   feature_plots.plot_feature_space(
       model=unlearner.model,
       forget_data=forget_loader,
       retain_data=retain_loader,
   )

Step 7: Certify (Optional)
---------------------------

.. code-block:: python

   from erasus.certification.verification import UnlearningVerifier

   verifier = UnlearningVerifier()
   cert = verifier.verify(
       original_model=original_model,
       unlearned_model=unlearner.model,
       forget_data=forget_loader,
   )
   print(f"Certified: {cert['verified']}")
