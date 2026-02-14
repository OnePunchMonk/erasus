Selector Guide
==============

Selectors choose which data points to include in the coreset for
efficient unlearning. Instead of unlearning on the entire forget
set, a well-chosen coreset can achieve similar results in less time.

When to Use Selectors
---------------------

- **Large forget sets** — Reduce compute cost with a representative coreset
- **Streaming deletions** — Prioritise the most impactful data points
- **Resource constraints** — Run unlearning within a time/memory budget

Selector Categories
-------------------

**Influence-Based** — Select by influence on model parameters:

- ``influence_functions`` — Compute Hessian-based influence scores
- ``tracin`` — Trace gradient inner products across checkpoints

**Geometry-Based** — Select by feature space coverage:

- ``herding`` — Greedy selection closest to class centroids
- ``k_center`` — Minimise maximum distance to nearest selected point
- ``facility_location`` — Maximise coverage of feature space
- ``craig`` — Coreset for Accelerating Incremental Gradient

**Learning-Based** — Select by training dynamics:

- ``forgetting_score`` — Points frequently forgotten during training
- ``active_learning`` — Uncertainty sampling for informative points

**Gradient-Based** — Select by gradient properties:

- ``gradient_matching`` — Match full-set gradients with coreset
- ``grad_norm`` — Select by gradient magnitude

**Ensemble** — Combine multiple selectors:

- ``stacking`` — Sequential selector refinement
- ``voting`` — Majority vote across selectors
- ``weighted_fusion`` — Weighted combination of selector scores

Quick Comparison
----------------

.. list-table::
   :header-rows: 1

   * - Selector
     - Speed
     - Quality
     - Memory
     - Best For
   * - random
     - ★★★
     - ★
     - ★★★
     - Baselines
   * - herding
     - ★★★
     - ★★★
     - ★★
     - General use
   * - influence_functions
     - ★
     - ★★★
     - ★
     - Small models
   * - forgetting_score
     - ★★
     - ★★★
     - ★★
     - Classification

Usage
-----

.. code-block:: python

   unlearner = ErasusUnlearner(
       model=model,
       strategy="gradient_ascent",
       selector="herding",
       device="cuda",
   )

   result = unlearner.fit(
       forget_data=forget_loader,
       retain_data=retain_loader,
       prune_ratio=0.3,  # Keep 30% as coreset
   )
