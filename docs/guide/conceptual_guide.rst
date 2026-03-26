Conceptual Guide — How Erasus Thinks About Unlearning
======================================================

This page explains *why* Erasus is structured the way it is, what its
opinions are, and how to decide which components to use.  If you want
API details, see the :doc:`overview` or the API reference.  This guide
is about building the right mental model.

The Organising Principle: Select → Unlearn → Verify
----------------------------------------------------

Every unlearning task in Erasus follows a three-stage pipeline:

.. code-block:: text

   ┌────────────────┐     ┌────────────────┐     ┌──────────────────┐
   │  1. SELECT      │ ──▶ │  2. UNLEARN     │ ──▶ │  3. VERIFY       │
   │  (Coreset)      │     │  (Strategy)     │     │  (Benchmark)     │
   └────────────────┘     └────────────────┘     └──────────────────┘

**Select**: Not every sample in the forget set matters equally.  A
coreset selector identifies the most influential samples — the ones
whose removal will actually change the model's behaviour.  Skipping
this step is like retraining on noise: expensive and often redundant.

**Unlearn**: A strategy modifies the model's weights to forget the
selected data.  Different strategies trade off thoroughness vs. cost.
Gradient ascent is simple and fast; Fisher-based methods are more
surgical; inference-time methods avoid weight changes entirely.

**Verify**: Unlearning is only as good as the evidence that it worked.
Standard accuracy is not enough — a model can score 0% on the forget
set while still leaking information through membership inference,
memorisation extraction, or relearning attacks.  Verification must be
adversarial.

This pipeline is the conceptual backbone of Erasus.  Every class in the
framework maps to one of these three stages.

Erasus's Opinions
-----------------

Every framework makes implicit choices.  Here are the ones Erasus
makes explicitly:

1. **Coreset selection is not optional** — it is the primary lever for
   efficiency.  Most forget sets contain redundant samples.  Selecting
   a 10% coreset typically preserves unlearning quality while cutting
   wall-clock time by 5-10×.

2. **Unlearning is not retraining** — Erasus strategies surgically
   modify weights rather than discarding and rebuilding.  This is
   faster, but requires verification that the surgery actually worked.

3. **Evaluation must be adversarial** — A passing accuracy score is
   necessary but not sufficient.  The ``UnlearningVerificationSuite``
   runs membership inference attacks, memorisation extraction, and
   relearning probes because real adversaries will.

4. **Modality matters less than you think** — The same select → unlearn
   → verify pipeline applies to LLMs, VLMs, diffusion models, and
   audio/video models.  Modality-specific logic lives in strategy
   implementations and model wrappers, not in the pipeline itself.

5. **Reproducibility requires a protocol** — Two users evaluating
   different models with different metrics produce incomparable results.
   ``UnlearningBenchmark`` ties a named protocol (TOFU, MUSE, WMDP)
   to a gold standard and returns confidence intervals so results are
   comparable across papers and teams.

When to Use Which Strategy
--------------------------

Choosing a strategy depends on your constraints.  Use this decision
tree:

.. code-block:: text

   Can you modify model weights?
   │
   ├─ NO ──▶ Inference-time methods
   │         • DExperts (detoxified experts)
   │         • Activation Steering
   │         Use when: you can't touch the checkpoint (serving, compliance)
   │
   └─ YES
      │
      ├─ Do you need certified guarantees?
      │  │
      │  └─ YES ──▶ Certified Removal / SISA
      │             Use when: legal/regulatory requirement for provable removal
      │
      └─ NO
         │
         ├─ Is the forget set small (< 1% of training data)?
         │  │
         │  └─ YES ──▶ Surgical methods
         │             • Fisher Forgetting (parameter-level precision)
         │             • Causal Tracing + Attention Surgery (LLM-specific)
         │             • Concept Erasure (diffusion-specific)
         │             Use when: targeted removal of specific facts/concepts
         │
         └─ NO (large forget set)
            │
            ├─ Is retain-set performance critical?
            │  │
            │  └─ YES ──▶ Distillation methods
            │             • SCRUB (student-teacher)
            │             • Knowledge Distillation
            │             Use when: you need to forget a lot but can't lose utility
            │
            └─ NO ──▶ Gradient methods
                      • Gradient Ascent (fastest, most aggressive)
                      • Negative Gradient / WGA (softer variants)
                      • NPO / SimNPO (preference-optimisation flavour)
                      Use when: speed matters more than precision

When to Use Which Selector
--------------------------

Selectors rank samples by importance.  The choice depends on your
compute budget and the size of the forget set.

+---------------------------+-------------+-------------------------------------------+
| Selector                  | Cost        | When to use                               |
+===========================+=============+===========================================+
| ``random``                | O(1)        | Baseline / sanity check                   |
+---------------------------+-------------+-------------------------------------------+
| ``gradient_norm``         | O(n)        | Fast default; works well in practice      |
+---------------------------+-------------+-------------------------------------------+
| ``influence``             | O(n·p)      | When you need attribution-quality ranking |
+---------------------------+-------------+-------------------------------------------+
| ``k_center``              | O(n²)       | Geometry-aware; good for diverse coresets  |
+---------------------------+-------------+-------------------------------------------+
| ``submodular``            | O(n²)       | Maximises coverage / representation       |
+---------------------------+-------------+-------------------------------------------+
| ``data_shapley``          | O(2ⁿ)       | Gold-standard valuation (small sets only) |
+---------------------------+-------------+-------------------------------------------+
| ``voting`` / ``weighted`` | varies      | Ensemble of selectors for robustness      |
+---------------------------+-------------+-------------------------------------------+

If in doubt, start with ``gradient_norm`` — it's fast and
competitive with more expensive methods on most benchmarks.

The Trainer / Module Split
--------------------------

Erasus separates *what* happens during unlearning (the module) from
*how* it is orchestrated (the trainer):

- **UnlearningModule**: you subclass this and override
  ``forget_step()`` and ``retain_step()`` to define custom unlearning
  logic.  This is analogous to PyTorch Lightning's ``LightningModule``.

- **UnlearningTrainer**: handles the training loop, validation,
  early stopping, and best-checkpoint selection.  You configure it;
  it calls your module's hooks.

This split means you can change the training schedule (add validation,
enable early stopping) without touching your unlearning logic, and
vice versa.

.. code-block:: python

   from erasus.core import UnlearningModule, UnlearningTrainer

   class MyModule(UnlearningModule):
       def forget_step(self, batch, batch_idx):
           x, y = batch
           return -F.cross_entropy(self.model(x), y)

       def retain_step(self, batch, batch_idx):
           x, y = batch
           return F.cross_entropy(self.model(x), y)

   trainer = UnlearningTrainer(
       max_epochs=10,
       validate_every=2,
       early_stopping_patience=3,
       monitor="val_forget_loss",
       monitor_mode="max",
   )
   result = trainer.fit(MyModule(model), forget_loader, retain_loader)

Coresets as First-Class Objects
-------------------------------

A ``Coreset`` is not just a list of indices — it's a composable object
you can inspect, filter, combine, and pass directly to the unlearning
pipeline:

.. code-block:: python

   from erasus.core import Coreset
   from erasus.selectors import InfluenceSelector, GradientNormSelector

   # Build from selectors
   cs_a = Coreset.from_selector(InfluenceSelector(), model, loader, k=100)
   cs_b = Coreset.from_selector(GradientNormSelector(), model, loader, k=100)

   # Compose
   consensus = cs_a.intersect(cs_b)    # samples both selectors agree on
   combined  = cs_a.union(cs_b)        # samples either selector chose

   # Filter by score
   high_impact = cs_a.filter(min_score=0.8)

   # Use directly
   result = unlearner.fit(forget_data=loader, coreset=consensus)

The ``UnlearningDataset`` Abstraction
-------------------------------------

Benchmark-specific datasets (TOFU, MUSE, WMDP) handle loading their
own data.  But for your own data, ``UnlearningDataset`` provides a
general interface:

.. code-block:: python

   from erasus.data.datasets import UnlearningDataset

   # Wrap any PyTorch dataset
   ds = UnlearningDataset(
       my_dataset,
       forget_indices=[42, 88, 123],   # sample-level
       forget_classes=[3, 7],           # class-level
       forget_weights={42: 2.0},        # importance weighting
   )

   # Streaming deletion — add/remove at any time
   ds.mark_forget([200, 201])
   ds.mark_retain([42])

   # Get loaders
   forget_loader, retain_loader = ds.to_loaders(batch_size=32)

Standardised Benchmarking
-------------------------

The ``UnlearningBenchmark`` ties a named protocol to a gold standard
so two independent evaluations are directly comparable:

.. code-block:: python

   from erasus.evaluation import UnlearningBenchmark

   bench = UnlearningBenchmark(
       protocol="tofu",
       gold_standard="retrain",
       n_runs=5,
       confidence_level=0.95,
   )
   report = bench.evaluate(
       unlearned_model=model,
       gold_model=retrained_model,
       forget_data=forget_loader,
       retain_data=retain_loader,
   )
   print(report.summary())    # table with CIs and PASS/FAIL per metric
   print(report.verdict)      # PASS / PARTIAL / FAIL

Available protocols: ``tofu``, ``muse``, ``wmdp``, ``general``.

Putting It All Together
-----------------------

A typical Erasus workflow combines all of the above:

.. code-block:: python

   from erasus import ErasusUnlearner
   from erasus.core import Coreset, UnlearningTrainer
   from erasus.data.datasets import UnlearningDataset
   from erasus.evaluation import UnlearningBenchmark

   # 1. Prepare data
   ds = UnlearningDataset(my_dataset, forget_indices=user_deletion_request)
   forget_loader, retain_loader = ds.to_loaders(batch_size=32)

   # 2. Select coreset
   unlearner = ErasusUnlearner(model, strategy="gradient_ascent", selector="gradient_norm")
   result = unlearner.fit(
       forget_loader, retain_loader,
       prune_ratio=0.1,
       epochs=10,
       validate_every=2,
       early_stopping_patience=3,
   )

   # 3. Verify
   benchmark = UnlearningBenchmark(protocol="general", n_runs=3)
   report = benchmark.evaluate(result.model, forget_loader, retain_loader)
   print(report.verdict)

The framework is designed so that each stage is independently useful.
You don't have to use all of it — start with the simplest thing that
solves your problem, and add stages as your requirements grow.
