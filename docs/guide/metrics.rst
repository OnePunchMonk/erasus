Metrics Guide
=============

Erasus provides 26+ metrics to evaluate unlearning quality across
four dimensions: forgetting, utility, privacy, and efficiency.

Forgetting Quality
------------------

Measures how well the model has forgotten the target data:

- **Accuracy** — Classification accuracy on forget vs retain sets.
  After unlearning, forget accuracy should drop while retain accuracy
  stays high.

- **MIA (Membership Inference Attack)** — Trains a shadow model to
  predict membership. AUC → 0.5 indicates successful unlearning.

- **KL Divergence** — Measures distribution shift between the unlearned
  model and a retrained-from-scratch model.

- **Extraction Attack** — Tests if memorised data can be extracted from
  the model after unlearning.

Model Utility
-------------

Measures preservation of useful model capabilities:

- **BLEU** — Machine translation / text generation quality
- **ROUGE** — Summarisation quality (ROUGE-N, ROUGE-L)
- **CLIP Score** — Image-text alignment quality
- **Inception Score** — Image generation quality / diversity
- **Downstream Tasks** — Performance on held-out evaluation tasks

Privacy
-------

Measures formal privacy guarantees:

- **Epsilon-Delta** — (ε, δ)-differential privacy accounting
- **Privacy Audit** — Empirical privacy leakage estimation
- **Differential Privacy** — DP-SGD compliance checking

Efficiency
----------

Measures computational cost:

- **Time Complexity** — Wall-clock time for unlearning
- **Memory Usage** — Peak GPU/CPU memory
- **Speedup** — Ratio vs retraining from scratch
- **FLOPs** — Floating point operations count

Using MetricSuite
-----------------

.. code-block:: python

   from erasus.metrics.metric_suite import MetricSuite

   # Run specific metrics
   suite = MetricSuite(["accuracy", "mia", "kl_divergence"])
   results = suite.run(model, forget_loader, retain_loader)

   # Print results
   for name, value in results.items():
       if isinstance(value, float):
           print(f"  {name}: {value:.4f}")

Benchmark Runner
----------------

For comprehensive benchmarks with statistical tests and visualisation:

.. code-block:: python

   from erasus.metrics.benchmarks import BenchmarkRunner

   runner = BenchmarkRunner(
       strategies=["gradient_ascent", "scrub", "fisher_forgetting"],
       metrics=["accuracy", "mia"],
       n_runs=3,
   )
   results = runner.run(model, forget_loader, retain_loader)
   runner.export_latex(results, "benchmark_table.tex")
