Metrics API
===========

Metrics evaluate the quality of unlearning across multiple dimensions:
forgetting quality, model utility, privacy, and efficiency.

Metric Suite
------------

.. automodule:: erasus.metrics.metric_suite
   :members:
   :show-inheritance:

Forgetting Metrics
------------------

.. automodule:: erasus.metrics.forgetting.accuracy
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.forgetting.mia
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.forgetting.kl_divergence
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.forgetting.extraction_attack
   :members:
   :show-inheritance:

Utility Metrics
---------------

.. automodule:: erasus.metrics.utility.bleu
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.utility.rouge
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.utility.inception_score
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.utility.downstream_tasks
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.utility.clip_score
   :members:
   :show-inheritance:

Privacy Metrics
---------------

.. automodule:: erasus.metrics.privacy.differential_privacy
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.privacy.epsilon_delta
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.privacy.privacy_audit
   :members:
   :show-inheritance:

Efficiency Metrics
------------------

.. automodule:: erasus.metrics.efficiency.time_complexity
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.efficiency.memory_usage
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.efficiency.speedup
   :members:
   :show-inheritance:

.. automodule:: erasus.metrics.efficiency.flops
   :members:
   :show-inheritance:

Benchmarks
----------

.. automodule:: erasus.metrics.benchmarks
   :members:
   :show-inheritance:

Metric Registry
---------------

.. code-block:: python

   from erasus.core.registry import metric_registry

   # List all registered metrics
   print(metric_registry.list())

   # Run a suite of metrics
   from erasus.metrics.metric_suite import MetricSuite
   suite = MetricSuite(["accuracy", "mia", "kl_divergence"])
   results = suite.run(model, forget_loader, retain_loader)
