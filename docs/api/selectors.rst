Selectors API
=============

Selectors choose which data points to include in the forget set
(coreset selection) for efficient unlearning.

Base Selector
-------------

.. automodule:: erasus.core.base_selector
   :members:
   :undoc-members:
   :show-inheritance:

Random Selection
----------------

.. automodule:: erasus.selectors.random_selector
   :members:
   :show-inheritance:

Influence-Based
---------------

.. automodule:: erasus.selectors.influence_based.influence_functions
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.influence_based.tracin
   :members:
   :show-inheritance:

Geometry-Based
--------------

.. automodule:: erasus.selectors.geometry_based.herding
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.geometry_based.k_center
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.geometry_based.facility_location
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.geometry_based.craig
   :members:
   :show-inheritance:

Learning-Based
--------------

.. automodule:: erasus.selectors.learning_based.forgetting_score
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.learning_based.active_learning
   :members:
   :show-inheritance:

Gradient-Based
--------------

.. automodule:: erasus.selectors.gradient_based.gradient_matching
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.gradient_based.grad_norm
   :members:
   :show-inheritance:

Ensemble Selectors
------------------

.. automodule:: erasus.selectors.ensemble.stacking
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.ensemble.voting
   :members:
   :show-inheritance:

.. automodule:: erasus.selectors.ensemble.weighted_fusion
   :members:
   :show-inheritance:

Quality Metrics
---------------

.. automodule:: erasus.selectors.quality_metrics
   :members:
   :show-inheritance:

Selector Registry
-----------------

.. code-block:: python

   from erasus.core.registry import selector_registry

   # List all registered selectors
   print(selector_registry.list())

   # Get a specific selector class
   SelectorClass = selector_registry.get("herding")
