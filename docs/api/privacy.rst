Privacy API
===========

Privacy tools for differential privacy, gradient clipping,
secure aggregation, and privacy accounting.

Privacy Accountant
------------------

.. automodule:: erasus.privacy.accountant
   :members:
   :undoc-members:
   :show-inheritance:

DP Mechanisms
-------------

.. automodule:: erasus.privacy.dp_mechanisms
   :members:
   :show-inheritance:

Gradient Clipping
-----------------

.. automodule:: erasus.privacy.gradient_clipping
   :members:
   :undoc-members:
   :show-inheritance:

Secure Aggregation
------------------

.. automodule:: erasus.privacy.secure_aggregation
   :members:
   :undoc-members:
   :show-inheritance:

Certificates
------------

.. automodule:: erasus.privacy.certificates
   :members:
   :show-inheritance:

Influence Bounds
----------------

.. automodule:: erasus.privacy.influence_bounds
   :members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from erasus.privacy.gradient_clipping import GradientClipper, calibrate_noise
   from erasus.privacy.accountant import PrivacyAccountant

   # Setup DP
   clipper = GradientClipper(max_grad_norm=1.0)
   sigma = calibrate_noise(epsilon=1.0, delta=1e-5, sensitivity=0.02)

   # Track privacy budget
   accountant = PrivacyAccountant()
   accountant.step(epsilon=0.2, delta=1e-6)
   total_eps, total_delta = accountant.get_budget(advanced_composition=True)
