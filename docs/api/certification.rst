Certification API
=================

Certification tools provide formal guarantees about the quality
of machine unlearning.

Certified Removal
-----------------

.. automodule:: erasus.certification.certified_removal
   :members:
   :undoc-members:
   :show-inheritance:

Verification
------------

.. automodule:: erasus.certification.verification
   :members:
   :undoc-members:
   :show-inheritance:

Bounds
------

.. automodule:: erasus.certification.bounds
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from erasus.certification.verification import UnlearningVerifier

   verifier = UnlearningVerifier()
   result = verifier.verify(
       original_model=original,
       unlearned_model=unlearned,
       forget_data=forget_loader,
   )
   print(f"Weight distance: {result['weight_distance']:.4f}")
   print(f"Verified: {result['verified']}")
