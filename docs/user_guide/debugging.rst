Debugging Guide
===============

Common issues and how to diagnose them.

Model Not Forgetting
--------------------

**Symptom**: Forget accuracy remains high after unlearning.

**Causes**:

1. **Learning rate too low** — Increase ``lr`` (try 1e-2 or 1e-1)
2. **Too few epochs** — Run for more epochs
3. **Wrong strategy** — Try ``gradient_ascent`` as a baseline
4. **Retain regularisation too strong** — Lower the retain weight

.. code-block:: python

   # Diagnostic: check loss trajectory
   result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=20)
   print(result.forget_loss_history)
   # Should see increasing trend

Retain Performance Degraded
----------------------------

**Symptom**: Retain accuracy drops significantly after unlearning.

**Causes**:

1. **Learning rate too high** — Reduce ``lr``
2. **Too many epochs** — Use early stopping
3. **No retain regularisation** — Use ``scrub`` or ``fisher_forgetting``

.. code-block:: python

   # Try early stopping
   from erasus.utils.early_stopping import EarlyStopping

   early_stop = EarlyStopping(patience=3, metric="retain_loss")

CUDA Out of Memory
------------------

**Symptom**: ``RuntimeError: CUDA out of memory``

**Solutions**:

1. Reduce ``batch_size``
2. Use gradient accumulation
3. Use ``selector="herding"`` with low ``prune_ratio``
4. Use ``layer_freezing`` strategy to freeze most layers

Strategy Not Found
------------------

**Symptom**: ``KeyError: 'my_strategy' not found in registry``

**Solution**: Ensure the strategy module is imported before use:

.. code-block:: python

   import erasus.strategies  # Triggers all registrations

Debugging Tips
--------------

1. **Enable verbose logging**:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.DEBUG)

2. **Check loss history** for convergence issues
3. **Visualise embeddings** before/after to understand the effect
4. **Start with a small model** and synthetic data to isolate bugs
