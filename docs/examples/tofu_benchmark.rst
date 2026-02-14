TOFU Benchmark
==============

Run the Task of Fictitious Unlearning (TOFU) benchmark to compare
unlearning strategies.

.. literalinclude:: ../../benchmarks/tofu/run.py
   :language: python
   :caption: benchmarks/tofu/run.py

This benchmark:

- Generates synthetic fictitious data
- Runs multiple strategy × selector combinations
- Measures accuracy and forgetting quality
- Saves results to JSON

Running the Benchmark
---------------------

.. code-block:: bash

   python benchmarks/tofu/run.py

Results are saved to ``benchmarks/tofu/results/tofu_results.json``.

Interpreting Results
--------------------

The output table shows for each strategy × selector combination:

- **Strategy**: The unlearning method used
- **Selector**: The coreset selection method (or "full" for no selection)
- **Time**: Wall-clock time in seconds
- **Final Loss**: Last forget loss value (lower = less forgetting)
