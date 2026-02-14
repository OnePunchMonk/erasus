Visualization Guide
===================

Erasus provides 13 visualisation modules for analysing and
communicating unlearning results.

Loss Curves
-----------

Track forget/retain loss during unlearning:

.. code-block:: python

   from erasus.visualization.loss_curves import plot_loss_curves

   plot_loss_curves(
       forget_losses=result.forget_loss_history,
       retain_losses=result.retain_loss_history,
       title="Unlearning Progress",
   )

Feature Space Plots
-------------------

Visualise embedding space with PCA or t-SNE:

.. code-block:: python

   from erasus.visualization.feature_plots import plot_feature_space

   plot_feature_space(
       model=model,
       forget_data=forget_loader,
       retain_data=retain_loader,
       method="tsne",
   )

MIA Analysis
------------

Plot membership inference attack confidence distributions:

.. code-block:: python

   from erasus.visualization.mia_plots import plot_mia_histogram

   plot_mia_histogram(
       forget_confidences=forget_confs,
       retain_confidences=retain_confs,
   )

Attention Maps
--------------

Visualise attention patterns before/after unlearning:

.. code-block:: python

   from erasus.visualization.attention import plot_attention_maps

   plot_attention_maps(
       model=model,
       input_data=sample_batch,
       heads=[0, 4, 8],
   )

Gradient Analysis
-----------------

Compare gradient magnitudes per layer:

.. code-block:: python

   from erasus.visualization.gradients import plot_gradient_norms

   plot_gradient_norms(model, forget_loader)

Comparison Plots
----------------

Side-by-side comparison of strategies:

.. code-block:: python

   from erasus.visualization.comparisons import plot_strategy_comparison

   plot_strategy_comparison(
       results_dict={
           "Gradient Ascent": ga_metrics,
           "SCRUB": scrub_metrics,
           "Fisher": fisher_metrics,
       },
       metric_names=["accuracy", "mia_auc", "time"],
   )

Influence Maps
--------------

Visualise data point influence on model predictions:

.. code-block:: python

   from erasus.visualization.influence_maps import InfluenceMapVisualizer

   viz = InfluenceMapVisualizer()
   viz.plot_ranking(influence_scores, labels=labels, top_k=20)
   viz.plot_heatmap(scores_matrix, row_labels=forget_ids, col_labels=param_names)

Interactive Reports
-------------------

Generate comprehensive HTML reports:

.. code-block:: python

   from erasus.visualization.reports import generate_report

   generate_report(
       results=all_metrics,
       output_path="unlearning_report.html",
   )
