"""
erasus.visualization — Tools for analyzing model behavior and unlearning efficacy.
"""

from erasus.visualization.embeddings import EmbeddingVisualizer
from erasus.visualization.surfaces import LossLandscapeVisualizer
from erasus.visualization.gradients import GradientVisualizer
from erasus.visualization.reports import ReportGenerator
from erasus.visualization.interactive import plot_interactive_embeddings, plot_interactive_loss

# Legacy / Utility plots
from erasus.visualization.loss_curves import plot_loss_curve
from erasus.visualization.feature_plots import plot_embeddings
from erasus.visualization.mia_plots import plot_mia_histogram, plot_mia_roc

__all__ = [
    "EmbeddingVisualizer",
    "LossLandscapeVisualizer",
    "GradientVisualizer",
    "ReportGenerator",
    "plot_interactive_embeddings",
    "plot_interactive_loss",
    "plot_loss_curve",
    "plot_embeddings",
    "plot_mia_histogram",
    "plot_mia_roc",
]
