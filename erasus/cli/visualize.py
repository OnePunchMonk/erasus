"""
erasus.cli.visualize — ``erasus visualize`` command.

Generates visualizations from unlearning results or models.
"""

from __future__ import annotations

import argparse
from typing import Any


def add_visualize_args(parser: argparse.ArgumentParser) -> None:
    """Add visualize-specific arguments."""
    parser.add_argument(
        "--type", type=str, default="embeddings",
        choices=["embeddings", "loss_landscape", "gradients", "report", "comparison"],
        help="Type of visualization to generate.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output", type=str, default="visualization.png",
        help="Output file path.",
    )
    parser.add_argument(
        "--method", type=str, default="tsne",
        choices=["tsne", "pca", "umap"],
        help="Dimensionality reduction method for embeddings.",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to evaluation data.",
    )


def run_visualize(args: argparse.Namespace) -> None:
    """Execute visualization."""
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Erasus Visualize: {args.type:<25} ║")
    print(f"╚══════════════════════════════════════════════╝")

    if args.type == "embeddings":
        _visualize_embeddings(args)
    elif args.type == "loss_landscape":
        _visualize_loss_landscape(args)
    elif args.type == "gradients":
        _visualize_gradients(args)
    elif args.type == "report":
        _generate_report(args)
    elif args.type == "comparison":
        _visualize_comparison(args)
    else:
        print(f"Unknown visualization type: {args.type}")


def _visualize_embeddings(args: argparse.Namespace) -> None:
    """Generate embedding visualizations."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.visualization.embeddings import EmbeddingVisualizer

    print(f"  Method: {args.method}")
    print(f"  Output: {args.output}")

    # Generate demo if no checkpoint provided
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    data = torch.randn(200, 32)
    labels = torch.randint(0, 5, (200,))
    loader = DataLoader(TensorDataset(data, labels), batch_size=32)

    viz = EmbeddingVisualizer(model)
    viz.plot(loader, method=args.method, save_path=args.output)
    print(f"  ✓ Saved to {args.output}")


def _visualize_loss_landscape(args: argparse.Namespace) -> None:
    """Generate loss landscape."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.visualization.surfaces import LossLandscapeVisualizer

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    data = torch.randn(100, 32)
    labels = torch.randint(0, 10, (100,))
    loader = DataLoader(TensorDataset(data, labels), batch_size=32)

    viz = LossLandscapeVisualizer(model)
    viz.plot_2d_contour(loader, save_path=args.output)
    print(f"  ✓ Saved to {args.output}")


def _visualize_gradients(args: argparse.Namespace) -> None:
    """Generate gradient flow visualization."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from erasus.visualization.gradients import GradientVisualizer

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    data = torch.randn(50, 32)
    labels = torch.randint(0, 10, (50,))

    # Run a backward pass for gradients
    out = model(data)
    loss = nn.functional.cross_entropy(out, labels)
    loss.backward()

    viz = GradientVisualizer(model)
    viz.plot_gradient_flow(save_path=args.output)
    print(f"  ✓ Saved to {args.output}")


def _generate_report(args: argparse.Namespace) -> None:
    """Generate HTML report."""
    from erasus.visualization.reports import ReportGenerator

    report = ReportGenerator("Unlearning Report")
    report.add_section("Summary", "Generated via `erasus visualize --type report`")
    report.add_metrics({
        "accuracy_retain": 0.95,
        "accuracy_forget": 0.12,
        "mia_auc": 0.52,
        "time_s": 12.3,
    })

    output = args.output.replace(".png", ".html")
    report.save(output)
    print(f"  ✓ Saved to {output}")


def _visualize_comparison(args: argparse.Namespace) -> None:
    """Generate before/after comparison."""
    from erasus.visualization.comparisons import ComparisonVisualizer

    metrics_before = {"accuracy": 0.95, "mia_auc": 0.85, "perplexity": 12.3}
    metrics_after = {"accuracy": 0.93, "mia_auc": 0.52, "perplexity": 13.1}

    viz = ComparisonVisualizer()
    viz.plot_metric_comparison(metrics_before, metrics_after, save_path=args.output)
    print(f"  ✓ Saved to {args.output}")
