"""
HuggingFace Spaces-ready Gradio demo for Erasus.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_demo_loaders(forget_size: int, retain_size: int) -> Tuple[DataLoader, DataLoader]:
    forget_loader = DataLoader(
        TensorDataset(torch.randn(forget_size, 32), torch.randint(0, 10, (forget_size,))),
        batch_size=16,
        shuffle=True,
    )
    retain_loader = DataLoader(
        TensorDataset(torch.randn(retain_size, 32), torch.randint(0, 10, (retain_size,))),
        batch_size=16,
        shuffle=True,
    )
    return forget_loader, retain_loader


def run_demo_unlearning(
    strategy: str,
    coreset_fraction: float,
    epochs: int,
) -> Tuple[str, Dict[str, float], Any]:
    """Run a compact Erasus workflow for the demo UI."""
    from erasus import ErasusUnlearner
    from erasus.metrics.metric_suite import MetricSuite
    from erasus.metrics.forgetting.mia_suite import MIASuite
    import erasus.strategies  # noqa: F401
    import erasus.selectors  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    forget_loader, retain_loader = _make_demo_loaders(forget_size=64, retain_size=256)
    unlearner = ErasusUnlearner(
        model=model,
        strategy=strategy,
        selector="random",
        device=device,
    )
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        prune_ratio=coreset_fraction,
        epochs=epochs,
    )

    accuracy_metrics = MetricSuite(["accuracy"]).run(unlearner.model, forget_loader, retain_loader)
    accuracy_metrics.pop("_meta", None)
    mia_metrics = MIASuite(attacks=["loss", "mink"]).compute(unlearner.model, forget_loader, retain_loader)

    forget_acc = accuracy_metrics.get("forget_accuracy", 0.0)
    retain_acc = accuracy_metrics.get("retain_accuracy", 0.0)
    live_metrics = {
        "forget_acc": float(forget_acc),
        "retain_acc": float(retain_acc),
        "mia_auc": float(mia_metrics.get("mia_suite_mean_auc", 0.5)),
    }

    plot = _build_ablation_plot(result.forget_loss_history)
    summary = (
        f"Strategy: {strategy}\n"
        f"Coreset fraction: {coreset_fraction:.2f}\n"
        f"Epochs: {epochs}\n"
        f"Final forget loss: {result.forget_loss_history[-1]:.4f}"
    )
    return summary, live_metrics, plot


def _build_ablation_plot(loss_history: list[float]):
    """Create a simple ablation/loss curve plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = list(range(1, len(loss_history) + 1))
    ax.plot(x, loss_history or [0.0], marker="o", color="#1f77b4")
    ax.set_title("Forget Loss Ablation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Forget Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_demo():
    """Build the HF Spaces-ready Gradio Blocks app."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install gradio to use the Spaces demo: pip install gradio") from exc

    with gr.Blocks(title="Erasus Spaces Demo") as demo:
        gr.Markdown("# Erasus Unlearning Demo")
        with gr.Row():
            strategy = gr.Dropdown(
                ["gradient_ascent", "negative_gradient", "flat", "simnpo"],
                value="gradient_ascent",
                label="Strategy",
            )
            coreset_fraction = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Coreset Fraction")
            epochs = gr.Slider(1, 5, value=2, step=1, label="Epochs")
        run_button = gr.Button("Run Unlearning", variant="primary")
        summary = gr.Textbox(label="Summary", lines=6)
        live_metrics = gr.JSON(label="Live Metrics")
        plot = gr.Plot(label="Ablation Plot")
        run_button.click(
            fn=run_demo_unlearning,
            inputs=[strategy, coreset_fraction, epochs],
            outputs=[summary, live_metrics, plot],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
