"""
Erasus Interactive Dashboard — Gradio.

Run: python apps/dashboard_gradio.py

Requires: pip install gradio
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import gradio as gr
except ImportError:
    raise ImportError("Install gradio: pip install gradio")


def run_unlearning(strategy: str, selector: str, epochs: int, lr: float, prune_ratio: float, forget_size: int, retain_size: int):
    from erasus.unlearners import ErasusUnlearner
    from erasus.metrics.metric_suite import MetricSuite
    import erasus.strategies  # noqa: F401
    import erasus.selectors   # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)
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
    unlearner = ErasusUnlearner(
        model=model,
        strategy=strategy,
        selector=selector,
        device=device,
        selector_kwargs={"prune_ratio": prune_ratio},
        strategy_kwargs={"lr": lr},
    )
    t0 = time.time()
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=epochs,
    )
    elapsed = time.time() - t0
    suite = MetricSuite(["accuracy"])
    metrics = suite.run(unlearner.model, forget_loader, retain_loader)
    metrics.pop("_meta", None)
    acc = metrics.get("accuracy", "N/A")
    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
    summary = (
        f"**Time:** {elapsed:.2f}s  \n"
        f"**Final forget loss:** {result.forget_loss_history[-1]:.4f}  \n"
        f"**Accuracy:** {acc_str}  \n"
        f"**Metrics:** {metrics}"
    )
    loss_curve = result.forget_loss_history if result.forget_loss_history else [0.0]
    plot_data = {"x": list(range(len(loss_curve))), "y": loss_curve}
    return summary, plot_data


def main():
    with gr.Blocks(title="Erasus Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 👻 Erasus — Machine Unlearning Dashboard")
        with gr.Row():
            strategy = gr.Dropdown(
                ["gradient_ascent", "negative_gradient", "fisher_forgetting", "scrub"],
                value="gradient_ascent",
                label="Strategy",
            )
            selector = gr.Dropdown(
                ["random", "full", "gradient_norm"],
                value="random",
                label="Selector",
            )
        with gr.Row():
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            lr = gr.Number(value=1e-3, label="Learning rate")
            prune_ratio = gr.Slider(0.1, 1.0, value=0.3, step=0.1, label="Coreset ratio")
        with gr.Row():
            forget_size = gr.Slider(20, 200, value=50, step=10, label="Forget set size")
            retain_size = gr.Slider(100, 500, value=200, step=50, label="Retain set size")
        run_btn = gr.Button("Run unlearning", variant="primary")
        out_text = gr.Markdown(label="Results")
        out_plot = gr.JSON(label="Forget loss curve (steps → loss)")
        run_btn.click(
            fn=run_unlearning,
            inputs=[strategy, selector, epochs, lr, prune_ratio, forget_size, retain_size],
            outputs=[out_text, out_plot],
        )
    demo.launch()


if __name__ == "__main__":
    main()
