"""
Erasus Interactive Dashboard — Streamlit.

Run: streamlit run apps/dashboard_streamlit.py

Requires: pip install streamlit
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import streamlit as st

# Lazy import erasus to avoid loading until needed
def _get_unlearner():
    from erasus.unlearners import ErasusUnlearner
    from erasus.metrics.metric_suite import MetricSuite
    import erasus.strategies  # noqa: F401
    import erasus.selectors   # noqa: F401
    return ErasusUnlearner, MetricSuite


st.set_page_config(page_title="Erasus Dashboard", page_icon="👻", layout="wide")
st.title("👻 Erasus — Machine Unlearning Dashboard")
st.caption("Configure a small unlearning run and view metrics.")

with st.sidebar:
    st.header("Configuration")
    strategy = st.selectbox(
        "Strategy",
        ["gradient_ascent", "negative_gradient", "fisher_forgetting", "scrub"],
        index=0,
    )
    selector = st.selectbox(
        "Selector",
        ["random", "full", "gradient_norm"],
        index=0,
    )
    epochs = st.slider("Epochs", 1, 10, 3)
    lr = st.number_input("Learning rate", value=1e-3, format="%e", step=1e-4)
    prune_ratio = st.slider("Coreset ratio (selector)", 0.1, 1.0, 0.3, 0.1)
    forget_size = st.slider("Forget set size", 20, 200, 50)
    retain_size = st.slider("Retain set size", 100, 500, 200)

if st.button("Run unlearning", type="primary"):
    ErasusUnlearner, MetricSuite = _get_unlearner()
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
    progress = st.progress(0)
    status = st.empty()
    status.info("Running unlearning...")
    t0 = time.time()
    result = unlearner.fit(
        forget_data=forget_loader,
        retain_data=retain_loader,
        epochs=epochs,
    )
    elapsed = time.time() - t0
    progress.progress(1.0)
    status.success(f"Done in {elapsed:.2f}s")
    suite = MetricSuite(["accuracy"])
    metrics = suite.run(unlearner.model, forget_loader, retain_loader)
    metrics.pop("_meta", None)
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Time (s)", f"{elapsed:.2f}")
    with col2:
        st.metric("Final forget loss", f"{result.forget_loss_history[-1]:.4f}" if result.forget_loss_history else "N/A")
    with col3:
        acc = metrics.get("accuracy")
        st.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc))
    st.json(metrics)
    if result.forget_loss_history:
        st.line_chart({"forget_loss": result.forget_loss_history})

st.divider()
st.markdown("**Erasus** — Efficient Representative And Surgical Unlearning Selection. [GitHub](https://github.com/OnePunchMonk/erasus)")
