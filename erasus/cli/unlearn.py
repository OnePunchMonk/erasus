"""
erasus unlearn — CLI command for running unlearning pipelines.

Usage::

    erasus unlearn --config configs/default.yaml
    erasus unlearn --config my_run.yaml --device cuda --epochs 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'unlearn' sub-command."""
    parser = subparsers.add_parser(
        "unlearn",
        help="Run an unlearning pipeline from a YAML config.",
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Override device (cuda / cpu).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of unlearning epochs.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save the unlearned model checkpoint.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print plan without executing.",
    )
    parser.set_defaults(func=run_unlearn)


def run_unlearn(args: argparse.Namespace) -> None:
    """Execute the unlearning pipeline."""
    from erasus.core.config import ErasusConfig
    from erasus.core.registry import strategy_registry, selector_registry

    # ---- Load config ----
    config = ErasusConfig.from_yaml(args.config)
    if args.device:
        config.device = args.device
    if args.epochs:
        config.epochs = args.epochs

    print("=" * 60)
    print("  ERASUS — Machine Unlearning Pipeline")
    print("=" * 60)
    print(f"  Model   : {config.model_name} ({config.model_type})")
    print(f"  Strategy: {config.strategy}")
    print(f"  Selector: {config.selector}")
    print(f"  Epochs  : {config.epochs}")
    print(f"  Device  : {config.device}")
    print(f"  Prune   : {config.prune_ratio:.0%}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY-RUN] Config is valid.  Full parameter dump:")
        for k, v in config.to_dict().items():
            print(f"  {k}: {v}")
        print("[DRY-RUN] Exiting without execution.")
        return

    # ---- Resolve model ----
    print(f"\n[1/4] Loading model: {config.model_name} ...")
    from erasus.core.registry import model_registry
    try:
        model_cls = model_registry.get(config.model_type)
        wrapper = model_cls(config.model_name)
        wrapper.load()
        model = wrapper.model
    except Exception as e:
        print(f"  ⚠ Could not load via model_registry ({e}).")
        print("  → Falling back: you must supply a pre-loaded model via the Python API.")
        print("  → Exiting CLI.  Use:  from erasus import ErasusUnlearner")
        sys.exit(1)

    # ---- Build unlearner ----
    print("[2/4] Building unlearner ...")
    from erasus.unlearners.multimodal_unlearner import MultimodalUnlearner

    unlearner = MultimodalUnlearner.from_model(
        model=model,
        model_type=config.model_type,
        strategy=config.strategy,
        selector=config.selector if config.selector != "none" else None,
        device=config.device,
        strategy_kwargs=config.strategy_kwargs,
        selector_kwargs=config.selector_kwargs,
    )

    # ---- Data loaders (placeholder) ----
    print("[3/4] Preparing data loaders ...")
    print("  ⚠ CLI data loading requires dataset paths in the YAML config.")
    print("  → For now, use the Python API to pass DataLoaders directly.")
    print("  → Example YAML fields: forget_data_dir, retain_data_dir")
    print("  → Skipping unlearning execution via CLI.")
    print()

    # ---- Save checkpoint ----
    if args.output:
        print(f"[4/4] (Skipped) Would save checkpoint to: {args.output}")
    else:
        print("[4/4] No --output specified; checkpoint not saved.")

    print()
    print("✅ Pipeline validated.  To run the full pipeline, use the Python API:")
    print()
    print("    from erasus import ErasusUnlearner")
    print("    unlearner = ErasusUnlearner(model, strategy='...', selector='...')")
    print("    result = unlearner.fit(forget_loader, retain_loader)")
    print()
