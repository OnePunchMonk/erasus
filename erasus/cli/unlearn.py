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
    parser.add_argument(
        "--forget-dir", type=str, default=None,
        help="Override path to forget-set data directory.",
    )
    parser.add_argument(
        "--retain-dir", type=str, default=None,
        help="Override path to retain-set data directory.",
    )
    parser.set_defaults(func=run_unlearn)


def _build_dataloader(data_dir: str, batch_size: int = 32) -> Optional[torch.utils.data.DataLoader]:
    """Build a DataLoader from a directory of data.

    Tries ``ImageFolder`` first (image classification), then falls back
    to a generic ``TensorDataset`` from ``.pt`` files.
    """
    from torch.utils.data import DataLoader

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  ⚠ Data path does not exist: {data_dir}")
        return None

    # Try ImageFolder (images organised in class sub-folders)
    try:
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T

        transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder(data_path, transform=transform)
        if len(dataset) > 0:
            print(f"  ✓ Loaded ImageFolder dataset: {len(dataset)} samples from {data_dir}")
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception:
        pass

    # Try loading raw .pt tensors
    pt_files = sorted(data_path.glob("*.pt"))
    if pt_files:
        tensors = [torch.load(f, map_location="cpu") for f in pt_files]
        dataset = torch.utils.data.TensorDataset(*tensors)
        print(f"  ✓ Loaded TensorDataset: {len(dataset)} samples from {len(pt_files)} .pt files")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  ⚠ Could not build DataLoader from: {data_dir}")
    return None


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

    # ---- Data loaders ----
    print("[3/4] Preparing data loaders ...")
    forget_dir = args.forget_dir or getattr(config, "forget_data_dir", None)
    retain_dir = args.retain_dir or getattr(config, "retain_data_dir", None)

    forget_loader = None
    retain_loader = None

    if forget_dir:
        forget_loader = _build_dataloader(forget_dir, batch_size=config.batch_size)
    if retain_dir:
        retain_loader = _build_dataloader(retain_dir, batch_size=config.batch_size)

    if forget_loader is not None:
        # ---- Run unlearning ----
        print(f"\n  Running unlearning ({config.epochs} epochs) ...")
        t0 = time.time()
        result = unlearner.fit(
            forget_data=forget_loader,
            retain_data=retain_loader,
            prune_ratio=config.prune_ratio,
            epochs=config.epochs,
        )
        elapsed = time.time() - t0
        print(f"  ✓ Unlearning complete in {elapsed:.1f}s")
        print(f"    Coreset: {result.coreset_size}/{result.original_forget_size} "
              f"({result.compression_ratio:.1%})")
        if result.forget_loss_history:
            print(f"    Final forget loss: {result.forget_loss_history[-1]:.4f}")
        if result.retain_loss_history:
            print(f"    Final retain loss: {result.retain_loss_history[-1]:.4f}")
    else:
        print("  ⚠ No forget dataset specified or loadable.")
        print("  → Add 'forget_data_dir' to your YAML config or use --forget-dir.")
        print("  → Alternatively, use the Python API to pass DataLoaders directly.")
        print("  → Skipping unlearning execution.")

    # ---- Save checkpoint ----
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unlearner.model.state_dict(), output_path)
        print(f"\n[4/4] ✓ Checkpoint saved to: {args.output}")
    else:
        print("\n[4/4] No --output specified; checkpoint not saved.")

    print()
    print("✅ Pipeline complete.")
    print()

