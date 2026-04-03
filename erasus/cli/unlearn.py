"""
erasus unlearn — CLI command for running unlearning pipelines.

Usage::

    erasus unlearn --config configs/default.yaml
    erasus unlearn --config my_run.yaml --device cuda --epochs 20
    erasus unlearn --config my_run.yaml --coreset-from influence --coreset-k 100
    erasus unlearn --config my_run.yaml --validate-every 2 --early-stop-patience 3
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
        help="Run an unlearning pipeline from a composed config.",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Optional config directory for Hydra-style composition.",
    )
    parser.add_argument(
        "--config-name", type=str, default=None,
        help="Optional config file name (without extension) for Hydra-style composition.",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="Dotted config override, e.g. strategy.name=npo or selector.prune_ratio=0.2.",
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

    # Coreset selection overrides
    coreset_group = parser.add_argument_group("coreset selection")
    coreset_group.add_argument(
        "--coreset-from", type=str, default=None,
        help="Selector name for coreset construction (e.g. influence, gradient_norm, el2n).",
    )
    coreset_group.add_argument(
        "--coreset-k", type=int, default=None,
        help="Number of samples to select for the coreset. Overrides --prune-ratio.",
    )

    # Validation / early stopping overrides
    val_group = parser.add_argument_group("validation & early stopping")
    val_group.add_argument(
        "--validate-every", type=int, default=0,
        help="Run validation metrics every N epochs. 0 disables (default).",
    )
    val_group.add_argument(
        "--early-stop-patience", type=int, default=0,
        help="Stop if monitored metric doesn't improve for N validation rounds. 0 disables.",
    )
    val_group.add_argument(
        "--early-stop-monitor", type=str, default="forget_loss",
        help="Metric name to monitor for early stopping (default: forget_loss).",
    )
    val_group.add_argument(
        "--early-stop-mode", type=str, default="max",
        choices=["min", "max"],
        help="Whether the monitored metric should be minimised or maximised (default: max).",
    )

    ckpt_group = parser.add_argument_group("checkpointing & resume")
    ckpt_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint dir or .pt from a previous run (see unlearning_checkpoint.*).",
    )
    ckpt_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save periodic checkpoints (requires --checkpoint-every > 0).",
    )
    ckpt_group.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N epochs (0 = disabled).",
    )

    track_group = parser.add_argument_group("experiment tracking (W&B / MLflow / local)")
    track_group.add_argument(
        "--tracking-backend",
        type=str,
        default=None,
        choices=["local", "wandb", "mlflow"],
        help="Override tracking backend from config (default: config or local).",
    )
    track_group.add_argument(
        "--tracking-project",
        type=str,
        default=None,
        help="Project name for wandb/mlflow.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print UnlearningProfiler timing for the fit() call.",
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
    from erasus.experiments.hydra_config import compose_experiment_config

    # ---- Load config ----
    config_path = args.config
    if config_path is None and args.config_name is not None:
        config_dir = Path(args.config_dir or ".")
        config_path = str(config_dir / f"{args.config_name}.yaml")

    if config_path is not None or args.override:
        experiment_config = compose_experiment_config(
            config_path=config_path,
            overrides=list(args.override),
        )
        config = experiment_config.to_erasus_config()
    else:
        config = ErasusConfig()

    if args.device:
        config.device = args.device
    if args.epochs:
        config.epochs = args.epochs

    # CLI overrides for selector
    if args.coreset_from:
        config.selector = args.coreset_from

    print("=" * 60)
    print("  ERASUS — Machine Unlearning Pipeline")
    print("=" * 60)
    print(f"  Experiment: {getattr(config, 'experiment_name', 'erasus_run')}")
    print(f"  Model   : {config.model_name} ({config.model_type})")
    print(f"  Strategy: {config.strategy}")
    print(f"  Selector: {config.selector}")
    print(f"  Epochs  : {config.epochs}")
    print(f"  Device  : {config.device}")
    print(f"  Prune   : {config.prune_ratio:.0%}")
    print(f"  Metrics : {getattr(config, 'metrics', ['accuracy'])}")
    tb = getattr(config, "tracking_backend", "local")
    print(f"  Tracking: {tb} (project={getattr(config, 'tracking_project', 'erasus')})")
    if args.coreset_k:
        print(f"  Coreset k: {args.coreset_k}")
    if args.validate_every > 0:
        print(f"  Validate: every {args.validate_every} epoch(s)")
    if args.early_stop_patience > 0:
        print(f"  Early stop: patience={args.early_stop_patience}, "
              f"monitor={args.early_stop_monitor} ({args.early_stop_mode})")
    print("=" * 60)

    if args.tracking_backend:
        config.tracking_backend = args.tracking_backend
    if args.tracking_project:
        config.tracking_project = args.tracking_project

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
        # ---- Build coreset if --coreset-k is specified ----
        coreset_obj = None
        if args.coreset_k and config.selector and config.selector != "none":
            print(f"\n  Building coreset: {config.selector} (k={args.coreset_k}) ...")
            from erasus.core.coreset import Coreset
            from erasus.core.registry import selector_registry as sel_reg

            sel_cls = sel_reg.get(config.selector)
            selector_inst = sel_cls(**config.selector_kwargs)
            coreset_obj = Coreset.from_selector(
                selector_inst, unlearner.model, forget_loader, k=args.coreset_k,
            )
            print(f"  ✓ Coreset: {len(coreset_obj)}/{len(forget_loader.dataset)} samples")

        # ---- Build validation metrics list ----
        validation_metrics = None
        if args.validate_every > 0:
            from erasus.metrics import MetricSuite

            validation_metrics = MetricSuite(["accuracy"]).metrics

        # ---- Run unlearning ----
        print(f"\n  Running unlearning ({config.epochs} epochs) ...")
        t0 = time.time()
        fit_kwargs = dict(
            forget_data=forget_loader,
            retain_data=retain_loader,
            prune_ratio=config.prune_ratio,
            epochs=config.epochs,
            validate_every=args.validate_every,
            early_stopping_patience=args.early_stop_patience,
            early_stopping_monitor=args.early_stop_monitor,
            early_stopping_mode=args.early_stop_mode,
        )
        if coreset_obj is not None:
            fit_kwargs["coreset"] = coreset_obj
        if validation_metrics is not None:
            fit_kwargs["validation_metrics"] = validation_metrics
        if args.resume_from:
            fit_kwargs["resume_from"] = args.resume_from
        if args.checkpoint_dir:
            fit_kwargs["checkpoint_dir"] = args.checkpoint_dir
        if args.checkpoint_every:
            fit_kwargs["checkpoint_every"] = args.checkpoint_every

        if args.profile:
            from erasus.utils.profiling import UnlearningProfiler

            profiler = UnlearningProfiler()
            with profiler.profile("unlearning_fit"):
                result = unlearner.fit(**fit_kwargs)
            print(profiler.summary())
        else:
            result = unlearner.fit(**fit_kwargs)
        elapsed = time.time() - t0
        print(f"  ✓ Unlearning complete in {elapsed:.1f}s")
        print(f"    Coreset: {result.coreset_size}/{result.original_forget_size} "
              f"({result.compression_ratio:.1%})")
        if result.forget_loss_history:
            print(f"    Final forget loss: {result.forget_loss_history[-1]:.4f}")
        if result.retain_loss_history:
            print(f"    Final retain loss: {result.retain_loss_history[-1]:.4f}")

        eval_metrics: dict = {}
        mnames = getattr(config, "metrics", None) or ["accuracy"]
        try:
            from erasus.metrics import MetricSuite

            suite = MetricSuite(mnames)
            eval_metrics = suite.run(unlearner.model, forget_loader, retain_loader or forget_loader)
            print(f"    Metrics: {eval_metrics}")
        except Exception as me:
            print(f"  ⚠ Metric evaluation skipped: {me}")

        try:
            from erasus.experiments.experiment_tracker import ExperimentTracker

            tracker = ExperimentTracker(
                name=getattr(config, "experiment_name", "erasus_run"),
                backend=getattr(config, "tracking_backend", "local"),
                project=getattr(config, "tracking_project", "erasus"),
            )
            tracker.log_config(config.to_dict())
            curves: dict[str, dict[str, list[float]]] = {
                "forget_loss": {
                    "x": [float(i) for i in range(len(result.forget_loss_history))],
                    "y": [float(x) for x in result.forget_loss_history],
                },
            }
            if result.retain_loss_history:
                curves["retain_loss"] = {
                    "x": [float(i) for i in range(len(result.retain_loss_history))],
                    "y": [float(x) for x in result.retain_loss_history],
                }
            summary_metrics: dict = {}
            if result.forget_loss_history:
                summary_metrics["final_forget_loss"] = float(result.forget_loss_history[-1])
            if result.retain_loss_history:
                summary_metrics["final_retain_loss"] = float(result.retain_loss_history[-1])
            summary_metrics.update({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})
            tracker.log_unlearning_run(
                strategy=config.strategy,
                selector=config.selector,
                metrics=summary_metrics,
                curves=curves,
                model_path=str(args.output) if args.output else None,
                metadata={"elapsed_s": elapsed},
            )
            tracker.finish()
        except Exception as te:
            print(f"  ⚠ Experiment tracking skipped: {te}")
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
