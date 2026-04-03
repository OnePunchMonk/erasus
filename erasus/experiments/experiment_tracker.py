"""
erasus.experiments.experiment_tracker — Unified experiment tracking.

Provides a backend-agnostic interface for logging unlearning experiments.
Supports local JSON logging, Weights & Biases, and MLflow.

Usage::

    tracker = ExperimentTracker("my_experiment", backend="local")
    tracker.log_config(config)
    tracker.log_metrics({"mia_auc": 0.52, "accuracy_retain": 0.94})
    tracker.log_artifact("model.pt", model_path)
    tracker.finish()
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Parameters
    ----------
    name : str
        Experiment name.
    backend : str
        One of ``"local"``, ``"wandb"``, ``"mlflow"``.
    project : str
        Project name (for wandb/mlflow).
    output_dir : str
        Local output directory (for ``"local"`` backend).
    tags : list[str], optional
        Tags for the experiment run.
    """

    def __init__(
        self,
        name: str,
        backend: str = "local",
        project: str = "erasus",
        output_dir: str = "experiments/runs",
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.backend = backend
        self.project = project
        self.output_dir = Path(output_dir)
        self.tags = tags or []

        self._run: Optional[Any] = None
        self._start_time = time.time()
        self._metrics_history: List[Dict[str, Any]] = []
        self._config: Dict[str, Any] = {}
        self._artifacts: List[str] = []

        self._init_backend()

    def _init_backend(self):
        """Initialise the chosen tracking backend."""
        if self.backend == "wandb":
            try:
                import wandb

                self._run = wandb.init(
                    project=self.project,
                    name=self.name,
                    tags=self.tags,
                    reinit=True,
                )
                print(f"  [W&B] Run started: {self._run.url}")
            except ImportError:
                print("  ⚠ wandb not installed. Falling back to local tracking.")
                self.backend = "local"

        elif self.backend == "mlflow":
            try:
                import mlflow

                mlflow.set_experiment(self.project)
                self._run = mlflow.start_run(run_name=self.name, tags={
                    t: "true" for t in self.tags
                })
                print(f"  [MLflow] Run started: {mlflow.active_run().info.run_id}")
            except ImportError:
                print("  ⚠ mlflow not installed. Falling back to local tracking.")
                self.backend = "local"

        if self.backend == "local":
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_dir = self.output_dir / f"{self.name}_{timestamp}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [Local] Tracking to: {self._run_dir}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration / hyperparameters."""
        self._config = config

        if self.backend == "wandb":
            import wandb
            wandb.config.update(config)
        elif self.backend == "mlflow":
            import mlflow
            for k, v in config.items():
                mlflow.log_param(k, v)
        else:
            with open(self._run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log a set of metrics."""
        entry = {**metrics, "_timestamp": time.time()}
        if step is not None:
            entry["_step"] = step
        self._metrics_history.append(entry)

        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics(metrics, step=step or 0)
        else:
            with open(self._run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def log_artifact(self, name: str, path: str) -> None:
        """Log a file artifact (model checkpoint, plot, etc.)."""
        self._artifacts.append(path)

        if self.backend == "wandb":
            import wandb
            wandb.save(path)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_artifact(path)
        else:
            # Copy or symlink to run directory
            src = Path(path)
            if src.exists():
                dst = self._run_dir / name
                import shutil
                shutil.copy2(src, dst)

    def log_curve(self, name: str, xs: List[float], ys: List[float]) -> None:
        """Log a curve such as forget-loss or retain-accuracy over time."""
        payload = {"name": name, "x": xs, "y": ys}
        if self.backend == "wandb":
            import wandb

            table = wandb.Table(data=[[x, y] for x, y in zip(xs, ys)], columns=["x", "y"])
            wandb.log({name: wandb.plot.line(table, "x", "y", title=name)})
        elif self.backend == "mlflow":
            curve_path = self._run_dir / f"{name}_curve.json" if hasattr(self, "_run_dir") else Path(f"{name}_curve.json")
            curve_path.write_text(json.dumps(payload, indent=2))
            import mlflow

            mlflow.log_artifact(str(curve_path))
        else:
            (self._run_dir / f"{name}_curve.json").write_text(json.dumps(payload, indent=2))

    def log_model_version(
        self,
        name: str,
        path: str,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log version metadata for an unlearned checkpoint."""
        payload = {
            "name": name,
            "path": path,
            "aliases": aliases or [],
            "metadata": metadata or {},
        }
        if self.backend == "mlflow":
            import mlflow

            version_path = self._run_dir / f"{name}_version.json" if hasattr(self, "_run_dir") else Path(f"{name}_version.json")
            version_path.write_text(json.dumps(payload, indent=2))
            mlflow.log_artifact(str(version_path))
        elif self.backend == "wandb":
            import wandb

            wandb.log({f"{name}_version": payload})
        else:
            (self._run_dir / f"{name}_version.json").write_text(json.dumps(payload, indent=2))

    def log_unlearning_run(
        self,
        *,
        strategy: str,
        selector: Optional[str],
        metrics: Dict[str, float],
        curves: Optional[Dict[str, Dict[str, List[float]]]] = None,
        model_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a complete unlearning run bundle."""
        self.log_summary(
            {
                "strategy": strategy,
                "selector": selector or "none",
                **(metadata or {}),
            }
        )
        self.log_metrics(metrics)

        for name, curve in (curves or {}).items():
            self.log_curve(name, curve.get("x", []), curve.get("y", []))

        if model_path is not None:
            self.log_model_version(
                name="unlearned_model",
                path=model_path,
                aliases=[strategy],
                metadata=metadata,
            )

    def log_model(self, model: Any, name: str = "model") -> None:
        """Save and log a PyTorch model checkpoint."""
        import torch

        if self.backend == "local":
            model_path = self._run_dir / f"{name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  [Local] Saved model: {model_path}")
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save(model.state_dict(), f.name)
                self.log_artifact(f"{name}.pt", f.name)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log a high-level summary for the run."""
        if self.backend == "wandb":
            import wandb
            for k, v in summary.items():
                wandb.run.summary[k] = v
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics({
                k: v for k, v in summary.items() if isinstance(v, (int, float))
            })
        else:
            with open(self._run_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    def finish(self) -> Dict[str, Any]:
        """Finalise tracking and return run metadata."""
        elapsed = time.time() - self._start_time

        run_info = {
            "name": self.name,
            "backend": self.backend,
            "elapsed_seconds": round(elapsed, 2),
            "n_metric_entries": len(self._metrics_history),
            "n_artifacts": len(self._artifacts),
        }

        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
        else:
            with open(self._run_dir / "run_info.json", "w") as f:
                json.dump(run_info, f, indent=2)
            print(f"  [Local] Run complete: {self._run_dir}")

        return run_info

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()

    def __repr__(self) -> str:
        return f"ExperimentTracker(name='{self.name}', backend='{self.backend}')"
