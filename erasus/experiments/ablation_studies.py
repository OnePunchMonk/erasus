"""
erasus.experiments.ablation_studies — Automated ablation runner.

Systematically varies one component at a time (strategy, selector,
hyperparams) to measure the contribution of each component.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AblationStudy:
    """
    Run systematic ablation studies.

    Fixes all components except one, varies it across a set of
    options, and records the results.

    Parameters
    ----------
    base_config : dict
        Default configuration.
    run_fn : callable
        Function ``(config) -> dict[str, float]`` that runs a single trial.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        run_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> None:
        self.base_config = base_config
        self.run_fn = run_fn
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def ablate(
        self,
        param_name: str,
        values: List[Any],
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ablate a single parameter.

        Parameters
        ----------
        param_name : str
            Config key to vary.
        values : list
            Values to try.
        label : str, optional
            Label for this ablation group.
        """
        label = label or param_name
        group_results: List[Dict[str, Any]] = []

        for val in values:
            config = copy.deepcopy(self.base_config)
            config[param_name] = val

            t0 = time.perf_counter()
            try:
                metrics = self.run_fn(config)
                elapsed = time.perf_counter() - t0
                result = {
                    "param": param_name,
                    "value": val,
                    "metrics": metrics,
                    "time_s": round(elapsed, 3),
                    "status": "ok",
                }
            except Exception as e:
                result = {
                    "param": param_name,
                    "value": val,
                    "error": str(e),
                    "status": "failed",
                }
            group_results.append(result)

        self.results[label] = group_results
        return group_results

    def run_full_ablation(
        self,
        ablation_spec: Dict[str, List[Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run ablation for multiple parameters.

        Parameters
        ----------
        ablation_spec : dict
            ``{param_name: [value1, value2, ...]}``.

        Returns
        -------
        dict
            Results grouped by parameter name.
        """
        for param_name, values in ablation_spec.items():
            self.ablate(param_name, values)

        return self.results

    def save_results(self, path: str) -> None:
        """Save results to JSON."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = ["=" * 60, "Ablation Study Summary", "=" * 60]

        for label, group in self.results.items():
            lines.append(f"\n--- {label} ---")
            for r in group:
                val = r["value"]
                if r["status"] == "ok":
                    metric_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in r["metrics"].items()
                    )
                    lines.append(f"  {val}: {metric_str} ({r['time_s']}s)")
                else:
                    lines.append(f"  {val}: FAILED ({r.get('error', 'unknown')})")

        return "\n".join(lines)
