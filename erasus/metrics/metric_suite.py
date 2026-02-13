"""
erasus.metrics.metric_suite — Unified Metric Runner.

Evaluates a model with multiple metrics in one call and returns
a consolidated results dictionary.

Example::

    suite = MetricSuite(["accuracy", "mia", "perplexity"])
    results = suite.run(model, forget_loader, retain_loader)
    print(results)
    # {'accuracy_forget': 0.12, 'accuracy_retain': 0.95, 'mia_auc': 0.51, ...}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.registry import metric_registry


class MetricSuite:
    """
    Runs a collection of metrics against a model.

    Parameters
    ----------
    metrics : list[str | Any]
        Metric names (resolved via ``metric_registry``) or
        pre-instantiated metric objects.
    device : str, optional
        Device for computation.
    """

    def __init__(
        self,
        metrics: Optional[List[Union[str, Any]]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device
        self._instances: List[Any] = []

        if metrics:
            for m in metrics:
                if isinstance(m, str):
                    cls = metric_registry.get(m)
                    self._instances.append(cls())
                else:
                    self._instances.append(m)

    @property
    def metric_names(self) -> List[str]:
        return [type(m).__name__ for m in self._instances]

    def add(self, metric: Union[str, Any]) -> "MetricSuite":
        """Add a metric to the suite (fluent API)."""
        if isinstance(metric, str):
            cls = metric_registry.get(metric)
            self._instances.append(cls())
        else:
            self._instances.append(metric)
        return self

    def run(
        self,
        model: nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute all metrics and return consolidated results.

        Returns
        -------
        dict
            Keys are metric names / sub-keys, values are floats or
            nested dicts.  Also includes ``_meta`` with timing info.
        """
        results: Dict[str, Any] = {}
        timings: Dict[str, float] = {}

        for metric in self._instances:
            name = type(metric).__name__
            t0 = time.time()
            try:
                result = metric.compute(
                    model=model,
                    forget_data=forget_data,
                    retain_data=retain_data,
                    **kwargs,
                )
                if isinstance(result, dict):
                    results.update(result)
                else:
                    results[name] = result
            except Exception as e:
                results[f"{name}_error"] = str(e)
            timings[name] = time.time() - t0

        results["_meta"] = {
            "metrics_evaluated": self.metric_names,
            "timings_seconds": timings,
        }
        return results

    @classmethod
    def default_for_modality(cls, modality: str) -> "MetricSuite":
        """
        Create a MetricSuite with sensible defaults for a modality.

        Parameters
        ----------
        modality : str
            One of ``vlm``, ``llm``, ``diffusion``, ``audio``, ``video``.
        """
        defaults = {
            "vlm": ["accuracy", "mia", "zero_shot"],
            "llm": ["perplexity", "mia"],
            "diffusion": ["fid"],
            "audio": ["accuracy", "mia"],
            "video": ["accuracy", "mia"],
        }
        metric_names = defaults.get(modality, ["accuracy", "mia"])
        suite = cls()
        for name in metric_names:
            try:
                suite.add(name)
            except KeyError:
                pass  # Metric not registered; skip gracefully
        return suite

    def __repr__(self) -> str:
        return f"MetricSuite(metrics={self.metric_names})"
