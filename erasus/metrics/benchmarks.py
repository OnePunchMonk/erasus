"""
erasus.metrics.benchmarks — Unified benchmark runner.

Provides a publication-ready benchmark framework that evaluates
unlearning across 5 dimensions: forgetting efficacy, utility
preservation, efficiency, privacy, and scalability.

Generates LaTeX tables, radar plots, and statistical significance tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ErasusBenchmark:
    """
    Unified benchmark runner for machine unlearning evaluation.

    Evaluates across 5 dimensions and produces publication-ready output.

    Parameters
    ----------
    name : str
        Benchmark name.
    output_dir : str
        Directory for output artefacts.
    dimensions : list[str], optional
        Which dimensions to evaluate (default: all 5).
    """

    ALL_DIMENSIONS = ("forgetting", "utility", "efficiency", "privacy", "scalability")

    def __init__(
        self,
        name: str = "erasus_benchmark",
        output_dir: str = "benchmark_results",
        dimensions: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dimensions = list(dimensions or self.ALL_DIMENSIONS)
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        method_name: str = "default",
        original_model: Optional[nn.Module] = None,
        retrained_model: Optional[nn.Module] = None,
        unlearning_time: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run full benchmark evaluation for a method.

        Returns
        -------
        dict
            Nested results ``{dimension: {metric: value}}``.
        """
        results: Dict[str, Any] = {"method": method_name}

        if "forgetting" in self.dimensions:
            results["forgetting"] = self._eval_forgetting(model, forget_loader, retain_loader)

        if "utility" in self.dimensions:
            results["utility"] = self._eval_utility(model, retain_loader)

        if "efficiency" in self.dimensions:
            results["efficiency"] = self._eval_efficiency(
                model, forget_loader, unlearning_time=unlearning_time
            )

        if "privacy" in self.dimensions:
            results["privacy"] = self._eval_privacy(model, forget_loader, retain_loader)

        if "scalability" in self.dimensions:
            results["scalability"] = self._eval_scalability(model)

        self.results[method_name] = results
        return results

    def compare(self, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare results across methods."""
        if methods is None:
            methods = list(self.results.keys())

        comparison: Dict[str, Any] = {"methods": methods, "dimensions": {}}
        for dim in self.dimensions:
            comparison["dimensions"][dim] = {
                m: self.results.get(m, {}).get(dim, {}) for m in methods
            }
        return comparison

    # ------------------------------------------------------------------
    # Output generation
    # ------------------------------------------------------------------

    def to_latex_table(self, methods: Optional[List[str]] = None) -> str:
        """Generate a LaTeX table comparing methods across dimensions."""
        if methods is None:
            methods = list(self.results.keys())

        # Collect all metrics across dimensions
        all_metrics: Dict[str, List[str]] = {}
        for dim in self.dimensions:
            for m in methods:
                dim_results = self.results.get(m, {}).get(dim, {})
                if dim not in all_metrics:
                    all_metrics[dim] = []
                for metric_name in dim_results:
                    if metric_name not in all_metrics[dim]:
                        all_metrics[dim].append(metric_name)

        # Build LaTeX
        n_cols = len(methods) + 2  # Dimension + Metric + methods
        header = " & ".join(["Dimension", "Metric"] + methods)
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{self.name} Results}}",
            rf"\begin{{tabular}}{{{'l' * 2}{'c' * len(methods)}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
        ]

        for dim in self.dimensions:
            metrics = all_metrics.get(dim, [])
            for i, metric in enumerate(metrics):
                dim_label = dim.capitalize() if i == 0 else ""
                values = []
                for m in methods:
                    val = self.results.get(m, {}).get(dim, {}).get(metric, "—")
                    if isinstance(val, float):
                        values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))
                row = " & ".join([dim_label, metric] + values)
                lines.append(row + r" \\")
            if dim != self.dimensions[-1]:
                lines.append(r"\midrule")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_json(self, path: Optional[str] = None) -> str:
        """Export results as JSON."""
        data = json.dumps(self.results, indent=2, default=str)
        if path:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def save_all(self) -> Dict[str, str]:
        """Save all output artefacts."""
        paths: Dict[str, str] = {}

        # JSON
        json_path = str(self.output_dir / f"{self.name}.json")
        self.to_json(json_path)
        paths["json"] = json_path

        # LaTeX
        latex_path = str(self.output_dir / f"{self.name}.tex")
        Path(latex_path).write_text(self.to_latex_table(), encoding="utf-8")
        paths["latex"] = latex_path

        # Radar plot
        try:
            radar_path = str(self.output_dir / f"{self.name}_radar.png")
            self.plot_radar(save_path=radar_path)
            paths["radar"] = radar_path
        except ImportError:
            pass

        return paths

    def plot_radar(self, save_path: Optional[str] = None) -> Any:
        """Generate radar/spider plot comparing methods."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for radar plots.")

        methods = list(self.results.keys())
        if not methods:
            return None

        # Aggregate per-dimension scores (mean of metrics)
        dim_scores: Dict[str, List[float]] = {dim: [] for dim in self.dimensions}
        for m in methods:
            for dim in self.dimensions:
                vals = self.results.get(m, {}).get(dim, {})
                if vals:
                    numeric = [v for v in vals.values() if isinstance(v, (int, float))]
                    dim_scores[dim].append(np.mean(numeric) if numeric else 0.0)
                else:
                    dim_scores[dim].append(0.0)

        # Normalise to [0, 1] per dimension
        for dim in self.dimensions:
            arr = np.array(dim_scores[dim])
            lo, hi = arr.min(), arr.max()
            rng = hi - lo if hi > lo else 1.0
            dim_scores[dim] = ((arr - lo) / rng).tolist()

        n_dims = len(self.dimensions)
        angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        for i, m in enumerate(methods):
            values = [dim_scores[dim][i] for dim in self.dimensions]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=m)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([d.capitalize() for d in self.dimensions])
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.set_title(f"{self.name} — Method Comparison", pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig

    @staticmethod
    def statistical_test(
        scores_a: List[float],
        scores_b: List[float],
        test: str = "wilcoxon",
    ) -> Dict[str, float]:
        """
        Run a statistical significance test between two methods.

        Parameters
        ----------
        scores_a, scores_b : list[float]
            Scores from two methods across multiple runs/folds.
        test : str
            ``"wilcoxon"`` or ``"paired_t"``.
        """
        from scipy import stats

        if test == "wilcoxon":
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
        elif test == "paired_t":
            stat, p_value = stats.ttest_rel(scores_a, scores_b)
        else:
            raise ValueError(f"Unknown test: {test}")

        return {
            "test": test,
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
        }

    # ------------------------------------------------------------------
    # Dimension evaluators
    # ------------------------------------------------------------------

    def _eval_forgetting(
        self, model: nn.Module, forget_loader: DataLoader, retain_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate forgetting efficacy."""
        device = next(model.parameters()).device
        model.eval()
        results: Dict[str, float] = {}

        # Forget accuracy (should be low)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        results["forget_accuracy"] = correct / max(total, 1)

        # Forget confidence (should be low)
        confidences: list = []
        with torch.no_grad():
            for batch in forget_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=-1)
                confidences.extend(probs.max(dim=-1).values.cpu().numpy().tolist())
        results["forget_confidence"] = float(np.mean(confidences)) if confidences else 0.0

        return results

    def _eval_utility(self, model: nn.Module, retain_loader: DataLoader) -> Dict[str, float]:
        """Evaluate utility preservation."""
        device = next(model.parameters()).device
        model.eval()
        results: Dict[str, float] = {}

        correct, total = 0, 0
        with torch.no_grad():
            for batch in retain_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        results["retain_accuracy"] = correct / max(total, 1)

        return results

    def _eval_efficiency(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        unlearning_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate computational efficiency."""
        n_params = sum(p.numel() for p in model.parameters())
        results: Dict[str, float] = {
            "n_parameters": float(n_params),
            "n_trainable": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
        if unlearning_time is not None:
            results["unlearning_time_s"] = unlearning_time
        return results

    def _eval_privacy(
        self, model: nn.Module, forget_loader: DataLoader, retain_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate privacy (simple MIA proxy)."""
        device = next(model.parameters()).device
        model.eval()

        # Compute mean loss on forget and retain
        def mean_loss(loader):
            total_loss, n = 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                    total_loss += loss.item()
                    n += len(labels)
            return total_loss / max(n, 1)

        forget_loss = mean_loss(forget_loader)
        retain_loss = mean_loss(retain_loader)

        # MIA proxy: ratio of losses (closer to 1.0 = better unlearning)
        ratio = forget_loss / max(retain_loss, 1e-8)

        return {
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "loss_ratio": min(ratio, 10.0),
            "mia_proxy": 1.0 - abs(1.0 - min(ratio, 2.0)),
        }

    def _eval_scalability(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model scalability characteristics."""
        n_params = sum(p.numel() for p in model.parameters())
        n_layers = sum(1 for _ in model.named_modules())
        return {
            "n_parameters_M": n_params / 1e6,
            "n_layers": float(n_layers),
        }
