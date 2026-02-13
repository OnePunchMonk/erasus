"""
erasus.experiments.hyperparameter_search — HPO for unlearning.

Integrates with Optuna for hyperparameter optimization of
unlearning configurations (strategy params, learning rate,
selector budget, etc.).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class HyperparameterSearch:
    """
    Hyperparameter search for unlearning.

    Uses Optuna if available, otherwise falls back to random search.

    Parameters
    ----------
    objective_fn : callable
        Function ``(trial_params) -> float`` that returns the score to minimise.
    param_space : dict
        Parameter search space definition.
    n_trials : int
        Number of trials.
    direction : str
        ``"minimize"`` or ``"maximize"``.
    """

    def __init__(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Any],
        n_trials: int = 20,
        direction: str = "minimize",
    ) -> None:
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.direction = direction
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _try_optuna(self) -> bool:
        """Check if optuna is available."""
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self) -> Dict[str, Any]:
        """Run hyperparameter search."""
        if self._try_optuna():
            return self._run_optuna()
        return self._run_random()

    def _run_optuna(self) -> Dict[str, Any]:
        """Run search with Optuna."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, spec in self.param_space.items():
                if spec["type"] == "float":
                    params[name] = trial.suggest_float(
                        name, spec["low"], spec["high"],
                        log=spec.get("log", False),
                    )
                elif spec["type"] == "int":
                    params[name] = trial.suggest_int(name, spec["low"], spec["high"])
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])

            score = self.objective_fn(params)
            self.results.append({"params": params, "score": score})
            return score

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_score = study.best_value

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(study.trials),
            "all_results": self.results,
        }

    def _run_random(self) -> Dict[str, Any]:
        """Fallback: random search."""
        import random

        for i in range(self.n_trials):
            params = {}
            for name, spec in self.param_space.items():
                if spec["type"] == "float":
                    if spec.get("log", False):
                        import math
                        log_low = math.log(spec["low"])
                        log_high = math.log(spec["high"])
                        params[name] = math.exp(random.uniform(log_low, log_high))
                    else:
                        params[name] = random.uniform(spec["low"], spec["high"])
                elif spec["type"] == "int":
                    params[name] = random.randint(spec["low"], spec["high"])
                elif spec["type"] == "categorical":
                    params[name] = random.choice(spec["choices"])

            score = self.objective_fn(params)
            self.results.append({"params": params, "score": score})

            # Track best
            is_better = (
                self.best_score is None
                or (self.direction == "minimize" and score < self.best_score)
                or (self.direction == "maximize" and score > self.best_score)
            )
            if is_better:
                self.best_params = params
                self.best_score = score

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "all_results": self.results,
        }
