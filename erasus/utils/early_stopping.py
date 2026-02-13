"""
erasus.utils.early_stopping — Early stopping for unlearning.

Monitors a metric during unlearning and stops training when
the metric stops improving, preventing over-unlearning.
"""

from __future__ import annotations

from typing import Optional


class EarlyStopping:
    """
    Early stopping monitor.

    Stops training when a monitored metric hasn't improved
    for ``patience`` epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``"min"`` (loss-like) or ``"max"`` (accuracy-like).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter: int = 0
        self.best: Optional[float] = None
        self.stopped: bool = False
        self.best_epoch: int = 0

    def __call__(self, epoch: int, value: float) -> bool:
        """
        Check if training should stop.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best is None:
            self.best = value
            self.best_epoch = epoch
            return False

        improved = False
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        elif self.mode == "max":
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            return True

        return False

    def reset(self) -> None:
        """Reset the monitor."""
        self.counter = 0
        self.best = None
        self.stopped = False
        self.best_epoch = 0
