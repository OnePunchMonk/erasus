"""
erasus.utils.callbacks — Training callbacks for unlearning.

Provides extensible callbacks for monitoring and controlling
the unlearning process.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class Callback:
    """Base callback class."""

    def on_epoch_start(self, epoch: int, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        pass

    def on_batch_start(self, batch_idx: int, **kwargs: Any) -> None:
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs: Any) -> None:
        pass

    def on_train_start(self, **kwargs: Any) -> None:
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def should_stop(self) -> bool:
        return False


class CallbackList:
    """Manage multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_epoch_start(self, epoch: int, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, **kwargs)

    def on_batch_start(self, batch_idx: int, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_batch_start(batch_idx, **kwargs)

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch_idx, loss, **kwargs)

    def on_train_start(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_start(**kwargs)

    def on_train_end(self, **kwargs: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(**kwargs)

    def should_stop(self) -> bool:
        return any(cb.should_stop() for cb in self.callbacks)


class LoggingCallback(Callback):
    """Log metrics to console."""

    def __init__(self, log_every: int = 1) -> None:
        self.log_every = log_every

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        if (epoch + 1) % self.log_every == 0:
            parts = [f"{k}={v:.4f}" for k, v in sorted(metrics.items())]
            print(f"[Epoch {epoch + 1}] {', '.join(parts)}")


class TimingCallback(Callback):
    """Track time per epoch."""

    def __init__(self) -> None:
        self.epoch_times: List[float] = []
        self._start: float = 0.0

    def on_epoch_start(self, epoch: int, **kwargs: Any) -> None:
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        self.epoch_times.append(time.perf_counter() - self._start)

    @property
    def total_time(self) -> float:
        return sum(self.epoch_times)

    @property
    def avg_epoch_time(self) -> float:
        return self.total_time / max(len(self.epoch_times), 1)


class LossHistoryCallback(Callback):
    """Record loss history."""

    def __init__(self) -> None:
        self.batch_losses: List[float] = []
        self.epoch_losses: List[float] = []
        self._epoch_batch_losses: List[float] = []

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs: Any) -> None:
        self.batch_losses.append(loss)
        self._epoch_batch_losses.append(loss)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        avg = sum(self._epoch_batch_losses) / max(len(self._epoch_batch_losses), 1)
        self.epoch_losses.append(avg)
        self._epoch_batch_losses = []


class CheckpointCallback(Callback):
    """Save model checkpoint at intervals."""

    def __init__(
        self,
        save_fn: Any,
        save_every: int = 5,
        save_best: bool = True,
        monitor: str = "loss",
    ) -> None:
        self.save_fn = save_fn
        self.save_every = save_every
        self.save_best = save_best
        self.monitor = monitor
        self.best_value: Optional[float] = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs: Any) -> None:
        # Periodic save
        if (epoch + 1) % self.save_every == 0:
            self.save_fn(epoch, metrics)

        # Best model save
        if self.save_best and self.monitor in metrics:
            val = metrics[self.monitor]
            if self.best_value is None or val < self.best_value:
                self.best_value = val
                self.save_fn(epoch, metrics, is_best=True)
