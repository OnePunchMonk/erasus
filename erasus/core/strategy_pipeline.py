"""
Strategy Pipeline — Chain multiple unlearning strategies sequentially.

Enables composing strategies so that the output model of one stage feeds
into the next (e.g., gradient ascent -> Fisher forgetting -> LoRA fine-tune).

Example
-------
>>> from erasus.core import StrategyPipeline
>>> pipeline = StrategyPipeline([
...     ("gradient_ascent", {"epochs": 3}),
...     ("fisher_forgetting", {"epochs": 2}),
... ])
>>> model, f_losses, r_losses = pipeline.unlearn(
...     model, forget_loader, retain_loader, epochs=5,
... )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


class StrategyPipeline(BaseStrategy):
    """
    Chain multiple unlearning strategies in sequence.

    Each stage receives the model output from the previous stage.
    Per-stage ``epochs`` can be overridden via the stage kwargs;
    otherwise the top-level ``epochs`` argument is used.

    Parameters
    ----------
    stages : list
        Each element is either:
        - A tuple ``(strategy_name_or_instance, kwargs_dict)``
        - A bare strategy name ``str`` (uses default kwargs)
        - A ``BaseStrategy`` instance (uses as-is)
    """

    def __init__(
        self,
        stages: Sequence[Union[str, BaseStrategy, Tuple[Union[str, BaseStrategy], Dict[str, Any]]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._stages: List[Tuple[BaseStrategy, Dict[str, Any]]] = []

        for entry in stages:
            if isinstance(entry, str):
                strategy = strategy_registry.get(entry)(**kwargs)
                stage_kwargs: Dict[str, Any] = {}
            elif isinstance(entry, BaseStrategy):
                strategy = entry
                stage_kwargs = {}
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                name_or_inst, stage_kwargs = entry[0], dict(entry[1])
                if isinstance(name_or_inst, str):
                    # Extract constructor kwargs vs runtime kwargs
                    ctor_keys = {"lr", "device", "alpha", "beta", "lambda_reg"}
                    ctor_kwargs = {k: v for k, v in stage_kwargs.items() if k in ctor_keys}
                    ctor_kwargs.update({k: v for k, v in kwargs.items() if k in ctor_keys and k not in ctor_kwargs})
                    strategy = strategy_registry.get(name_or_inst)(**ctor_kwargs)
                elif isinstance(name_or_inst, BaseStrategy):
                    strategy = name_or_inst
                else:
                    raise TypeError(
                        f"Stage strategy must be str or BaseStrategy, got {type(name_or_inst)}"
                    )
            else:
                raise TypeError(
                    f"Each pipeline stage must be str, BaseStrategy, or "
                    f"(name/instance, kwargs) tuple. Got: {type(entry)}"
                )
            self._stages.append((strategy, stage_kwargs))

    @property
    def stages(self) -> List[Tuple[BaseStrategy, Dict[str, Any]]]:
        """Return the list of (strategy, kwargs) stages."""
        return list(self._stages)

    def __len__(self) -> int:
        return len(self._stages)

    def __repr__(self) -> str:
        names = [s.__class__.__name__ for s, _ in self._stages]
        return f"StrategyPipeline([{' -> '.join(names)}])"

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Run each strategy stage sequentially.

        Per-stage ``epochs`` override the top-level value if set in the
        stage kwargs.

        Returns
        -------
        tuple
            (model, all_forget_losses, all_retain_losses) aggregated across stages.
        """
        all_forget_losses: List[float] = []
        all_retain_losses: List[float] = []

        for idx, (strategy, stage_kw) in enumerate(self._stages):
            stage_epochs = stage_kw.get("epochs", epochs)
            merged_kw = {**kwargs, **stage_kw}
            merged_kw.pop("epochs", None)

            model, f_losses, r_losses = strategy.unlearn(
                model=model,
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                epochs=stage_epochs,
                **merged_kw,
            )
            all_forget_losses.extend(f_losses)
            all_retain_losses.extend(r_losses)

        return model, all_forget_losses, all_retain_losses
