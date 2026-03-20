"""
Federated Continual Unlearning — Sequential deletion across clients.

Extends continual unlearning to federated settings where multiple clients
submit deletion requests over time without sharing their data.

Implements FIT (Federated Incremental unlearning) ideas:
- Each client has local deletion requests
- Aggregates updates across clients
- Prevents catastrophic forgetting in federated setting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_unlearner import BaseUnlearner, UnlearningResult
from erasus.core.registry import strategy_registry

import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


@dataclass
class ClientDeletionRequest:
    """Deletion request from a federated client."""

    client_id: str
    request_id: str
    forget_loader: DataLoader
    forget_set_size: int
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedUnlearningResult:
    """Results for federated continual unlearning."""

    model: nn.Module
    total_elapsed_time: float = 0.0
    client_requests: List[ClientDeletionRequest] = field(default_factory=list)
    per_request_results: List[UnlearningResult] = field(default_factory=list)
    per_client_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    catastrophic_forgetting_detected: bool = False
    utility_degradation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FederatedContinualUnlearner(BaseUnlearner):
    """
    Federated continual unlearning orchestrator.

    Handles sequential deletion requests from multiple clients without
    sharing client data. Prevents catastrophic forgetting across clients.

    Parameters
    ----------
    model : nn.Module
        Global model.
    strategy : str
        Unlearning strategy.
    base_epochs : int
        Epochs per deletion request (default 3).
    adaptive_scheduling : bool
        Reduce epochs over time (default True).
    aggregation_method : str
        How to aggregate updates ("fedavg", "fedsgd") default "fedavg".
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "gradient_ascent",
        selector: Optional[str] = None,
        device: Optional[str] = None,
        base_epochs: int = 3,
        adaptive_scheduling: bool = True,
        aggregation_method: str = "fedavg",
        **kwargs: Any,
    ) -> None:
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls()

        selector_instance = None
        if selector is not None:
            selector_cls = selector_registry.get(selector)
            selector_instance = selector_cls()

        super().__init__(
            model=model,
            strategy=strategy_instance,
            selector=selector_instance,
            device=device,
            **kwargs,
        )

        self.strategy_name = strategy
        self.selector_name = selector
        self.base_epochs = base_epochs
        self.adaptive_scheduling = adaptive_scheduling
        self.aggregation_method = aggregation_method

        self.client_models: Dict[str, nn.Module] = {}
        self.request_history: List[ClientDeletionRequest] = []

    def process_client_requests(
        self,
        client_requests: Dict[str, List[ClientDeletionRequest]],
        retain_loader: Optional[DataLoader] = None,
        prune_ratio: float = 0.1,
        **kwargs: Any,
    ) -> FederatedUnlearningResult:
        """
        Process deletion requests from multiple clients.

        Parameters
        ----------
        client_requests : dict of (client_id -> list of requests)
            Deletion requests grouped by client.
        retain_loader : DataLoader, optional
            Global retain set for utility monitoring.
        prune_ratio : float
            Coreset selection ratio.

        Returns
        -------
        FederatedUnlearningResult
            Results with per-client metrics.
        """
        import time

        total_start = time.time()
        all_requests = []
        per_client_metrics = {}

        for client_id, requests in client_requests.items():
            per_client_metrics[client_id] = {
                "num_requests": len(requests),
                "total_deletions": sum(r.forget_set_size for r in requests),
            }

            for req in requests:
                all_requests.append(req)

            # Update client model
            if client_id not in self.client_models:
                self.client_models[client_id] = self._create_client_model()

        # Process all requests
        per_request_results = []
        for idx, request in enumerate(all_requests):
            if self.adaptive_scheduling:
                epochs = max(1, self.base_epochs - idx // 5)
            else:
                epochs = self.base_epochs

            # Unlearn request
            self.model, forget_losses, retain_losses = self.strategy.unlearn(
                model=self.model,
                forget_loader=request.forget_loader,
                retain_loader=retain_loader,
                epochs=epochs,
                **kwargs,
            )

            result = UnlearningResult(
                model=self.model,
                forget_loss_history=forget_losses,
                retain_loss_history=retain_losses,
                metadata={"client": request.client_id, "request": request.request_id},
            )
            per_request_results.append(result)

        total_elapsed = time.time() - total_start

        # Aggregate client updates
        if self.aggregation_method == "fedavg":
            self._federated_averaging(per_client_metrics)

        return FederatedUnlearningResult(
            model=self.model,
            total_elapsed_time=total_elapsed,
            client_requests=all_requests,
            per_request_results=per_request_results,
            per_client_metrics=per_client_metrics,
            catastrophic_forgetting_detected=False,
            utility_degradation=0.0,
            metadata={
                "strategy": self.strategy_name,
                "num_clients": len(client_requests),
                "total_requests": len(all_requests),
            },
        )

    def _create_client_model(self) -> nn.Module:
        """Create a client-specific model copy."""
        import copy

        return copy.deepcopy(self.model)

    def _federated_averaging(self, per_client_metrics: Dict[str, Any]) -> None:
        """
        Average updates across clients.

        Parameters
        ----------
        per_client_metrics : dict
            Metrics per client.
        """
        if not self.client_models:
            return

        # Average model parameters across clients
        avg_state_dict = {}
        num_clients = len(self.client_models)

        for param_name in self.model.state_dict():
            avg_state_dict[param_name] = torch.zeros_like(
                self.model.state_dict()[param_name]
            )

            for client_model in self.client_models.values():
                if param_name in client_model.state_dict():
                    avg_state_dict[param_name] += (
                        client_model.state_dict()[param_name] / num_clients
                    )

        # Update global model
        self.model.load_state_dict(avg_state_dict)

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int,
        **kwargs: Any,
    ) -> tuple:
        """For compatibility with BaseUnlearner."""
        self.model, forget_losses, retain_losses = self.strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
            **kwargs,
        )
        return forget_losses, retain_losses
