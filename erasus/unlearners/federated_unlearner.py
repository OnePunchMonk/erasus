"""
erasus.unlearners.federated_unlearner — Federated unlearning orchestrator.

Coordinates unlearning across multiple clients in a federated learning
setting, supporting FedAvg-style aggregation with selective forgetting.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_unlearner import BaseUnlearner, UnlearningResult
from erasus.core.registry import strategy_registry, selector_registry

import erasus.strategies  # noqa: F401
import erasus.selectors  # noqa: F401


class FederatedUnlearner(BaseUnlearner):
    """
    Federated unlearning orchestrator.

    Manages unlearning across multiple federated clients. Supports:

    - **Client-level forgetting**: Remove all data from a specific client.
    - **Sample-level forgetting**: Remove specific samples across clients.
    - **Concept-level forgetting**: Remove a concept distributed across clients.

    Parameters
    ----------
    model : nn.Module
        The global model.
    strategy : str
        Unlearning strategy name.
    n_clients : int
        Number of federated clients.
    aggregation : str
        Aggregation method (``"fedavg"`` or ``"weighted"``).
    communication_rounds : int
        Number of federated communication rounds.
    """

    DEFAULT_STRATEGY = "gradient_ascent"

    def __init__(
        self,
        model: nn.Module,
        strategy: str = DEFAULT_STRATEGY,
        n_clients: int = 5,
        aggregation: str = "fedavg",
        communication_rounds: int = 3,
        local_epochs: int = 3,
        strategy_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        strategy_cls = strategy_registry.get(strategy)
        strategy_instance = strategy_cls(**(strategy_kwargs or {}))

        super().__init__(
            model=model,
            strategy=strategy_instance,
            selector=None,
            device=device,
            **kwargs,
        )
        self.strategy_name = strategy
        self.n_clients = n_clients
        self.aggregation = aggregation
        self.communication_rounds = communication_rounds
        self.local_epochs = local_epochs
        self._client_models: List[nn.Module] = []

    # ------------------------------------------------------------------
    # Core federated unlearning
    # ------------------------------------------------------------------

    def _run_unlearning(
        self,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader],
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[List[float], List[float]]:
        """
        Run federated unlearning.

        ``forget_loader`` and ``retain_loader`` can be:
        - Single loaders (will be split across clients)
        - kwargs may contain ``client_forget_loaders`` and
          ``client_retain_loaders`` for pre-split data.
        """
        client_forget_loaders = kwargs.get("client_forget_loaders", None)
        client_retain_loaders = kwargs.get("client_retain_loaders", None)
        forget_client_ids = kwargs.get("forget_client_ids", None)

        # If no per-client data, simulate splitting
        if client_forget_loaders is None:
            client_forget_loaders = [forget_loader] * self.n_clients
        if client_retain_loaders is None:
            client_retain_loaders = [retain_loader] * self.n_clients

        all_forget_losses: List[float] = []
        all_retain_losses: List[float] = []

        # Initialise client models
        self._client_models = [
            copy.deepcopy(self.model) for _ in range(self.n_clients)
        ]

        for comm_round in range(self.communication_rounds):
            round_forget_losses: List[float] = []
            round_retain_losses: List[float] = []

            for client_id in range(self.n_clients):
                client_model = self._client_models[client_id]

                if forget_client_ids and client_id not in forget_client_ids:
                    # This client has no data to forget — just retain fine-tune
                    if client_retain_loaders[client_id] is not None:
                        _, _, retain_losses = self.strategy.unlearn(
                            model=client_model,
                            forget_loader=client_retain_loaders[client_id],
                            retain_loader=client_retain_loaders[client_id],
                            epochs=self.local_epochs,
                        )
                        round_retain_losses.extend(retain_losses)
                else:
                    # This client performs unlearning
                    client_model, forget_losses, retain_losses = self.strategy.unlearn(
                        model=client_model,
                        forget_loader=client_forget_loaders[client_id],
                        retain_loader=client_retain_loaders[client_id],
                        epochs=self.local_epochs,
                    )
                    self._client_models[client_id] = client_model
                    round_forget_losses.extend(forget_losses)
                    round_retain_losses.extend(retain_losses)

            # Aggregate
            self.model = self._aggregate(self._client_models)

            # Broadcast global model back to clients
            for client_id in range(self.n_clients):
                self._client_models[client_id].load_state_dict(
                    self.model.state_dict()
                )

            if round_forget_losses:
                all_forget_losses.append(sum(round_forget_losses) / len(round_forget_losses))
            if round_retain_losses:
                all_retain_losses.append(sum(round_retain_losses) / len(round_retain_losses))

        return all_forget_losses, all_retain_losses

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, client_models: List[nn.Module]) -> nn.Module:
        """Aggregate client models."""
        if self.aggregation == "fedavg":
            return self._fedavg(client_models)
        elif self.aggregation == "weighted":
            return self._weighted_avg(client_models)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _fedavg(self, client_models: List[nn.Module]) -> nn.Module:
        """Standard Federated Averaging."""
        global_model = copy.deepcopy(client_models[0])
        global_state = global_model.state_dict()

        for key in global_state:
            stacked = torch.stack([
                cm.state_dict()[key].float() for cm in client_models
            ])
            global_state[key] = stacked.mean(dim=0).to(global_state[key].dtype)

        global_model.load_state_dict(global_state)
        return global_model

    def _weighted_avg(
        self, client_models: List[nn.Module], weights: Optional[List[float]] = None
    ) -> nn.Module:
        """Weighted aggregation."""
        if weights is None:
            weights = [1.0 / len(client_models)] * len(client_models)

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        global_model = copy.deepcopy(client_models[0])
        global_state = global_model.state_dict()

        for key in global_state:
            weighted_sum = sum(
                w * cm.state_dict()[key].float()
                for w, cm in zip(weights, client_models)
            )
            global_state[key] = weighted_sum.to(global_state[key].dtype)

        global_model.load_state_dict(global_state)
        return global_model

    # ------------------------------------------------------------------
    # Federated-specific utilities
    # ------------------------------------------------------------------

    def forget_client(
        self,
        client_id: int,
        client_data: DataLoader,
        retain_data: Optional[DataLoader] = None,
        epochs: int = 5,
    ) -> UnlearningResult:
        """
        Remove all contributions of a specific client.

        This is the simplest form of federated unlearning:
        exclude the client and re-aggregate.
        """
        remaining_models = [
            m for i, m in enumerate(self._client_models) if i != client_id
        ]

        if remaining_models:
            self.model = self._aggregate(remaining_models)
        else:
            # If only one client, fall back to standard unlearning
            return self.fit(
                forget_data=client_data,
                retain_data=retain_data,
                epochs=epochs,
            )

        return UnlearningResult(
            model=self.model,
            forget_loss_history=[],
            retain_loss_history=[],
            metadata={"removed_client": client_id},
        )

    def get_client_model(self, client_id: int) -> nn.Module:
        """Get a specific client's local model."""
        if 0 <= client_id < len(self._client_models):
            return self._client_models[client_id]
        raise IndexError(f"Client {client_id} not found (have {len(self._client_models)} clients)")
