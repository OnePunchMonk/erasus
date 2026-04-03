"""
Delta-Unlearning — Logit offsets for black-box LLMs.

Paper: "Delta-Unlearning: Removing Information from Language Models"

Key idea: Train a small white-box proxy model to compute logit offsets.
Apply these offsets to black-box model outputs at inference time.
No access to target model parameters needed.

Enables unlearning on:
- OpenAI GPT-4, GPT-3.5 (via API)
- Google Bard, Claude (via API)
- Any black-box LLM service
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


class DeltaUnlearningProxy(nn.Module):
    """
    Small proxy model that computes logit offsets.

    Trained to predict the difference between black-box and forget distributions.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 50257) -> None:
        """
        Initialize proxy.

        Parameters
        ----------
        input_dim : int
            Input dimension (embedding size).
        hidden_dim : int
            Hidden layer size.
        num_classes : int
            Vocabulary size.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_classes, input_dim)
        self.feature_proj = nn.LazyLinear(hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logit offsets for token IDs or continuous features."""
        if inputs.dtype in (torch.int32, torch.int64, torch.long):
            x = self.embedding(inputs)
            if x.dim() > 2:
                x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
        else:
            x = inputs.float()
            if x.dim() > 2:
                x = x.mean(dim=1)
            x = torch.relu(self.feature_proj(x))
        offsets = self.fc2(x)
        return offsets


@strategy_registry.register("delta_unlearning")
class DeltaUnlearningStrategy(BaseStrategy):
    """
    Delta-Unlearning: black-box unlearning via proxy offsets.

    Train a small proxy model to predict logit offsets that, when applied
    to the black-box model, cause it to forget the target information.

    Parameters
    ----------
    proxy_hidden_dim : int
        Hidden dimension of proxy model (default 256).
    offset_weight : float
        Weight of offset loss (default 1.0).
    consistency_weight : float
        Weight of consistency with base model (default 0.5).
    lr : float
        Learning rate (default 1e-3).
    """

    def __init__(
        self,
        proxy_hidden_dim: int = 256,
        offset_weight: float = 1.0,
        consistency_weight: float = 0.5,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy_hidden_dim = proxy_hidden_dim
        self.offset_weight = offset_weight
        self.consistency_weight = consistency_weight
        self.lr = lr
        self.proxy = None

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Train proxy model for black-box unlearning.

        Parameters
        ----------
        model : nn.Module
            Black-box model (or proxy in testing).
        forget_loader : DataLoader
            Forget set.
        retain_loader : DataLoader, optional
            Retain set.
        epochs : int
            Training epochs.

        Returns
        -------
        tuple
            (wrapped_model, forget_losses, retain_losses)
        """
        device = next(model.parameters()).device
        first_batch, forget_batches = self._prepare_forget_batches(forget_loader)
        num_classes = self._infer_num_classes(model, first_batch, device)

        # Create proxy
        self.proxy = DeltaUnlearningProxy(
            hidden_dim=self.proxy_hidden_dim,
            num_classes=num_classes,
        )
        self.proxy.to(device)
        self.proxy.train()

        optimizer = torch.optim.Adam(self.proxy.parameters(), lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass ---
            for batch in forget_batches:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # Get base model logits
                with torch.no_grad():
                    base_out = model(inputs)
                    base_logits = (
                        base_out.logits if hasattr(base_out, "logits") else base_out
                    )

                # Get proxy offsets
                proxy_offsets = self.proxy(inputs)

                # Adjusted logits = base_logits + proxy_offsets
                adjusted_logits = base_logits + proxy_offsets

                # Loss: maximize entropy (forget the target)
                target_probs = torch.ones_like(adjusted_logits) / adjusted_logits.size(-1)
                loss = F.kl_div(
                    F.log_softmax(adjusted_logits, dim=-1),
                    target_probs,
                    reduction="batchmean",
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_forget += loss.item()
                n_forget += 1

            forget_losses.append(epoch_forget / max(n_forget, 1))

            # --- Retain pass ---
            if retain_loader is not None:
                for batch in retain_loader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad():
                        base_out = model(inputs)
                        base_logits = (
                            base_out.logits if hasattr(base_out, "logits") else base_out
                        )

                    # Proxy should output near-zero offsets on retain data
                    proxy_offsets = self.proxy(inputs)
                    consistency_loss = torch.mean(proxy_offsets**2)

                    optimizer.zero_grad()
                    consistency_loss.backward()
                    optimizer.step()

                    epoch_retain += consistency_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        # Wrap model with proxy
        wrapped_model = DeltaUnlearningWrapper(model, self.proxy)
        return wrapped_model, forget_losses, retain_losses

    @staticmethod
    def _prepare_forget_batches(
        forget_loader: DataLoader,
    ) -> Tuple[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]]:
        """Materialize forget batches so the first one can be reused for shape inference."""
        forget_batches = list(forget_loader)
        if not forget_batches:
            raise ValueError("forget_loader must contain at least one batch")
        first_batch = forget_batches[0]
        return first_batch, forget_batches

    @staticmethod
    def _infer_num_classes(
        model: nn.Module,
        batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> int:
        """Infer output dimension from the wrapped model on a forget batch."""
        inputs = batch[0].to(device)
        with torch.no_grad():
            base_out = model(inputs)
            base_logits = base_out.logits if hasattr(base_out, "logits") else base_out
        if base_logits.dim() == 1:
            return 1
        return int(base_logits.size(-1))


class DeltaUnlearningWrapper(nn.Module):
    """
    Wraps black-box model and applies proxy offsets.

    At inference, computes logits = black_box(x) + proxy_offsets(x)
    """

    def __init__(self, base_model: nn.Module, proxy: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.proxy = proxy

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply offsets to base model output."""
        base_out = self.base_model(*args, **kwargs)
        base_logits = base_out.logits if hasattr(base_out, "logits") else base_out

        with torch.no_grad():
            proxy_offsets = self.proxy(args[0])

        adjusted_logits = base_logits + proxy_offsets
        return adjusted_logits

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
