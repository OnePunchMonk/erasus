"""
Activation Steering — Hidden state manipulation at inference time.

Manipulates internal activations during forward pass to steer model behavior
without modifying weights. Enables:
- Runtime concept suppression
- Steering model outputs toward desired directions
- No retraining required
- Fully reversible

Inspiration: https://arxiv.org/abs/2308.10307
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from erasus.core.base_strategy import BaseStrategy
from erasus.core.registry import strategy_registry


class ActivationSteeringHook:
    """Hook for manipulating activations during forward pass."""

    def __init__(
        self,
        layer: nn.Module,
        steering_vector: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> None:
        """
        Initialize steering hook.

        Parameters
        ----------
        layer : nn.Module
            Layer to hook.
        steering_vector : torch.Tensor, optional
            Direction to steer activations.
        alpha : float
            Steering strength (default 1.0).
        """
        self.layer = layer
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.hook = None

    def hook_fn(self, module: nn.Module, input: Any, output: Any) -> torch.Tensor:
        """Forward hook that steers activations."""
        if self.steering_vector is None:
            return output

        # Handle tuple outputs (residuals, etc.)
        if isinstance(output, tuple):
            output = output[0]

        # Steer activation
        steered = output + self.alpha * self.steering_vector
        return steered

    def register(self) -> None:
        """Register the hook."""
        self.hook = self.layer.register_forward_hook(self.hook_fn)

    def remove(self) -> None:
        """Remove the hook."""
        if self.hook is not None:
            self.hook.remove()


@strategy_registry.register("activation_steering")
class ActivationSteeringStrategy(BaseStrategy):
    """
    Unlearning via activation steering.

    Learns steering vectors that, when applied to hidden states during
    inference, cause the model to suppress forget-set outputs without
    modifying weights.

    Parameters
    ----------
    target_layer : str
        Which layer to steer ("middle", "early", "late") default "middle".
    steering_strength : float
        Magnitude of steering vector (default 1.0).
    lr : float
        Learning rate for vector optimization (default 1e-2).
    num_vectors : int
        Number of steering vectors to learn (default 1).
    """

    def __init__(
        self,
        target_layer: str = "middle",
        steering_strength: float = 1.0,
        lr: float = 1e-2,
        num_vectors: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_layer = target_layer
        self.steering_strength = steering_strength
        self.lr = lr
        self.num_vectors = num_vectors
        self.steering_vectors = None

    def unlearn(
        self,
        model: nn.Module,
        forget_loader: DataLoader,
        retain_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        **kwargs: Any,
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Learn steering vectors for unlearning.

        Parameters
        ----------
        model : nn.Module
            Model to unlearn.
        forget_loader : DataLoader
            Forget set.
        retain_loader : DataLoader, optional
            Retain set.
        epochs : int
            Training epochs.

        Returns
        -------
        tuple
            (steered_model, forget_losses, retain_losses)
        """
        device = next(model.parameters()).device

        # Get target layer
        target_module = self._get_target_layer(model)
        if target_module is None:
            raise ValueError(f"Could not find target layer: {self.target_layer}")

        # Get hidden dimension
        hidden_dim = self._get_hidden_dim(model, target_module)

        # Initialize steering vectors
        self.steering_vectors = nn.Parameter(
            torch.randn(self.num_vectors, hidden_dim, device=device)
        )
        optimizer = torch.optim.Adam([self.steering_vectors], lr=self.lr)

        forget_losses: List[float] = []
        retain_losses: List[float] = []

        for epoch in range(epochs):
            epoch_forget = 0.0
            epoch_retain = 0.0
            n_forget = 0
            n_retain = 0

            # --- Forget pass ---
            for batch in forget_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # Register steering hooks
                hooks = []
                for vec in self.steering_vectors:
                    hook = ActivationSteeringHook(
                        target_module,
                        steering_vector=vec,
                        alpha=self.steering_strength,
                    )
                    hook.register()
                    hooks.append(hook)

                # Forward with steering
                out = model(inputs)
                logits = out.logits if hasattr(out, "logits") else out

                # Loss: maximize entropy (suppress forget info)
                target_probs = torch.ones_like(logits) / logits.size(-1)
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits, dim=-1),
                    target_probs,
                    reduction="batchmean",
                )

                # Remove hooks
                for hook in hooks:
                    hook.remove()

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

                    # Without steering on retain data
                    out = model(inputs)
                    logits = out.logits if hasattr(out, "logits") else out

                    # Standard classification loss
                    retain_loss = torch.nn.functional.cross_entropy(logits, labels)

                    optimizer.zero_grad()
                    retain_loss.backward()
                    optimizer.step()

                    epoch_retain += retain_loss.item()
                    n_retain += 1

                retain_losses.append(epoch_retain / max(n_retain, 1))

        # Wrap model with steering
        wrapped_model = SteeringModel(model, target_module, self.steering_vectors)
        return wrapped_model, forget_losses, retain_losses

    def _get_target_layer(self, model: nn.Module) -> Optional[nn.Module]:
        """Get the target layer to steer."""
        layers = list(model.modules())

        if self.target_layer == "middle":
            return layers[len(layers) // 2]
        elif self.target_layer == "early":
            return layers[len(layers) // 4]
        elif self.target_layer == "late":
            return layers[3 * len(layers) // 4]

        return None

    def _get_hidden_dim(self, model: nn.Module, layer: nn.Module) -> int:
        """Get hidden dimension of layer."""
        if hasattr(layer, "out_features"):
            return layer.out_features
        elif hasattr(layer, "hidden_size"):
            return layer.hidden_size
        else:
            # Default
            return 768


class SteeringModel(nn.Module):
    """Model with activation steering applied."""

    def __init__(
        self,
        base_model: nn.Module,
        target_layer: nn.Module,
        steering_vectors: nn.Parameter,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.target_layer = target_layer
        self.steering_vectors = steering_vectors
        self.hooks = []

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward with steering applied."""
        # Register hooks
        for vec in self.steering_vectors:
            hook = ActivationSteeringHook(
                self.target_layer,
                steering_vector=vec,
                alpha=1.0,
            )
            hook.register()
            self.hooks.append(hook)

        # Forward
        out = self.base_model(*args, **kwargs)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        return out

    def __getattr__(self, name: str) -> Any:
        """Proxy to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
