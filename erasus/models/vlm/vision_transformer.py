"""
Vision Transformer Utilities — ViT helper functions for VLM unlearning.

Provides:
- ViT feature extraction at arbitrary layers
- Patch embedding manipulation
- Attention map extraction
- CLS token / mean pooling helpers

These utilities are used across CLIP, LLaVA, BLIP, and Flamingo wrappers
that rely on ViT-based vision encoders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ViTFeatureExtractor:
    """
    Hook-based feature extractor for Vision Transformer models.

    Attaches forward hooks to specified ViT layers and captures their
    outputs during a forward pass.

    Parameters
    ----------
    model : nn.Module
        A ViT-based model (e.g. ``CLIPVisionModel``, ``ViTModel``).
    layer_indices : list[int], optional
        Indices of transformer layers to hook.  ``None`` → all layers.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass and return captured features.

        Parameters
        ----------
        pixel_values : Tensor
            Input images of shape ``(B, C, H, W)``.
        return_attention : bool
            If ``True``, also capture attention weights.

        Returns
        -------
        dict[str, Tensor]
            Mapping from ``"layer_{i}"`` → hidden state tensor,
            and optionally ``"attn_{i}"`` → attention weight tensor.
        """
        self._features.clear()
        self._register_hooks(return_attention)

        try:
            with torch.no_grad():
                if hasattr(self.model, "forward"):
                    self.model(pixel_values)
                else:
                    raise RuntimeError("Model has no forward method.")
        finally:
            self._remove_hooks()

        return dict(self._features)

    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract the raw patch embeddings (before any transformer block).

        Returns shape ``(B, num_patches + 1, D)`` where +1 is the CLS token.
        """
        embeddings_module = self._find_embeddings_module()
        if embeddings_module is None:
            raise RuntimeError("Could not locate patch embedding layer.")

        with torch.no_grad():
            return embeddings_module(pixel_values)

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cls_pool(hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the CLS token (index 0) from hidden states."""
        return hidden_states[:, 0]

    @staticmethod
    def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
        """Mean-pool over the sequence dimension (excluding CLS)."""
        return hidden_states[:, 1:].mean(dim=1)

    @staticmethod
    def spatial_pool(
        hidden_states: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Reshape patch tokens into a spatial feature map.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(B, num_patches + 1, D)``.
        grid_size : (H, W), optional
            Spatial grid dimensions.  If ``None``, assumes square.

        Returns
        -------
        Tensor of shape ``(B, D, H, W)``
        """
        patches = hidden_states[:, 1:]  # remove CLS
        B, N, D = patches.shape
        if grid_size is None:
            h = w = int(N ** 0.5)
        else:
            h, w = grid_size
        return patches.reshape(B, h, w, D).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    # Attention map utilities
    # ------------------------------------------------------------------

    @staticmethod
    def rollout_attention(
        attention_maps: List[torch.Tensor],
        head_reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute attention rollout across layers.

        Parameters
        ----------
        attention_maps : list[Tensor]
            List of attention tensors ``(B, heads, seq, seq)`` per layer.
        head_reduction : str
            How to reduce across heads: ``"mean"`` | ``"max"``.

        Returns
        -------
        Tensor of shape ``(B, seq, seq)``
        """
        result = None
        for attn in attention_maps:
            if head_reduction == "mean":
                attn_reduced = attn.mean(dim=1)
            elif head_reduction == "max":
                attn_reduced = attn.max(dim=1).values
            else:
                raise ValueError(f"Unknown head_reduction: {head_reduction}")

            # Add residual connection (identity)
            eye = torch.eye(attn_reduced.size(-1), device=attn_reduced.device)
            attn_with_residual = 0.5 * attn_reduced + 0.5 * eye

            if result is None:
                result = attn_with_residual
            else:
                result = result @ attn_with_residual

        # Normalise rows
        if result is not None:
            result = result / result.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_encoder_layers(self) -> List[nn.Module]:
        """Locate the list of transformer encoder layers."""
        # Common attribute paths
        for attr_path in [
            "encoder.layers",
            "encoder.layer",
            "blocks",
            "layers",
            "transformer.resblocks",
        ]:
            obj = self.model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        return []

    def _find_embeddings_module(self) -> Optional[nn.Module]:
        """Locate the patch embedding module."""
        for attr in ["embeddings", "patch_embed", "conv_proj", "patch_embedding"]:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        return None

    def _register_hooks(self, return_attention: bool = False) -> None:
        layers = self._get_encoder_layers()
        indices = self.layer_indices if self.layer_indices else range(len(layers))

        for i in indices:
            if i >= len(layers):
                continue
            layer = layers[i]

            def _make_hook(idx: int):
                def hook(module, _input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self._features[f"layer_{idx}"] = out.detach()
                return hook

            self._hooks.append(layer.register_forward_hook(_make_hook(i)))

            if return_attention:
                # Try to hook the attention sub-module
                attn_module = None
                for name, mod in layer.named_modules():
                    if "self_attn" in name or "attention" in name.lower():
                        attn_module = mod
                        break

                if attn_module is not None:
                    def _attn_hook(idx: int):
                        def hook(module, _input, output):
                            if isinstance(output, tuple) and len(output) > 1:
                                self._features[f"attn_{idx}"] = output[1].detach()
                        return hook

                    self._hooks.append(attn_module.register_forward_hook(_attn_hook(i)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_patch_importance(
    hidden_states: torch.Tensor,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-patch importance scores.

    Parameters
    ----------
    hidden_states : Tensor
        Shape ``(B, num_patches + 1, D)``.
    reference : Tensor, optional
        Reference features for comparison.  If ``None``, uses the CLS
        token as reference.

    Returns
    -------
    Tensor of shape ``(B, num_patches)`` — cosine similarity to reference.
    """
    import torch.nn.functional as F

    cls_token = hidden_states[:, 0:1]  # (B, 1, D)
    patches = hidden_states[:, 1:]  # (B, N, D)

    ref = reference if reference is not None else cls_token
    if ref.dim() == 2:
        ref = ref.unsqueeze(1)

    sim = F.cosine_similarity(patches, ref.expand_as(patches), dim=-1)
    return sim
