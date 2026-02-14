"""
T5 Model Wrapper — Encoder-decoder model for sequence-to-sequence unlearning.

Supports:
- Conditional generation (text-to-text)
- Encoder/decoder layer activation extraction
- Separate encoder/decoder gradient control
- Compatible with T5, Flan-T5, mT5 variants

Reference: Raffel et al. (2020) — "Exploring the Limits of Transfer Learning
with a Unified Text-to-Text Transformer"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from erasus.core.registry import model_registry
from erasus.models.model_wrapper import BaseLLMModel


@model_registry.register("t5")
class T5Wrapper(BaseLLMModel):
    """
    T5 / Flan-T5 encoder-decoder wrapper.

    Features
    --------
    - Encoder and decoder can be accessed independently
    - Layer-level activation extraction for both encoder and decoder
    - Gradient isolation between encoder and decoder sub-networks
    - Support for text-to-text generation

    Supported models
    ----------------
    - ``google/t5-small`` / ``t5-base`` / ``t5-large`` / ``t5-3b`` / ``t5-11b``
    - ``google/flan-t5-base`` / ``flan-t5-large`` / ``flan-t5-xl``
    - ``google/mt5-base`` (multilingual)
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "auto",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, device, **kwargs)
        self.tokenizer = None
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        device = self._device if self._device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(device)

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate text tokens given encoder input_ids.

        Parameters
        ----------
        input_ids : Tensor
            Encoder input token IDs of shape ``(B, seq_len)``.
        **kwargs
            Additional generation arguments (``max_new_tokens``, ``num_beams``, etc.).

        Returns
        -------
        Tensor
            Generated token IDs.
        """
        defaults = dict(max_new_tokens=128, num_beams=1)
        defaults.update(kwargs)
        return self.model.generate(input_ids.to(self.device), **defaults)

    def generate_text(self, prompts: Any, **kwargs) -> List[str]:
        """
        Generate text from string prompts (convenience wrapper).

        Parameters
        ----------
        prompts : str | list[str]
            Input texts.

        Returns
        -------
        list[str]
            Decoded output strings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        encoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        input_ids = encoding["input_ids"].to(self.device)

        output_ids = self.generate(input_ids, **kwargs)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Layer activations
    # ------------------------------------------------------------------

    def get_layer_activations(
        self,
        text: str,
        layer_indices: List[int],
        component: str = "encoder",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract hidden state activations from specified layers.

        Parameters
        ----------
        text : str
            Input text.
        layer_indices : list[int]
            Which transformer layer indices to capture.
        component : str
            ``"encoder"`` or ``"decoder"`` — which sub-network to look at.

        Returns
        -------
        dict[str, Tensor]
            Mapping ``"layer_{i}"`` → activation tensor.
        """
        activations: Dict[str, torch.Tensor] = {}
        hooks: list = []

        if component == "encoder":
            blocks = self.model.encoder.block
        elif component == "decoder":
            blocks = self.model.decoder.block
        else:
            raise ValueError(f"component must be 'encoder' or 'decoder', got '{component}'")

        def hook_fn(name: str):
            def hook(module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                activations[name] = out.detach()
            return hook

        for idx in layer_indices:
            if idx < len(blocks):
                hooks.append(blocks[idx].register_forward_hook(hook_fn(f"layer_{idx}")))

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # T5 needs decoder_input_ids for a full forward pass
        decoder_start = self.model.config.decoder_start_token_id or 0
        decoder_input_ids = torch.full(
            (inputs["input_ids"].size(0), 1), decoder_start, dtype=torch.long, device=self.device,
        )

        with torch.no_grad():
            self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=decoder_input_ids,
            )

        for h in hooks:
            h.remove()

        return activations

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze only the encoder parameters."""
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def freeze_decoder(self) -> None:
        """Freeze only the decoder parameters."""
        for p in self.model.decoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze only the encoder parameters."""
        for p in self.model.encoder.parameters():
            p.requires_grad = True

    def unfreeze_decoder(self) -> None:
        """Unfreeze only the decoder parameters."""
        for p in self.model.decoder.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Full forward pass for training / unlearning.

        Parameters
        ----------
        input_ids : Tensor
            Encoder input IDs.
        labels : Tensor, optional
            Target decoder token IDs.  If provided, computes loss.

        Returns
        -------
        Model outputs (with loss if labels given).
        """
        inputs = {"input_ids": input_ids.to(self.device)}
        if labels is not None:
            inputs["labels"] = labels.to(self.device)
        inputs.update(kwargs)
        return self.model(**inputs)
