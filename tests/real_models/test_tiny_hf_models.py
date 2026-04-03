"""
Slow tests for tiny real-model integrations.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytestmark = pytest.mark.slow


def _make_token_loader(vocab_size: int = 64) -> DataLoader:
    tokens = torch.randint(0, vocab_size, (12, 8))
    return DataLoader(TensorDataset(tokens, torch.randint(0, vocab_size, (12, 8))), batch_size=4)


def test_tiny_gpt2_npo_runs():
    transformers = pytest.importorskip("transformers")
    from erasus.strategies.llm_specific.npo import NPOStrategy

    config = transformers.GPT2Config(vocab_size=64, n_positions=16, n_layer=1, n_head=2, n_embd=16)
    model = transformers.GPT2LMHeadModel(config)

    class Wrapper(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids):
            return self.lm(input_ids=input_ids).logits

    wrapped = Wrapper(model).train()
    loader = _make_token_loader()
    strategy = NPOStrategy(lr=1e-3)
    updated, forget_losses, _ = strategy.unlearn(wrapped, loader, epochs=1)
    assert updated is wrapped
    assert len(forget_losses) == 1


def test_tiny_clip_instantiation():
    transformers = pytest.importorskip("transformers")
    config = transformers.CLIPConfig(
        projection_dim=16,
        text_config_dict={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "max_position_embeddings": 16,
        },
        vision_config_dict={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "num_channels": 3,
        },
    )
    model = transformers.CLIPModel(config)

    pixel_values = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    assert outputs.logits_per_image.shape[0] == 2
