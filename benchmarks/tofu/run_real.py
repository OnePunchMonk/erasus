"""
Real TOFU benchmark runner using the locuslab/TOFU dataset and GPT-2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from erasus.data.datasets.tofu import TOFUDataset

DEFAULT_MODEL_NAME = "gpt2"


def build_gpt2_components(
    model_name: str = DEFAULT_MODEL_NAME,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Load GPT-2 and its tokenizer for real TOFU evaluation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for the real TOFU runner") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    return model, tokenizer


def build_tofu_loaders(
    data_dir: str = "./data/tofu",
    forget_split: str = "forget_01",
    retain_split: str = "retain",
    tokenizer: Optional[Any] = None,
    max_length: int = 256,
    batch_size: int = 4,
) -> Dict[str, DataLoader]:
    """Build tokenized TOFU forget/retain loaders."""
    forget = TOFUDataset(data_dir=data_dir, split=forget_split, tokenizer=tokenizer, max_length=max_length)
    retain = TOFUDataset(data_dir=data_dir, split=retain_split, tokenizer=tokenizer, max_length=max_length)
    return {
        "forget": DataLoader(forget, batch_size=batch_size, shuffle=False),
        "retain": DataLoader(retain, batch_size=batch_size, shuffle=False),
    }


def _mean_lm_loss(model: Any, loader: DataLoader) -> float:
    """Average causal-LM loss on a loader."""
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict):
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels", input_ids).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                losses.append(float(outputs.loss.item()))
                continue

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            losses.append(float(loss.item()))

    return float(sum(losses) / max(len(losses), 1))


def evaluate_real_tofu(
    model: Any,
    loaders: Dict[str, DataLoader],
) -> Dict[str, Any]:
    """Evaluate GPT-2-style utility and forgetting on real TOFU splits."""
    forget_loss = _mean_lm_loss(model, loaders["forget"])
    retain_loss = _mean_lm_loss(model, loaders["retain"])

    return {
        "benchmark": "tofu_real",
        "forget_loss": float(forget_loss),
        "retain_loss": float(retain_loss),
        "forget_perplexity": float(math.exp(min(forget_loss, 20.0))),
        "retain_perplexity": float(math.exp(min(retain_loss, 20.0))),
        "forget_effectiveness": float(forget_loss / max(retain_loss, 1e-8)),
    }


def run_real_tofu(
    data_dir: str = "./data/tofu",
    forget_split: str = "forget_01",
    retain_split: str = "retain",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 4,
    max_length: int = 256,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the real TOFU benchmark with GPT-2."""
    if model is None or tokenizer is None:
        model, tokenizer = build_gpt2_components(model_name=model_name)

    loaders = build_tofu_loaders(
        data_dir=data_dir,
        forget_split=forget_split,
        retain_split=retain_split,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    results = evaluate_real_tofu(model, loaders)
    results["model_name"] = getattr(model, "name_or_path", model_name)
    results["forget_split"] = forget_split
    results["retain_split"] = retain_split

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real TOFU with GPT-2")
    parser.add_argument("--data-dir", default="./data/tofu")
    parser.add_argument("--forget-split", default="forget_01")
    parser.add_argument("--retain-split", default="retain")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_real_tofu(
        data_dir=args.data_dir,
        forget_split=args.forget_split,
        retain_split=args.retain_split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
