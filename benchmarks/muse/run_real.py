"""
Real MUSE benchmark runner using the MUSE dataset and a causal LM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from erasus.data.datasets.muse import MUSEDataset

DEFAULT_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


def build_zephyr_components(
    model_name: str = DEFAULT_MODEL_NAME,
    device_map: str = "auto",
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    """Load a Zephyr model and tokenizer for real MUSE evaluation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for the real MUSE runner") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
    )
    return model, tokenizer


def build_muse_suite(
    data_dir: str = "./data/muse",
    subset: str = "news",
    tokenizer: Optional[Any] = None,
    max_length: int = 256,
    batch_size: int = 4,
) -> Dict[str, DataLoader]:
    """Build tokenized MUSE split loaders."""
    datasets = MUSEDataset.get_evaluation_suite(
        data_dir=data_dir,
        subset=subset,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False)
        for split, ds in datasets.items()
    }


def _mean_lm_loss(model: Any, loader: DataLoader) -> float:
    """Average causal-LM loss over one split."""
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, dict) or "input_ids" not in batch:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get("labels", input_ids).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
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


def evaluate_muse_six_way(
    model: Any,
    suite: Dict[str, DataLoader],
) -> Dict[str, Any]:
    """Compute a six-axis MUSE-style report on real splits."""
    forget_loss = _mean_lm_loss(model, suite["forget"])
    retain_loss = _mean_lm_loss(model, suite["retain"])
    holdout_loss = _mean_lm_loss(model, suite["holdout"])
    test_loss = _mean_lm_loss(model, suite["test"])

    privacy_leakage = max(0.0, retain_loss - forget_loss)
    knowledge_retention = 1.0 / (1.0 + retain_loss)
    generalization = 1.0 / (1.0 + holdout_loss)
    consistency = 1.0 / (1.0 + abs(holdout_loss - test_loss))
    forget_quality = forget_loss
    efficiency = 1.0 / (1.0 + test_loss)

    return {
        "benchmark": "muse_real",
        "six_way": {
            "forget_quality": float(forget_quality),
            "model_utility": float(knowledge_retention),
            "privacy_leakage": float(privacy_leakage),
            "knowledge_retention": float(generalization),
            "consistency": float(consistency),
            "efficiency": float(efficiency),
        },
        "raw": {
            "forget_loss": float(forget_loss),
            "retain_loss": float(retain_loss),
            "holdout_loss": float(holdout_loss),
            "test_loss": float(test_loss),
            "forget_perplexity": float(math.exp(min(forget_loss, 20.0))),
            "retain_perplexity": float(math.exp(min(retain_loss, 20.0))),
        },
    }


def run_real_muse(
    subset: str = "news",
    data_dir: str = "./data/muse",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 4,
    max_length: int = 256,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the real MUSE benchmark."""
    if model is None or tokenizer is None:
        model, tokenizer = build_zephyr_components(model_name=model_name)

    suite = build_muse_suite(
        data_dir=data_dir,
        subset=subset,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    results = evaluate_muse_six_way(model, suite)
    results["subset"] = subset
    results["model_name"] = getattr(model, "name_or_path", DEFAULT_MODEL_NAME)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real MUSE with Zephyr-7B")
    parser.add_argument("--subset", default="news")
    parser.add_argument("--data-dir", default="./data/muse")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_real_muse(
        subset=args.subset,
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
