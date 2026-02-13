"""
erasus.data.datasets.tofu — TOFU (Task of Fictitious Unlearning) benchmark.

Loads the TOFU dataset for evaluating LLM unlearning. TOFU contains
QA pairs about fictitious authors, enabling controlled evaluation
of knowledge removal.

Reference: Maini et al. (2024) — "TOFU: A Task of Fictitious Unlearning
for LLMs"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TOFUDataset(Dataset):
    """
    TOFU benchmark dataset wrapper.

    Loads QA pairs from the TOFU dataset for forget/retain splitting.

    Parameters
    ----------
    data_dir : str
        Path to TOFU data files (JSON/JSONL).
    split : str
        One of ``"forget_01"`` (1% forget), ``"forget_05"`` (5% forget),
        ``"forget_10"`` (10% forget), ``"retain"``.
    tokenizer : Any, optional
        HuggingFace tokenizer instance for tokenising QA pairs.
    max_length : int
        Maximum sequence length for tokenization.
    """

    SPLITS = ["forget_01", "forget_05", "forget_10", "retain", "world_facts", "real_authors"]

    def __init__(
        self,
        data_dir: str = "./data/tofu",
        split: str = "forget_01",
        tokenizer: Optional[Any] = None,
        max_length: int = 256,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = self._load_data()

    def _load_data(self):
        """Load samples from disk or HuggingFace hub."""
        import json

        # Try local files first
        data_file = self.data_dir / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try JSONL format
        data_file = self.data_dir / f"{self.split}.jsonl"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        # Try loading from HuggingFace datasets
        try:
            from datasets import load_dataset as hf_load

            ds = hf_load("locuslab/TOFU", split=self.split)
            return [{"question": s["question"], "answer": s["answer"]} for s in ds]
        except Exception:
            pass

        # Return empty if nothing found
        print(f"  ⚠ TOFU data not found in {self.data_dir}. Run download_datasets.py first.")
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        text = f"Question: {question}\nAnswer: {answer}"

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            # For causal LM, labels = input_ids
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        return {"text": text, "question": question, "answer": answer}

    @classmethod
    def get_forget_retain_split(
        cls, data_dir: str = "./data/tofu", forget_pct: str = "01", **kwargs
    ) -> Tuple["TOFUDataset", "TOFUDataset"]:
        """Convenience method to get matched forget/retain datasets."""
        forget = cls(data_dir, split=f"forget_{forget_pct}", **kwargs)
        retain = cls(data_dir, split="retain", **kwargs)
        return forget, retain
