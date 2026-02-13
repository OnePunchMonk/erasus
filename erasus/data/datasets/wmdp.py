"""
erasus.data.datasets.wmdp — WMDP (Weapons of Mass Destruction Proxy) benchmark.

Wraps the WMDP benchmark for evaluating hazardous knowledge removal.

Reference: Li et al. (2024) — "The WMDP Benchmark"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class WMDPDataset(Dataset):
    """
    WMDP benchmark dataset wrapper.

    Parameters
    ----------
    data_dir : str
        Path to WMDP data files.
    subset : str
        One of ``"bio"`` (biosecurity), ``"cyber"`` (cybersecurity),
        ``"chem"`` (chemical).
    tokenizer : Any, optional
        HuggingFace tokenizer for tokenisation.
    max_length : int
        Maximum sequence length.
    """

    SUBSETS = ["bio", "cyber", "chem"]

    def __init__(
        self,
        data_dir: str = "./data/wmdp",
        subset: str = "bio",
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data()

    def _load_data(self):
        import json

        data_file = self.data_dir / f"wmdp_{self.subset}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)

        data_file = self.data_dir / f"wmdp_{self.subset}.jsonl"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        try:
            from datasets import load_dataset as hf_load
            ds = hf_load("cais/wmdp", f"wmdp-{self.subset}")
            return list(ds["test"])
        except Exception:
            pass

        print(f"  ⚠ WMDP data not found in {self.data_dir}.")
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # WMDP format: multiple-choice QA
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        answer = sample.get("answer", 0)

        if self.tokenizer is not None and choices:
            text = question + " " + " ".join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(choices)
            )
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(answer, dtype=torch.long),
            }

        return sample
