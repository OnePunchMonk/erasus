"""
erasus.data.datasets.muse — MUSE (Machine Unlearning Six-way Evaluation).

Loads the MUSE benchmark dataset for evaluating unlearning across
multiple dimensions: forgetting quality, model utility, privacy,
efficiency, and robustness.

Reference: Shi et al. (2024) — "MUSE: Machine Unlearning Six-way Evaluation"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class MUSEDataset(Dataset):
    """
    MUSE benchmark dataset wrapper.

    Supports multiple evaluation splits for comprehensive unlearning
    assessment across six dimensions.

    Parameters
    ----------
    data_dir : str
        Path to MUSE data files.
    split : str
        One of ``"forget"``, ``"retain"``, ``"holdout"``, ``"test"``.
    subset : str
        MUSE subset: ``"news"`` (BBC News) or ``"books"`` (Harry Potter).
    tokenizer : Any, optional
        HuggingFace tokenizer for text processing.
    max_length : int
        Maximum sequence length.
    """

    SPLITS = ["forget", "retain", "holdout", "test"]
    SUBSETS = ["news", "books"]

    def __init__(
        self,
        data_dir: str = "./data/muse",
        split: str = "forget",
        subset: str = "news",
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {self.SPLITS}")
        if subset not in self.SUBSETS:
            raise ValueError(f"Invalid subset '{subset}'. Choose from {self.SUBSETS}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.subset = subset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = self._load_data()

    def _load_data(self) -> List[Dict[str, str]]:
        """Load samples from disk or HuggingFace Hub."""
        import json

        # Try local files
        data_file = self.data_dir / self.subset / f"{self.split}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)

        data_file = self.data_dir / self.subset / f"{self.split}.jsonl"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        # Try flat structure
        data_file = self.data_dir / f"{self.subset}_{self.split}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try HuggingFace
        try:
            from datasets import load_dataset as hf_load

            ds = hf_load("muse-bench/MUSE", self.subset, split=self.split)
            return [dict(s) for s in ds]
        except Exception:
            pass

        print(f"  ⚠ MUSE data not found for subset='{self.subset}', split='{self.split}'.")
        return []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # MUSE typically has "text" and optionally "label"
        text = sample.get("text", sample.get("content", ""))

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
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }

        return {"text": text, **{k: v for k, v in sample.items() if k != "text"}}

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @classmethod
    def get_forget_retain_split(
        cls,
        data_dir: str = "./data/muse",
        subset: str = "news",
        **kwargs,
    ) -> Tuple["MUSEDataset", "MUSEDataset"]:
        """Get matched forget/retain datasets."""
        forget = cls(data_dir, split="forget", subset=subset, **kwargs)
        retain = cls(data_dir, split="retain", subset=subset, **kwargs)
        return forget, retain

    @classmethod
    def get_evaluation_suite(
        cls,
        data_dir: str = "./data/muse",
        subset: str = "news",
        **kwargs,
    ) -> Dict[str, "MUSEDataset"]:
        """Get all evaluation splits as a dict."""
        return {
            split: cls(data_dir, split=split, subset=subset, **kwargs)
            for split in cls.SPLITS
        }

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def get_texts(self) -> List[str]:
        """Return all raw texts."""
        return [s.get("text", s.get("content", "")) for s in self.samples]
