"""
erasus.data.datasets.conceptual_captions — CC3M/CC12M dataset wrapper.

Loads Conceptual Captions for VLM unlearning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset


class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions (CC3M / CC12M) dataset wrapper.

    Parameters
    ----------
    data_dir : str
        Path to downloaded CC data.
    variant : str
        ``"cc3m"`` or ``"cc12m"``.
    transform : callable, optional
        Image transform.
    tokenizer : Any, optional
        Text tokenizer.
    max_samples : int, optional
        Cap the number of samples loaded (for debugging).
    """

    def __init__(
        self,
        data_dir: str = "./data/conceptual_captions",
        variant: str = "cc3m",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.variant = variant
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_samples = max_samples

        self.samples = self._load_data()

    def _load_data(self):
        import json

        data_file = self.data_dir / f"{self.variant}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if self.max_samples:
                    data = data[:self.max_samples]
                return data

        # Try TSV format (standard CC distribution)
        tsv_file = self.data_dir / f"{self.variant}.tsv"
        if tsv_file.exists():
            import csv
            samples = []
            with open(tsv_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        samples.append({"caption": row[0], "url": row[1]})
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
            return samples

        # Try HuggingFace
        try:
            from datasets import load_dataset as hf_load
            hf_name = "conceptual_captions" if self.variant == "cc3m" else "conceptual_12m"
            ds = hf_load(hf_name, split="train")
            samples = [{"caption": s["caption"], "url": s.get("image_url", "")} for s in ds]
            if self.max_samples:
                samples = samples[:self.max_samples]
            return samples
        except Exception:
            pass

        print(f"  ⚠ Conceptual Captions data not found in {self.data_dir}.")
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        caption = sample.get("caption", "")

        # Try to load image
        image = torch.zeros(3, 224, 224)  # Placeholder
        image_path = self.data_dir / "images" / f"{idx}.jpg"
        if image_path.exists():
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                if self.transform:
                    image = self.transform(img)
            except Exception:
                pass

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption, max_length=77, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            return {
                "pixel_values": image,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

        return {"image": image, "caption": caption}
