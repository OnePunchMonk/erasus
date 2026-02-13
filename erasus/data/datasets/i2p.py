"""
erasus.data.datasets.i2p — I2P (Inappropriate Image Prompts) dataset.

Wraps the I2P benchmark for evaluating NSFW concept removal in
Stable Diffusion and other text-to-image models.

Reference: Schramowski et al. (2023)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from torch.utils.data import Dataset


class I2PDataset(Dataset):
    """
    I2P (Inappropriate Image Prompts) dataset.

    Contains text prompts that may generate inappropriate images.
    Used mainly for evaluating diffusion model unlearning (NSFW removal).

    Parameters
    ----------
    data_dir : str
        Path to I2P data files.
    categories : list of str, optional
        Filter to specific categories: ``"sexual"``, ``"violence"``,
        ``"disturbing"``, ``"hateful"``, etc.
    """

    def __init__(
        self,
        data_dir: str = "./data/i2p",
        categories: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.categories = categories
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> List[dict]:
        import json

        data_file = self.data_dir / "i2p_prompts.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        else:
            # Try CSV
            csv_file = self.data_dir / "i2p_prompts.csv"
            if csv_file.exists():
                import csv
                prompts = []
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prompts.append(row)
            else:
                # Try HuggingFace
                try:
                    from datasets import load_dataset as hf_load
                    ds = hf_load("AIML-TUDA/i2p", split="train")
                    prompts = [{"prompt": s["prompt"], "categories": s.get("categories", [])} for s in ds]
                except Exception:
                    print(f"  ⚠ I2P data not found in {self.data_dir}.")
                    return []

        # Filter by category if specified
        if self.categories:
            filtered = []
            for p in prompts:
                cats = p.get("categories", [])
                if isinstance(cats, str):
                    cats = [cats]
                if any(c in cats for c in self.categories):
                    filtered.append(p)
            return filtered

        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return self.prompts[idx]

    def get_prompt_texts(self) -> List[str]:
        """Return all prompt strings."""
        return [p.get("prompt", "") for p in self.prompts]
