"""
erasus.data.datasets.coco — COCO Captions dataset wrapper.

Provides a loader for COCO Captions used in VLM unlearning experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset wrapper.

    Loads image–caption pairs from COCO for VLM unlearning.

    Parameters
    ----------
    image_dir : str
        Path to COCO images (e.g., ``data/coco/train2017``).
    ann_file : str
        Path to annotations JSON (e.g., ``data/coco/annotations/captions_train2017.json``).
    transform : callable, optional
        Image transform (e.g., torchvision transforms).
    tokenizer : Any, optional
        Text tokenizer for captions.
    max_caption_length : int
        Maximum token length for captions.
    """

    def __init__(
        self,
        image_dir: str = "./data/coco/train2017",
        ann_file: str = "./data/coco/annotations/captions_train2017.json",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_caption_length: int = 77,
    ):
        self.image_dir = Path(image_dir)
        self.ann_file = Path(ann_file)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

        self.annotations = self._load_annotations()

    def _load_annotations(self) -> list:
        """Load COCO annotations."""
        import json

        if not self.ann_file.exists():
            print(f"  ⚠ COCO annotations not found: {self.ann_file}")
            return []

        with open(self.ann_file, "r") as f:
            data = json.load(f)

        # Build image_id → filename lookup
        id_to_file = {}
        for img in data.get("images", []):
            id_to_file[img["id"]] = img["file_name"]

        # Combine annotations with image paths
        samples = []
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in id_to_file:
                samples.append({
                    "image_path": str(self.image_dir / id_to_file[img_id]),
                    "caption": ann["caption"],
                    "image_id": img_id,
                })

        return samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        sample = self.annotations[idx]

        # Load image
        image = None
        try:
            from PIL import Image
            image = Image.open(sample["image_path"]).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception:
            # Fallback: return placeholder tensor
            image = torch.zeros(3, 224, 224)

        caption = sample["caption"]

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "pixel_values": image,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

        return {"image": image, "caption": caption}
