"""
Multimodal Dataset Utilities.

Datasets for Image-Text pairs, etc.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageTextDataset(Dataset):
    """
    Simple generic dataset for Image-Text pairs.
    Expects a list of (image_path, text) tuples or a metadata file.
    """

    def __init__(
        self,
        data: List[Tuple[str, str]], # (image_path, caption)
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        root_dir: str = "",
    ) -> None:
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Any], Union[torch.Tensor, str]]:
        img_path, text = self.data[index]
        full_path = os.path.join(self.root_dir, img_path)
        
        try:
            image = Image.open(full_path).convert("RGB")
        except:
            # Placeholder for missing images
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        if self.tokenizer:
            # Tokenizer might return dict
            text = self.tokenizer(text)
            
        return image, text
