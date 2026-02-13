"""
Data Transforms.

Standard transforms for Vision and Text processing pipelines in Erasus.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import torch
import torchvision.transforms as T


def get_vision_transforms(
    model_name: str = "clip",
    image_size: int = 224,
    train: bool = True
) -> Callable:
    """
    Get standard image transforms.
    """
    mean = (0.48145466, 0.4578275, 0.40821073) # CLIP/OpenAI mean
    std = (0.26862954, 0.26130258, 0.27577711)

    transforms_list = []
    
    if train:
        transforms_list.append(T.RandomResizedCrop(image_size))
        transforms_list.append(T.RandomHorizontalFlip())
    else:
        transforms_list.append(T.Resize(image_size))
        transforms_list.append(T.CenterCrop(image_size))
        
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms_list)


def get_text_transform(tokenizer: Any, max_length: int = 77) -> Callable:
    """
    Returns a callable that tokenizes text.
    """
    def transform(text: Union[str, List[str]]):
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    return transform
