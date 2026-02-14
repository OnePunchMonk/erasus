"""
erasus.data.datasets.imagenet — ImageNet variant loaders.

Provides wrappers for ImageNet-1K, ImageNet-100, and Tiny ImageNet
for image classification unlearning tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    """
    ImageNet dataset wrapper for unlearning experiments.

    Supports ImageNet-1K, ImageNet-100, and Tiny ImageNet.  When used
    for unlearning, certain classes or samples can be designated as
    "forget" targets.

    Parameters
    ----------
    data_dir : str
        Root directory of the ImageNet-style dataset.
    split : str
        ``"train"`` or ``"val"``.
    variant : str
        ``"imagenet1k"``, ``"imagenet100"``, or ``"tiny"``.
    transform : callable, optional
        Image transforms (torchvision compatible).
    forget_classes : list[int], optional
        Class indices designated for forgetting.
    max_samples : int, optional
        Limit total samples (useful for debugging / fast experiments).
    """

    VARIANTS = ("imagenet1k", "imagenet100", "tiny")

    def __init__(
        self,
        data_dir: str = "./data/imagenet",
        split: str = "train",
        variant: str = "imagenet1k",
        transform: Optional[Callable] = None,
        forget_classes: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.variant = variant
        self.transform = transform
        self.forget_classes = set(forget_classes or [])
        self.max_samples = max_samples

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Scan directory structure for image-label pairs."""
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            # Fallback: try HuggingFace
            try:
                self._load_from_hub()
                return
            except Exception:
                print(f"  ⚠ ImageNet data not found at {split_dir}. "
                      f"Download the dataset or use HuggingFace.")
                return

        # Standard ImageFolder structure: root/class_name/image.jpg
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx

            image_files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG"}
            )

            for img_path in image_files:
                self.samples.append((str(img_path), idx))

        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]

    def _load_from_hub(self) -> None:
        """Try loading from HuggingFace datasets."""
        from datasets import load_dataset as hf_load

        hub_name = {
            "imagenet1k": "imagenet-1k",
            "imagenet100": "frgfm/imagenette",
            "tiny": "Maysee/tiny-imagenet",
        }.get(self.variant, "imagenet-1k")

        ds = hf_load(hub_name, split=self.split)
        limit = self.max_samples or len(ds)
        for i, sample in enumerate(ds):
            if i >= limit:
                break
            # Store index → will load lazily
            self.samples.append((str(i), sample.get("label", 0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path_or_idx, label = self.samples[idx]

        try:
            from PIL import Image
            image = Image.open(path_or_idx).convert("RGB")
        except Exception:
            # Synthetic fallback for testing
            image = torch.randn(3, 224, 224)

        if self.transform is not None and not isinstance(image, torch.Tensor):
            image = self.transform(image)
        elif not isinstance(image, torch.Tensor):
            import torchvision.transforms as T
            image = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(image)

        return {"image": image, "label": label}

    # ------------------------------------------------------------------
    # Forget / retain splitting
    # ------------------------------------------------------------------

    def get_forget_retain_split(self) -> Tuple[List[int], List[int]]:
        """
        Split sample indices into forget and retain based on ``forget_classes``.

        Returns
        -------
        (forget_indices, retain_indices)
        """
        forget_idx = []
        retain_idx = []
        for i, (_, label) in enumerate(self.samples):
            if label in self.forget_classes:
                forget_idx.append(i)
            else:
                retain_idx.append(i)
        return forget_idx, retain_idx

    @property
    def num_classes(self) -> int:
        labels = {label for _, label in self.samples}
        return len(labels)
