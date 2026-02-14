"""erasus.models.vlm — Vision-Language Model wrappers."""
from erasus.models.vlm.clip import CLIPWrapper
from erasus.models.vlm.flamingo import FlamingoWrapper
from erasus.models.vlm.vision_transformer import ViTFeatureExtractor, compute_patch_importance

__all__ = [
    "CLIPWrapper",
    "FlamingoWrapper",
    "ViTFeatureExtractor",
    "compute_patch_importance",
]
