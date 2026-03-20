"""erasus.strategies — Unlearning algorithm implementations.

Importing this module eagerly registers all strategies with the
strategy_registry so they can be resolved by name.
"""

# Gradient methods
from erasus.strategies.gradient_methods.gradient_ascent import GradientAscentStrategy
from erasus.strategies.gradient_methods.modality_decoupling import ModalityDecouplingStrategy
from erasus.strategies.gradient_methods.scrub import SCRUBStrategy
from erasus.strategies.gradient_methods.fisher_forgetting import FisherForgettingStrategy
from erasus.strategies.gradient_methods.negative_gradient import NegativeGradientStrategy
from erasus.strategies.gradient_methods.weighted_gradient_ascent import WGAStrategy

# Parameter methods
from erasus.strategies.parameter_methods.lora_unlearning import LoRAUnlearningStrategy
from erasus.strategies.parameter_methods.sparse_aware import SparseAwareUnlearningStrategy
from erasus.strategies.parameter_methods.mask_based import MaskBasedUnlearningStrategy
from erasus.strategies.parameter_methods.neuron_pruning import NeuronPruningStrategy
from erasus.strategies.parameter_methods.lora_unlearning_efficient import (
    LoRAUnlearningStrategy as EfficientLoRAStrategy,
    LoRAComposition,
)

# Data methods
from erasus.strategies.data_methods.amnesiac import AmnesiacUnlearningStrategy
from erasus.strategies.data_methods.sisa import SISAStrategy
from erasus.strategies.data_methods.certified_removal import CertifiedRemovalStrategy

# LLM-specific
from erasus.strategies.llm_specific.ssd import SelectiveSynapticDampeningStrategy
from erasus.strategies.llm_specific.token_masking import TokenMaskingStrategy
from erasus.strategies.llm_specific.embedding_alignment import EmbeddingAlignmentStrategy
from erasus.strategies.llm_specific.causal_tracing import CausalTracingStrategy

# Diffusion-specific
from erasus.strategies.diffusion_specific.concept_erasure import ConceptErasureStrategy
from erasus.strategies.diffusion_specific.noise_injection import NoiseInjectionStrategy
from erasus.strategies.diffusion_specific.unet_surgery import UNetSurgeryStrategy
from erasus.strategies.diffusion_specific.meta_unlearning import MetaUnlearningStrategy

# VLM-specific
from erasus.strategies.vlm_specific.contrastive_unlearning import ContrastiveUnlearningStrategy
from erasus.strategies.vlm_specific.attention_unlearning import AttentionUnlearningStrategy
from erasus.strategies.vlm_specific.vision_text_split import VisionTextSplitStrategy

# New gradient methods
from erasus.strategies.gradient_methods.saliency_unlearning import SaliencyUnlearningStrategy

# New parameter methods
from erasus.strategies.parameter_methods.layer_freezing import LayerFreezingStrategy

# New data methods
from erasus.strategies.data_methods.knowledge_distillation import KnowledgeDistillationStrategy

# New LLM-specific
from erasus.strategies.llm_specific.attention_surgery import AttentionSurgeryStrategy
from erasus.strategies.llm_specific.npo import NPOStrategy
from erasus.strategies.llm_specific.simnpo import SimNPOStrategy
from erasus.strategies.llm_specific.altpo import AltPOStrategy
from erasus.strategies.llm_specific.flat import FLATStrategy
from erasus.strategies.llm_specific.rmu import RMUStrategy
from erasus.strategies.llm_specific.undial import UNDIALStrategy
from erasus.strategies.llm_specific.delta_unlearning import DeltaUnlearningStrategy

# Inference-time strategies
from erasus.strategies.inference_time.dexperts import DExpertsStrategy
from erasus.strategies.inference_time.activation_steering import ActivationSteeringStrategy

# New diffusion-specific
from erasus.strategies.diffusion_specific.timestep_masking import TimestepMaskingStrategy
from erasus.strategies.diffusion_specific.safe_latents import SafeLatentsStrategy

# Ensemble
from erasus.strategies.ensemble_strategy import EnsembleStrategy

__all__ = [
    "GradientAscentStrategy",
    "ModalityDecouplingStrategy",
    "SCRUBStrategy",
    "FisherForgettingStrategy",
    "NegativeGradientStrategy",
    "WGAStrategy",
    "SaliencyUnlearningStrategy",
    "LoRAUnlearningStrategy",
    "EfficientLoRAStrategy",
    "LoRAComposition",
    "SparseAwareUnlearningStrategy",
    "MaskBasedUnlearningStrategy",
    "NeuronPruningStrategy",
    "LayerFreezingStrategy",
    "AmnesiacUnlearningStrategy",
    "SISAStrategy",
    "CertifiedRemovalStrategy",
    "KnowledgeDistillationStrategy",
    "SelectiveSynapticDampeningStrategy",
    "TokenMaskingStrategy",
    "EmbeddingAlignmentStrategy",
    "CausalTracingStrategy",
    "AttentionSurgeryStrategy",
    "NPOStrategy",
    "SimNPOStrategy",
    "AltPOStrategy",
    "FLATStrategy",
    "RMUStrategy",
    "UNDIALStrategy",
    "DeltaUnlearningStrategy",
    "DExpertsStrategy",
    "ActivationSteeringStrategy",
    "ConceptErasureStrategy",
    "NoiseInjectionStrategy",
    "UNetSurgeryStrategy",
    "TimestepMaskingStrategy",
    "SafeLatentsStrategy",
    "MetaUnlearningStrategy",
    "ContrastiveUnlearningStrategy",
    "AttentionUnlearningStrategy",
    "VisionTextSplitStrategy",
    "EnsembleStrategy",
]
