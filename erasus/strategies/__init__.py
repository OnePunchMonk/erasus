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

# Parameter methods
from erasus.strategies.parameter_methods.lora_unlearning import LoRAUnlearningStrategy
from erasus.strategies.parameter_methods.sparse_aware import SparseAwareUnlearningStrategy
from erasus.strategies.parameter_methods.mask_based import MaskBasedUnlearningStrategy
from erasus.strategies.parameter_methods.neuron_pruning import NeuronPruningStrategy

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
    "SaliencyUnlearningStrategy",
    "LoRAUnlearningStrategy",
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
    "ConceptErasureStrategy",
    "NoiseInjectionStrategy",
    "UNetSurgeryStrategy",
    "TimestepMaskingStrategy",
    "SafeLatentsStrategy",
    "ContrastiveUnlearningStrategy",
    "AttentionUnlearningStrategy",
    "VisionTextSplitStrategy",
    "EnsembleStrategy",
]
