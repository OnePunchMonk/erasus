"""
erasus.strategies.llm_specific — LLM-specific unlearning strategies.
"""

from erasus.strategies.llm_specific.ssd import SelectiveSynapticDampeningStrategy
from erasus.strategies.llm_specific.causal_tracing import CausalTracingStrategy
from erasus.strategies.llm_specific.embedding_alignment import EmbeddingAlignmentStrategy
from erasus.strategies.llm_specific.token_masking import TokenMaskingStrategy
from erasus.strategies.llm_specific.npo import NPOStrategy
from erasus.strategies.llm_specific.simnpo import SimNPOStrategy
from erasus.strategies.llm_specific.altpo import AltPOStrategy
from erasus.strategies.llm_specific.flat import FLATStrategy
from erasus.strategies.llm_specific.rmu import RMUStrategy
from erasus.strategies.llm_specific.undial import UNDIALStrategy
from erasus.strategies.llm_specific.delta_unlearning import (
    DeltaUnlearningStrategy,
    DeltaUnlearningWrapper,
)

__all__ = [
    "SelectiveSynapticDampeningStrategy",
    "CausalTracingStrategy",
    "EmbeddingAlignmentStrategy",
    "TokenMaskingStrategy",
    "NPOStrategy",
    "SimNPOStrategy",
    "AltPOStrategy",
    "FLATStrategy",
    "RMUStrategy",
    "UNDIALStrategy",
    "DeltaUnlearningStrategy",
    "DeltaUnlearningWrapper",
]
