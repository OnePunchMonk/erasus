"""erasus.selectors — Coreset selection methods.

Importing this module eagerly registers all selectors with the
selector_registry so they can be resolved by name.
"""

# Top-level selectors
from erasus.selectors.random_selector import RandomSelector
from erasus.selectors.full_selector import FullSelector
from erasus.selectors.auto_selector import AutoSelector

# Gradient-based
from erasus.selectors.gradient_based.influence import InfluenceSelector
from erasus.selectors.gradient_based.tracin import TracInSelector
from erasus.selectors.gradient_based.gradient_norm import GradientNormSelector
from erasus.selectors.gradient_based.grad_match import GradMatchSelector
from erasus.selectors.gradient_based.el2n import EL2NSelector
from erasus.selectors.gradient_based.representer import RepresenterSelector
from erasus.selectors.gradient_based.forgetting_score import ForgettingScoreSelector

# Geometry-based
from erasus.selectors.geometry_based.kcenter import KCenterSelector
from erasus.selectors.geometry_based.herding import HerdingSelector
from erasus.selectors.geometry_based.glister import GlisterSelector
from erasus.selectors.geometry_based.submodular import SubmodularSelector
from erasus.selectors.geometry_based.kmeans_coreset import KMeansSelector

# Learning-based
from erasus.selectors.learning_based.forgetting_events import ForgettingEventsSelector
from erasus.selectors.learning_based.data_shapley import DataShapleySelector
from erasus.selectors.learning_based.valuation_network import ValuationNetworkSelector

# Ensemble
from erasus.selectors.ensemble.voting import VotingSelector

__all__ = [
    "RandomSelector",
    "FullSelector",
    "AutoSelector",
    "InfluenceSelector",
    "TracInSelector",
    "GradientNormSelector",
    "GradMatchSelector",
    "EL2NSelector",
    "RepresenterSelector",
    "ForgettingScoreSelector",
    "KCenterSelector",
    "HerdingSelector",
    "GlisterSelector",
    "SubmodularSelector",
    "KMeansSelector",
    "ForgettingEventsSelector",
    "DataShapleySelector",
    "ValuationNetworkSelector",
    "VotingSelector",
]
