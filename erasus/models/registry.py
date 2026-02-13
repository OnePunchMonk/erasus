"""
Model registry — maps string names to wrapper classes.
"""

from erasus.core.registry import model_registry

# Auto-register models when their modules are imported.
# Each model wrapper decorates itself with @model_registry.register(...)
