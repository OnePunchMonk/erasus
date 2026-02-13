"""
Test Suite for Erasus Imports.

Verifies that all modules can be imported without errors.
"""

import unittest
import importlib

class TestImports(unittest.TestCase):
    
    def test_core_imports(self):
        import erasus.core
        import erasus.core.base_unlearner
        import erasus.core.base_strategy
        import erasus.core.base_selector
        import erasus.core.registry
        import erasus.core.config
    
    def test_model_imports(self):
        import erasus.models
        import erasus.models.model_wrapper
        import erasus.models.vlm.clip
        import erasus.models.llm.llama
        import erasus.models.diffusion.stable_diffusion
        
    def test_strategy_imports(self):
        # Gradient
        import erasus.strategies.gradient_methods.fisher_forgetting
        import erasus.strategies.gradient_methods.negative_gradient
        # Parameter
        import erasus.strategies.parameter_methods.lora_unlearning
        import erasus.strategies.parameter_methods.mask_based
        # Data
        import erasus.strategies.data_methods.amnesiac
        # Specific
        import erasus.strategies.llm_specific.token_masking
        import erasus.strategies.diffusion_specific.noise_injection
        import erasus.strategies.vlm_specific.contrastive_unlearning

    def test_selector_imports(self):
        import erasus.selectors.gradient_based.influence
        import erasus.selectors.geometry_based.k_center
        import erasus.selectors.learning_based.loss_accum
        import erasus.selectors.random_selector

    def test_data_imports(self):
        import erasus.data.loaders
        import erasus.data.datasets
        import erasus.data.transforms
        
    def test_privacy_imports(self):
        import erasus.privacy.dp_mechanisms
        import erasus.privacy.accountant

    def test_visualization_imports(self):
        import erasus.visualization.loss_curves

if __name__ == "__main__":
    unittest.main()
