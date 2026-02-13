"""
Tests for Multi-modal models and Ensemble functionality.
"""
import unittest
import torch
import torch.nn as nn
from erasus.models.registry import model_registry

# Explicitly import to trigger registration
import erasus.models.video  
import erasus.models.audio
from erasus.selectors.ensemble.voting import VotingSelector
# Ensure random selector is registered for voting test
import erasus.selectors.random_selector 

class TestMultiModalWrappers(unittest.TestCase):
    def test_videomae_import(self):
        # We check if class is registered
        cls = model_registry.get("videomae")
        self.assertIsNotNone(cls)

    def test_whisper_import(self):
        cls = model_registry.get("whisper")
        self.assertIsNotNone(cls)

class TestVotingSelector(unittest.TestCase):
    def test_voting_mechanism(self):
        selector = VotingSelector(selector_names=["random", "random"])
        model = nn.Linear(5, 2)
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
        loader = DataLoader(dataset, batch_size=2)
        
        indices = selector.select(model, loader, k=2)
        # Should return list of ints
        self.assertIsInstance(indices, list)
        self.assertTrue(len(indices) <= 2) # Might condense duplicates

if __name__ == "__main__":
    unittest.main()
