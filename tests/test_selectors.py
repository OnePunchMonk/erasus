"""
Tests for Selectors.
"""
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from erasus.selectors.gradient_based.el2n import EL2NSelector
from erasus.selectors.geometry_based.submodular import SubmodularSelector
from erasus.selectors.geometry_based.kmeans_coreset import KMeansSelector
from erasus.selectors.gradient_based.grad_match import GradMatchSelector
from erasus.selectors.learning_based.valuation_network import ValuationNetworkSelector
from erasus.core.exceptions import SelectorError

class TestSelectors(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(5, 2)
        self.data = DataLoader(TensorDataset(torch.randn(20, 5), torch.randint(0, 2, (20,))), batch_size=5)

    def test_el2n(self):
        sel = EL2NSelector()
        idc = sel.select(self.model, self.data, k=5)
        self.assertEqual(len(idc), 5)

    def test_el2n_requires_labels(self):
        sel = EL2NSelector()
        unlabeled = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=5)
        with self.assertRaises(SelectorError):
            sel.select(self.model, unlabeled, k=3)

    def test_submodular(self):
        sel = SubmodularSelector()
        idc = sel.select(self.model, self.data, k=5)
        self.assertEqual(len(idc), 5)
        
    def test_kmeans(self):
        # Requires sklearn
        try:
            import sklearn
            sel = KMeansSelector()
            idc = sel.select(self.model, self.data, k=3)
            self.assertEqual(len(idc), 3)
        except ImportError:
            pass

    def test_grad_match(self):
        sel = GradMatchSelector()
        idc = sel.select(self.model, self.data, k=5)
        self.assertEqual(len(idc), 5)

    def test_valuation_network_requires_labels(self):
        class DummyValNet(nn.Module):
            def forward(self, x, y):
                return torch.ones(x.size(0), device=x.device)

        sel = ValuationNetworkSelector()
        unlabeled = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=5)
        with self.assertRaises(SelectorError):
            sel.select(self.model, unlabeled, k=3, val_net=DummyValNet())

if __name__ == "__main__":
    unittest.main()
