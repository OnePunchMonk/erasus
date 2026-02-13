"""
Tests for newly implemented components (Wrappers, Metrics, Selectors, Privacy, Viz).
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import os
import shutil

from erasus.privacy.accountant import PrivacyAccountant
from erasus.privacy.certificates import CertifiedRemover
from erasus.visualization.loss_curves import plot_loss_curve
from erasus.visualization.feature_plots import plot_embeddings

class TestNewComponents(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "test_artifacts"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_privacy_accountant(self):
        acc = PrivacyAccountant()
        acc.step(epsilon=1.0, delta=0.1)
        acc.step(epsilon=1.0, delta=0.1)
        eps, delta = acc.get_budget()
        self.assertAlmostEqual(eps, 2.0)
        self.assertAlmostEqual(delta, 0.2)

    def test_certified_remover(self):
        remover = CertifiedRemover(sigma=1.0)
        # Success case: Norm(1.0) / Lambda(2.0) = 0.5 <= Epsilon(1.0) * Sigma(1.0)
        is_cert = remover.check_certificate(1.0, 2.0, 1.0)
        self.assertTrue(is_cert)
        
        # Fail case
        is_cert = remover.check_certificate(10.0, 1.0, 1.0)
        self.assertFalse(is_cert)

    def test_visualization_generation(self):
        # Check if functions run without error and save files
        save_path = os.path.join(self.test_dir, "loss.png")
        plot_loss_curve([0.5, 0.4, 0.3], [0.1, 0.1, 0.1], save_path=save_path)
        self.assertTrue(os.path.exists(save_path))
        
        save_path_emb = os.path.join(self.test_dir, "emb.png")
        embs = np.random.randn(10, 5)
        plot_embeddings(embs, save_path=save_path_emb, method="pca") # Expect fallback if sklearn missing


if __name__ == "__main__":
    unittest.main()
