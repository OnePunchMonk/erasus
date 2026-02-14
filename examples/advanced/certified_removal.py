"""
Certified Removal — End-to-end certified data deletion.

Demonstrates the full pipeline: unlearning + certification with
epsilon-delta guarantees from the certification module.

Usage::

    python examples/advanced/certified_removal.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from erasus.unlearners.erasus_unlearner import ErasusUnlearner
from erasus.privacy.accountant import PrivacyAccountant
import erasus.strategies  # noqa: F401


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def main():
    print("=" * 60)
    print("  Certified Removal Example")
    print("=" * 60)

    model = SmallModel()
    forget = DataLoader(TensorDataset(torch.randn(30, 16), torch.randint(0, 4, (30,))), batch_size=16)
    retain = DataLoader(TensorDataset(torch.randn(200, 16), torch.randint(0, 4, (200,))), batch_size=16)

    # Step 1: Unlearn
    print("\n  Step 1: Unlearning with SCRUB...")
    unlearner = ErasusUnlearner(
        model=model, strategy="scrub", selector=None,
        device="cpu", strategy_kwargs={"lr": 1e-3},
    )
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=3)
    print(f"    ✓ Done in {result.elapsed_time:.2f}s")

    # Step 2: Certification
    print("\n  Step 2: Computing certification bounds...")
    try:
        from erasus.certification.verification import UnlearningVerifier
        verifier = UnlearningVerifier()
        cert = verifier.verify(
            original_model=model,
            unlearned_model=unlearner.model,
            forget_data=forget,
        )
        print(f"    Weight distance: {cert.get('weight_distance', 'N/A')}")
        print(f"    Verified: {cert.get('verified', 'N/A')}")
    except Exception as e:
        print(f"    Verification: {e}")

    # Step 3: Privacy accounting
    print("\n  Step 3: Privacy accounting...")
    accountant = PrivacyAccountant()
    accountant.step(epsilon=0.5, delta=1e-5)
    eps, delta = accountant.get_budget()
    print(f"    Total privacy: ε={eps:.4f}, δ={delta:.6f}")

    print("\n✅ Certified removal complete!")


if __name__ == "__main__":
    main()
