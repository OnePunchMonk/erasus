import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from erasus.unlearners.erasus_unlearner import ErasusUnlearner

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def reproduce():
    model = SimpleModel()
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    retain_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    selectors = ["auto", "influence", "representer"]
    
    for selector in selectors:
        print(f"\nTesting selector: {selector}")
        try:
            unlearner = ErasusUnlearner(
                model=model,
                strategy="gradient_ascent",
                selector=selector,
                device="cpu",
                strategy_kwargs={"lr": 0.01}
            )
            # prune_ratio=0.5 -> select 50 samples
            unlearner.fit(forget_data=loader, retain_data=retain_loader, prune_ratio=0.5, epochs=1)
            print(f"Selector {selector} passed.")
        except Exception as e:
            print(f"Selector {selector} FAILED with error:")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    reproduce()
