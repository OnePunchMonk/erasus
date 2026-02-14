# Adding New Selectors

## Step 1: Choose a Category

Place your selector in the matching package:
- `erasus/selectors/influence_based/` — Influence functions, TracIn
- `erasus/selectors/geometry_based/` — Herding, k-center, facility location
- `erasus/selectors/gradient_based/` — Gradient matching, grad norm
- `erasus/selectors/learning_based/` — Forgetting score, active learning
- `erasus/selectors/ensemble/` — Stacking, voting, weighted fusion

## Step 2: Implement the Selector

```python
# erasus/selectors/my_category/my_selector.py
from erasus.core.base_selector import BaseSelector
from erasus.core.registry import selector_registry

@selector_registry.register("my_selector")
class MySelector(BaseSelector):
    """My custom coreset selector."""

    def __init__(self, budget: int = 100):
        super().__init__()
        self.budget = budget

    def select(self, dataset, n, model=None):
        """Select n indices from dataset.

        Args:
            dataset: The full dataset to select from
            n: Number of points to select
            model: Optional model for model-dependent selection

        Returns:
            List of selected indices
        """
        # Your selection logic
        import torch
        scores = self._compute_scores(dataset, model)
        _, indices = torch.topk(scores, n)
        return indices.tolist()

    def _compute_scores(self, dataset, model):
        # Score each data point
        import torch
        return torch.rand(len(dataset))
```

## Step 3: Register

Import in the `__init__.py` of the parent package so the
`@selector_registry.register` decorator fires on import.

## Step 4: Use

```python
unlearner = ErasusUnlearner(
    model=model,
    strategy="gradient_ascent",
    selector="my_selector",
)
result = unlearner.fit(forget_data=forget, retain_data=retain, prune_ratio=0.3)
```

## Step 5: Test

Verify:
- Selection produces valid indices
- Selected set size matches requested `n`
- Quality metrics (coverage, diversity) are reasonable
