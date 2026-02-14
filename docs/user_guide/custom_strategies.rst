Writing Custom Strategies
=========================

Extend Erasus with your own unlearning strategies.

Step 1: Create the Strategy
----------------------------

.. code-block:: python

   # erasus/strategies/my_methods/my_strategy.py
   import torch
   from erasus.core.base_strategy import BaseStrategy
   from erasus.core.registry import strategy_registry

   @strategy_registry.register("my_custom_strategy")
   class MyCustomStrategy(BaseStrategy):
       """My custom unlearning strategy."""

       def __init__(self, lr: float = 1e-3, alpha: float = 0.5):
           super().__init__()
           self.lr = lr
           self.alpha = alpha

       def unlearn(self, model, forget_loader, retain_loader, **kwargs):
           optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
           epochs = kwargs.get("epochs", 5)
           loss_history = []

           for epoch in range(epochs):
               model.train()
               epoch_loss = 0
               for x, y in forget_loader:
                   # Your custom unlearning logic
                   out = model(x)
                   forget_loss = -torch.nn.functional.cross_entropy(out, y)
                   loss = self.alpha * forget_loss
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   epoch_loss += loss.item()
               loss_history.append(epoch_loss / len(forget_loader))

           return loss_history

Step 2: Register the Strategy
------------------------------

The ``@strategy_registry.register()`` decorator handles registration
automatically when the module is imported.

Ensure your module is imported by adding it to the relevant
``__init__.py``:

.. code-block:: python

   # erasus/strategies/my_methods/__init__.py
   from .my_strategy import MyCustomStrategy

Step 3: Use Your Strategy
--------------------------

.. code-block:: python

   unlearner = ErasusUnlearner(
       model=model,
       strategy="my_custom_strategy",
       strategy_kwargs={"lr": 1e-3, "alpha": 0.7},
   )
   result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=5)

Best Practices
--------------

- Always accept ``**kwargs`` in ``unlearn()`` for forward compatibility
- Return a list of loss values per epoch
- Use ``model.train()`` / ``model.eval()`` appropriately
- Support both CPU and CUDA tensors
- Add type hints and docstrings
