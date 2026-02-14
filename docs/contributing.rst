Contributing to Erasus
======================

Thank you for contributing to Erasus! This guide covers the
development workflow.

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/OnePunchMonk/erasus.git
   cd erasus

   # Install in development mode
   pip install -e ".[dev]"

   # Run tests
   python -m pytest tests/ -v

Code Style
----------

- Follow PEP 8 with 88-character line width (Black format)
- Use type hints for all public functions
- Write Google-style docstrings
- Keep functions focused and under 50 lines when possible

Adding a New Strategy
---------------------

1. Create ``erasus/strategies/<category>/<name>.py``
2. Subclass ``BaseStrategy`` and implement ``unlearn()``
3. Decorate with ``@strategy_registry.register("<name>")``
4. Add tests in ``tests/unit/``
5. Update ``strategies/__init__.py``

.. code-block:: python

   from erasus.core.base_strategy import BaseStrategy
   from erasus.core.registry import strategy_registry

   @strategy_registry.register("my_strategy")
   class MyStrategy(BaseStrategy):
       def unlearn(self, model, forget_loader, retain_loader, **kwargs):
           # Implementation
           return loss_history

Adding a New Selector
---------------------

Same pattern: subclass ``BaseSelector``, implement ``select()``,
register with ``@selector_registry.register("<name>")``.

Adding a New Metric
-------------------

Subclass ``BaseMetric``, implement ``compute()``, register with
``@metric_registry.register("<name>")``.

Testing
-------

.. code-block:: bash

   # Run all tests
   python -m pytest tests/ -v

   # Run specific test file
   python -m pytest tests/unit/test_strategies.py -v

   # Run with coverage
   python -m pytest tests/ --cov=erasus --cov-report=html

Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Write tests for your changes
4. Ensure all tests pass: ``python -m pytest tests/ -v``
5. Submit a PR with a clear description
