# Testing Guide

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific sprint
python -m pytest tests/unit/test_sprint_b.py -v

# With coverage
python -m pytest tests/ --cov=erasus --cov-report=html
```

## Test Organisation

```
tests/
├── unit/
│   ├── test_strategies.py        # Strategy registration & execution
│   ├── test_selectors.py         # Selector registration & execution
│   ├── test_metrics.py           # Metric computation
│   ├── test_unlearners.py        # Unlearner integration
│   ├── test_data.py              # Data loading & preprocessing
│   ├── test_visualization.py     # Visualization modules
│   ├── test_privacy.py           # Privacy modules
│   ├── test_certification.py     # Certification modules
│   ├── test_cli.py               # CLI commands
│   ├── test_sprint_a.py          # Sprint A additions
│   ├── test_sprint_b.py          # Sprint B additions
│   ├── test_sprint_c.py          # Sprint C additions
│   ├── test_sprint_d.py          # Sprint D additions
│   └── test_sprint_f.py          # Sprint F additions
└── integration/
    └── (future integration tests)
```

## Writing Tests

### Strategy Tests
```python
def test_my_strategy():
    model = nn.Sequential(nn.Linear(16, 4))
    forget = DataLoader(TensorDataset(torch.randn(32, 16), torch.randint(0, 4, (32,))), batch_size=8)
    retain = DataLoader(TensorDataset(torch.randn(64, 16), torch.randint(0, 4, (64,))), batch_size=8)

    unlearner = ErasusUnlearner(model=model, strategy="my_strategy", device="cpu")
    result = unlearner.fit(forget_data=forget, retain_data=retain, epochs=2)

    assert result.elapsed_time > 0
    assert len(result.forget_loss_history) == 2
```

### Import Tests (for examples/scripts)
```python
def test_script_importable():
    spec = importlib.util.spec_from_file_location("mod", "examples/my_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
```

## Conventions

- Use `torch.randn` / `torch.randint` for synthetic test data
- Keep models small (hidden=64 or less)
- Use `device="cpu"` in all tests
- Each test file should be self-contained
- Group related tests in classes
