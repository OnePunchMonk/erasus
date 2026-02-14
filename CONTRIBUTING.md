# Contributing to Erasus

Thank you for your interest in contributing to Erasus! This guide
will help you get started.

## Development Setup

```bash
git clone https://github.com/OnePunchMonk/erasus.git
cd erasus
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## How to Contribute

### Reporting Bugs
Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python/PyTorch version

### Adding Features
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Follow the coding conventions (PEP 8, type hints, docstrings)
4. Write tests for your changes
5. Ensure all tests pass: `python -m pytest tests/ -v`
6. Submit a pull request

### Adding Strategies, Selectors, or Metrics
See the developer guide in `docs/developer_guide/` for patterns.

## Code Style

- **PEP 8** with 88-character lines (Black format)
- **Type hints** for all public functions
- **Google-style docstrings**
- **Tests** for all new functionality

## Publishing to PyPI

Releases are published to PyPI as the **erasus** package via GitHub Actions.

### Prerequisites

1. **PyPI trusted publishing** (recommended): On [PyPI](https://pypi.org), add your GitHub repo and configure the `pypi` environment with trusted publisher. No API token is needed in CI.
2. Or **API token**: Create a PyPI token and add it as repository secret `PYPI_API_TOKEN` (used by older workflows).

### Release steps

1. Bump version in `erasus/version.py` and `pyproject.toml` (if not auto-synced).
2. Commit and push to `main`.
3. On GitHub: **Releases** → **Create a new release**:
   - Choose a tag (e.g. `v0.1.0`); create the tag if needed.
   - Title e.g. `v0.1.0`.
   - Publish the release.
4. The workflow `.github/workflows/publish.yml` runs on `release: published` and uploads to PyPI.
5. Users can install with: `pip install erasus` or `pip install erasus[full]`, `pip install erasus[hub]`.

### Local build (no upload)

```bash
pip install build
python -m build
# Artifacts in dist/: erasus-0.1.0.tar.gz, erasus-0.1.0-*.whl
```

## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
