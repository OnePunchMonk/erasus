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

Releases are published to PyPI as the **erasus** package via GitHub Actions. The **version is taken from the Git tag** (e.g. tag `v0.1.2` → PyPI version `0.1.2`), so you don’t bump version in the repo for releases and you never publish the same version twice.

### Prerequisites

1. **PyPI trusted publishing** (recommended): On [PyPI](https://pypi.org), add your GitHub repo and configure the `pypi` environment with trusted publisher. No API token is needed in CI.
2. Or **API token**: Create a PyPI token and add it as repository secret `PYPI_API_TOKEN` (used by older workflows).

### Release steps

1. Commit and push to `main` (no need to edit `version.py` or `pyproject.toml` for the release).
2. On GitHub: **Releases** → **Create a new release**:
   - Create a **new tag** (e.g. `v0.1.2`). The tag name must be a valid version: `v0.1.2` or `0.1.2`.
   - Title e.g. `v0.1.2`.
   - Publish the release.
3. The workflow `.github/workflows/publish.yml` runs on `release: published`, sets the package version from the tag, builds once, and uploads to PyPI.
4. Users can install with: `pip install erasus==0.1.2` or `pip install erasus`, `pip install erasus[full]`, `pip install erasus[hub]`.

### Why a version didn’t change / 400 on upload

- **Version is defined by the tag.** The workflow overwrites `pyproject.toml` and `erasus/version.py` with the tag-derived version at build time. If you expected 0.1.2 but the release tag is still `v0.1.1`, the published package will be 0.1.1.
- **PyPI rejects re-uploading the same version.** If you see `400 Bad Request`, that version already exists on PyPI. Create a **new** tag (e.g. `v0.1.3`) and publish a new release; do not re-publish the same tag.
- **Stale `dist/`:** The workflow runs `rm -rf dist/` before building so only the current tag’s artifacts are uploaded.

### Local build (no upload)

```bash
pip install build
python -m build
# Artifacts in dist/: erasus-<version>.tar.gz, erasus-<version>-*.whl
# Version comes from pyproject.toml (for local dev).
```

## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
