"""
Tests for Sprint E — CI/CD, Infrastructure & Sprint G Publishing.

Validates CI/CD workflows, GitHub templates, Docker configs,
project metadata, and Sprint D docs are complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# 1. CI/CD Workflows
# ===========================================================================
class TestCIWorkflows:
    """Verify all CI/CD workflow files exist."""

    WORKFLOWS = [
        ".github/workflows/ci.yml",
        ".github/workflows/publish.yml",
        ".github/workflows/docs.yml",
        ".github/workflows/benchmarks.yml",
        ".github/workflows/security.yml",
    ]

    @pytest.mark.parametrize("wf", WORKFLOWS)
    def test_exists(self, wf):
        path = ROOT / wf
        assert path.exists(), f"Missing: {wf}"

    def test_ci_has_test_job(self):
        content = (ROOT / ".github/workflows/ci.yml").read_text()
        assert "test:" in content or "Tests" in content

    def test_publish_has_pypi(self):
        content = (ROOT / ".github/workflows/publish.yml").read_text()
        assert "pypi" in content.lower()

    def test_benchmarks_has_tofu(self):
        content = (ROOT / ".github/workflows/benchmarks.yml").read_text()
        assert "tofu" in content.lower()


# ===========================================================================
# 2. GitHub Templates
# ===========================================================================
class TestGitHubTemplates:
    """Verify GitHub templates exist."""

    TEMPLATES = [
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
    ]

    @pytest.mark.parametrize("t", TEMPLATES)
    def test_exists(self, t):
        assert (ROOT / t).exists(), f"Missing: {t}"


# ===========================================================================
# 3. Docker
# ===========================================================================
class TestDocker:
    """Verify Docker configurations."""

    FILES = [
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docker/Dockerfile.gpu",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        assert (ROOT / f).exists(), f"Missing: {f}"

    def test_gpu_dockerfile_has_cuda(self):
        content = (ROOT / "docker/Dockerfile.gpu").read_text()
        assert "cuda" in content.lower()


# ===========================================================================
# 4. Project Metadata
# ===========================================================================
class TestProjectMetadata:
    """Verify project metadata files."""

    FILES = [
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        assert (ROOT / f).exists(), f"Missing: {f}"

    def test_citation_has_title(self):
        content = (ROOT / "CITATION.cff").read_text()
        assert "erasus" in content.lower()

    def test_pyproject_has_name(self):
        content = (ROOT / "pyproject.toml").read_text()
        assert "erasus" in content.lower()


# ===========================================================================
# 5. Version Module
# ===========================================================================
class TestVersionModule:
    """Verify version is accessible."""

    def test_version_file_exists(self):
        assert (ROOT / "erasus" / "version.py").exists()

    def test_version_importable(self):
        from erasus.version import __version__
        assert __version__
        assert "." in __version__


# ===========================================================================
# 6. Docs Makefile
# ===========================================================================
class TestDocsMakefile:
    """Verify Sphinx Makefile."""

    def test_makefile_exists(self):
        assert (ROOT / "docs" / "Makefile").exists()

    def test_makefile_has_html_target(self):
        content = (ROOT / "docs" / "Makefile").read_text()
        assert "html" in content


# ===========================================================================
# 7. Developer Guide
# ===========================================================================
class TestDeveloperGuide:
    """Verify developer guide docs."""

    FILES = [
        "docs/developer_guide/architecture.md",
        "docs/developer_guide/adding_models.md",
        "docs/developer_guide/adding_selectors.md",
        "docs/developer_guide/testing.md",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        assert (ROOT / f).exists(), f"Missing: {f}"


# ===========================================================================
# 8. Research Docs
# ===========================================================================
class TestResearchDocs:
    """Verify research documentation."""

    FILES = [
        "docs/research/theory.md",
        "docs/research/coreset_analysis.md",
        "docs/research/utility_bounds.md",
        "docs/research/benchmarks.md",
        "docs/research/paper_reproductions.md",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        assert (ROOT / f).exists(), f"Missing: {f}"
