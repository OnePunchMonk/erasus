"""
Tests for Sprint D — Documentation & Tutorials.

Verifies that all documentation files referenced in the Sphinx toctree
exist and contain valid RST content.
"""

from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


# ===========================================================================
# 1. Top-level docs
# ===========================================================================
class TestTopLevelDocs:
    """Verify top-level doc files exist."""

    FILES = [
        "conf.py",
        "index.rst",
        "quickstart.rst",
        "installation.rst",
        "contributing.rst",
        "changelog.rst",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        assert (DOCS / f).exists(), f"Missing: docs/{f}"


# ===========================================================================
# 2. API Reference docs
# ===========================================================================
class TestAPIReferenceDocs:
    """Verify all API reference pages exist and have content."""

    FILES = [
        "api/core.rst",
        "api/unlearners.rst",
        "api/strategies.rst",
        "api/selectors.rst",
        "api/metrics.rst",
        "api/data.rst",
        "api/visualization.rst",
        "api/certification.rst",
        "api/privacy.rst",
        "api/utils.rst",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        path = DOCS / f
        assert path.exists(), f"Missing: docs/{f}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100, f"docs/{f} is too short"

    @pytest.mark.parametrize("f", FILES)
    def test_valid_rst(self, f):
        content = (DOCS / f).read_text(encoding="utf-8")
        # Every RST file should have a title (= underline)
        assert "===" in content or "---" in content, f"No RST heading in docs/{f}"


# ===========================================================================
# 3. User Guide docs
# ===========================================================================
class TestUserGuideDocs:
    """Verify all user guide pages exist."""

    FILES = [
        "guide/overview.rst",
        "guide/unlearning_pipeline.rst",
        "guide/strategies.rst",
        "guide/selectors.rst",
        "guide/metrics.rst",
        "guide/visualization.rst",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        path = DOCS / f
        assert path.exists(), f"Missing: docs/{f}"
        content = path.read_text(encoding="utf-8")
        assert len(content) > 200, f"docs/{f} is too short"


# ===========================================================================
# 4. Example docs
# ===========================================================================
class TestExampleDocs:
    """Verify example documentation pages exist."""

    FILES = [
        "examples/clip_basic.rst",
        "examples/llama_concept_removal.rst",
        "examples/stable_diffusion_nsfw.rst",
        "examples/tofu_benchmark.rst",
    ]

    @pytest.mark.parametrize("f", FILES)
    def test_exists(self, f):
        path = DOCS / f
        assert path.exists(), f"Missing: docs/{f}"


# ===========================================================================
# 5. Index toctree completeness
# ===========================================================================
class TestIndexToctree:
    """Verify all toctree entries in index.rst point to real files."""

    def test_all_toctree_entries_exist(self):
        index = (DOCS / "index.rst").read_text(encoding="utf-8")
        # Extract toctree entries (indented lines after .. toctree::)
        entries = []
        in_toctree = False
        for line in index.splitlines():
            stripped = line.strip()
            if ".. toctree::" in line:
                in_toctree = True
                continue
            if in_toctree:
                if stripped.startswith(":"):
                    continue
                if stripped == "":
                    if entries:
                        in_toctree = False
                    continue
                entries.append(stripped)

        for entry in entries:
            path = DOCS / f"{entry}.rst"
            assert path.exists(), f"toctree entry '{entry}' has no file at {path}"


# ===========================================================================
# 6. Sphinx conf.py validity
# ===========================================================================
class TestSphinxConfig:
    """Verify Sphinx configuration is valid."""

    def test_conf_has_project(self):
        conf = (DOCS / "conf.py").read_text(encoding="utf-8")
        assert "project" in conf
        assert "Erasus" in conf

    def test_conf_has_extensions(self):
        conf = (DOCS / "conf.py").read_text(encoding="utf-8")
        assert "extensions" in conf
        assert "sphinx.ext.autodoc" in conf

    def test_conf_has_theme(self):
        conf = (DOCS / "conf.py").read_text(encoding="utf-8")
        assert "html_theme" in conf
