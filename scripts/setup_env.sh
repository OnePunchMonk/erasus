#!/usr/bin/env bash
# setup_env.sh — Set up the Erasus development environment.
#
# Usage:
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --gpu    # Install GPU PyTorch
#
# Prerequisites: Python 3.10+, pip

set -euo pipefail

echo "============================================"
echo "  Erasus — Environment Setup"
echo "============================================"

GPU_MODE=false
if [[ "${1:-}" == "--gpu" ]]; then
    GPU_MODE=true
fi

# Create virtual environment if not active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo ""
    echo "  Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    echo "  ✓ Virtual environment activated"
else
    echo "  ✓ Virtual environment already active: $VIRTUAL_ENV"
fi

# Upgrade pip
echo ""
echo "  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo ""
if $GPU_MODE; then
    echo "  Installing PyTorch (GPU/CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  Installing PyTorch (CPU)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo ""
echo "  Installing project dependencies..."
pip install -r requirements.txt

# Install Erasus in editable mode
echo ""
echo "  Installing Erasus (editable)..."
pip install -e ".[dev]" 2>/dev/null || pip install -e .

# Dev tools
echo ""
echo "  Installing development tools..."
pip install pytest pytest-cov ruff mypy ipython

# Verify
echo ""
echo "  Verifying installation..."
python -c "import erasus; print(f'  ✓ Erasus imported successfully')"
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')"
python -c "import torch; print(f'  ✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "============================================"
echo "  Setup complete! Activate with:"
echo "    source .venv/bin/activate"
echo "============================================"
