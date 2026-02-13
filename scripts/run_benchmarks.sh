#!/usr/bin/env bash
# run_benchmarks.sh — Run all Erasus benchmarks.
#
# Usage:
#   bash scripts/run_benchmarks.sh
#   bash scripts/run_benchmarks.sh --tofu-only
#   bash scripts/run_benchmarks.sh --epochs 5

set -euo pipefail

EPOCHS="${EPOCHS:-3}"
TOFU_ONLY=false

for arg in "$@"; do
    case $arg in
        --tofu-only) TOFU_ONLY=true ;;
        --epochs) shift; EPOCHS="$1" ;;
        --epochs=*) EPOCHS="${arg#*=}" ;;
    esac
done

echo "============================================"
echo "  Erasus — Benchmark Suite"
echo "  Epochs: $EPOCHS"
echo "============================================"

echo ""
echo "--- TOFU Benchmark ---"
python benchmarks/tofu/run.py --epochs "$EPOCHS"

if ! $TOFU_ONLY; then
    echo ""
    echo "--- WMDP Benchmark ---"
    python benchmarks/wmdp/run.py --epochs "$EPOCHS"
fi

echo ""
echo "============================================"
echo "  All benchmarks complete!"
echo "============================================"
