#!/bin/bash
# Reproduce all results from scratch.
# Run from the repo root with: bash run_all.sh
set -euo pipefail

echo "============================================================"
echo " EEG MI Benchmark — Full Reproduction Pipeline"
echo "============================================================"

# Activate venv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo ""
echo "[1/3] Verifying dataset..."
python3 scripts/00_quick_verify.py

echo ""
echo "[2/3] Running LOSO experiments (checkpoints saved after each fold)..."
python3 scripts/02_run_experiments.py

echo ""
echo "[3/3] Generating figures and tables..."
python3 scripts/03_generate_figures.py

echo ""
echo "============================================================"
echo " All done. Results in results/logs/, results/tables/, results/figures/"
echo "============================================================"
