#!/usr/bin/env bash
set -euo pipefail

# Training script for Stage 2

# Activate uv venv if present
if [ -d .venv ]; then
  source .venv/bin/activate
fi

# Default to base config; allow override via $1
CONFIG=${1:-configs/base_config.yaml}
OVERRIDE=${2:-}

python src/training/train_qlora.py --config "$CONFIG" ${OVERRIDE:+--override "$OVERRIDE"}

# Run evaluation afterwards (optional)
python src/evaluation/evaluate_helpfulness.py --config "$CONFIG"
