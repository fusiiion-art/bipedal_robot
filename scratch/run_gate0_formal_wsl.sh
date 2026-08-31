#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/bipedal_robot
PYTHON="$ROOT/venv_wsl/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing WSL Python: $PYTHON" >&2
  exit 2
fi

cd "$ROOT"
export PYTHONPATH="$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

SECONDS_TO_RUN="${1:-10}"
SEED_START="${2:-0}"
SEED_COUNT="${3:-1}"

for ((offset=0; offset<SEED_COUNT; offset++)); do
  seed=$((SEED_START + offset))
  echo "[Gate0] seed=$seed seconds=$SECONDS_TO_RUN python=$PYTHON"
  "$PYTHON" scratch/gate0_formal_eval.py \
    --seconds "$SECONDS_TO_RUN" \
    --seed "$seed"
done
