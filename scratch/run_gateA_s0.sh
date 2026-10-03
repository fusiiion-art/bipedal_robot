#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/bipedal_robot

H=$(git rev-parse --short HEAD)
echo "=== [$(date)] Step 1: Commit hash is ${H} ==="

echo "=== [$(date)] Step 2: Running baseline evaluation ==="
mkdir -p "log/baseline_${H}"
./venv_wsl/bin/python scratch/phase0_eval_diagnostics.py --zero-policy --out "log/baseline_${H}/gate_a_zero.json"

# echo "=== [$(date)] Step 3: Training seed 0 ==="
# ./venv_wsl/bin/python train/train_mjx.py --seed 0 --exp_name "gateA_dbg_${H}_s0" --num_envs 256 --steps 10000000 --batch_size 256 --num_minibatches 16 --num_evals 20

# echo "=== [$(date)] Step 4: Evaluating seed 0 & comparing with baseline ==="
# ./venv_wsl/bin/python scratch/phase0_eval_diagnostics.py --exp_name "gateA_dbg_${H}_s0"
# ./venv_wsl/bin/python scratch/gate_a_qualification.py --reports "log/gateA_dbg_${H}_s0/version_0/gate_a.json" --baseline "log/baseline_${H}/gate_a_zero.json"

echo "=== [$(date)] Step 2 Baseline completed. Stopping before training as requested. ==="

