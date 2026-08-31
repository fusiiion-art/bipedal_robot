#!/bin/bash
export VIRTUAL_ENV=/mnt/c/bipedal_robot/venv_wsl
export PATH=/mnt/c/bipedal_robot/venv_wsl/bin:$PATH
cd /mnt/c/bipedal_robot

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

echo "=== Starting GPU Training (Memory-Optimized) ==="
python3 train/train_mjx.py --num_envs 128 --batch_size 128 --num_minibatches 8
