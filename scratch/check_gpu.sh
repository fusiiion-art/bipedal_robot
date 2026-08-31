#!/bin/bash
export VIRTUAL_ENV=/mnt/c/bipedal_robot/venv_wsl
export PATH=/mnt/c/bipedal_robot/venv_wsl/bin:$PATH
cd /mnt/c/bipedal_robot

echo "=== Current JAX version ==="
python3 -c "import jax; print('jax:', jax.__version__)" 2>/dev/null
python3 -c "import jaxlib; print('jaxlib:', jaxlib.__version__)" 2>/dev/null

echo "=== NVIDIA GPU check ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not found"

echo "=== CUDA version ==="
nvcc --version 2>/dev/null || echo "nvcc not found"
ls /usr/local/cuda*/version.txt 2>/dev/null && cat /usr/local/cuda*/version.txt 2>/dev/null
ls /usr/local/cuda/lib64/libcudart* 2>/dev/null || echo "No CUDA runtime libs found in /usr/local/cuda"

echo "=== pip list (jax related) ==="
pip list 2>/dev/null | grep -i -E "jax|cuda|nvidia"
