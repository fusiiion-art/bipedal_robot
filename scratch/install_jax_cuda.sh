#!/bin/bash
export VIRTUAL_ENV=/mnt/c/bipedal_robot/venv_wsl
export PATH=/mnt/c/bipedal_robot/venv_wsl/bin:$PATH
cd /mnt/c/bipedal_robot

echo "=== Installing JAX with CUDA support ==="
# JAX 0.11.0 with CUDA 12 (bundled CUDA libs via pip, no need for separate CUDA toolkit)
pip install --upgrade "jax[cuda12]"

echo ""
echo "=== Verifying GPU access ==="
python3 -c "
import jax
print('Devices:', jax.devices())
print('Platform:', jax.devices()[0].platform)
if jax.devices()[0].platform == 'gpu':
    print('SUCCESS: GPU is ready!')
else:
    print('FAILED: Still on CPU')
"
