#!/bin/bash
# Quick setup script for AIF_pymdp project
# Run this after container restart to restore the environment

set -e

cd /workspace/AIF_pymdp

echo "=== Setting up AIF_pymdp environment ==="

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing pymdp and CUDA-enabled JAX..."
pip install inferactively-pymdp jax[cuda12] 2>&1 | tail -5

echo ""
echo "Verifying GPU access..."
python3 -c "
import jax
print('JAX devices:', jax.devices())
print('JAX backend:', jax.default_backend())
import jax.numpy as jnp
x = jnp.ones((100,100))
_ = jnp.dot(x,x).block_until_ready()
print('GPU compute: OK')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source /workspace/AIF_pymdp/.venv/bin/activate"
echo "Run benchmarks: python3 run_benchmarks.py"
