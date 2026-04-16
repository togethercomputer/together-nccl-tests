#!/bin/bash
#
# setup.sh — Install FA4 benchmark venv on shared filesystem
#
# Creates a uv-managed venv with flash-attn-4 + PyTorch (cu129).
# The venv is placed on /mnt/vast so compute nodes can access it.
#
# Prerequisites: uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Usage:
#   ./setup.sh              # install to default path
#   VENV_DIR=/path ./setup.sh  # install to custom path
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/mnt/vast/dgxc-benchmarking-auto/fa4-venv}"

echo "=== FA4 Benchmark Setup ==="
echo "  Venv: ${VENV_DIR}"

# Create venv with uv
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "  Creating venv..."
    uv venv --python 3.12 "${VENV_DIR}"
fi

echo "  Installing flash-attn-4 + PyTorch cu129..."
# flash-attn-4 pulls cu130 PyTorch by default, but B200 driver only supports cu129
uv pip install --python "${VENV_DIR}/bin/python" \
    "torch>=2.11.0" \
    --index-url https://download.pytorch.org/whl/cu129

uv pip install --python "${VENV_DIR}/bin/python" \
    "flash-attn-4>=4.0.0b8"

# Force cu129 torch in case flash-attn-4 overwrote it
TORCH_VER=$("${VENV_DIR}/bin/python" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
if [[ "$TORCH_VER" != 12.9* ]] && [[ "$TORCH_VER" != "12.9" ]]; then
    echo "  Fixing PyTorch CUDA version (got ${TORCH_VER}, need 12.9)..."
    uv pip install --python "${VENV_DIR}/bin/python" \
        "torch>=2.11.0+cu129" \
        --index-url https://download.pytorch.org/whl/cu129
fi

echo "  Verifying..."
"${VENV_DIR}/bin/python" -c "
import torch
from flash_attn.cute import flash_attn_func
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  flash_attn_func: OK')
print(f'  GPUs visible: {torch.cuda.device_count()}')
"

echo ""
echo "=== Setup complete ==="
echo "  Venv: ${VENV_DIR}"
echo "  Run:  ./run.sh"
