#!/bin/bash
#
# setup_and_build.sh — Install NCCL + build nccl-tests on the cluster
#
# Run on a node with internet access (login node or salloc'd node).
# Builds to a shared filesystem so all nodes can use the binaries.
#
# Usage:
#   ./setup_and_build.sh                    # default: install to /mnt/vast/exemplar/llmb/nccl-tests
#   ./setup_and_build.sh /path/to/install   # custom install path
#
set -euo pipefail

INSTALL_DIR="${1:-/mnt/vast/exemplar/llmb/nccl-tests}"
NCCL_TESTS_VERSION="${2:-v2.13.11}"

echo "=== NCCL Tests Setup ==="
echo "Install dir: ${INSTALL_DIR}"
echo ""

# ──────── Step 1: Check/Install NCCL ────────
if ldconfig -p 2>/dev/null | grep -q libnccl.so; then
    echo "[OK] NCCL already installed"
elif dpkg -l libnccl2 2>/dev/null | grep -q ^ii; then
    echo "[OK] libnccl2 package installed"
else
    echo "[WARN] NCCL not found on this host."
    echo ""
    echo "Install NCCL system-wide (needs sudo):"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y libnccl2 libnccl-dev"
    echo ""
    echo "Or use the container approach instead (no sudo needed):"
    echo "  See Dockerfile and run_nccl_test.sh"
    echo ""

    # Try to install if we have sudo
    if sudo -n true 2>/dev/null; then
        echo "sudo available, installing NCCL..."
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends libnccl2 libnccl-dev
    else
        echo "No sudo access. Options:"
        echo "  1. Ask admin to install: sudo apt install libnccl2 libnccl-dev"
        echo "  2. Use the container approach (recommended)"
        echo "  3. Download NCCL manually from https://developer.nvidia.com/nccl"
        exit 1
    fi
fi

# ──────── Step 2: Source HPC-X ────────
echo ""
echo "Loading HPC-X..."
source /opt/hpcx/hpcx-init.sh
hpcx_load

echo "  MPI:  $(which mpicc)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}')"

# ──────── Step 3: Clone & Build ────────
echo ""
echo "Building nccl-tests ${NCCL_TESTS_VERSION}..."

TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

git clone --depth 1 --branch "${NCCL_TESTS_VERSION}" \
    https://github.com/NVIDIA/nccl-tests.git "${TMPDIR}/nccl-tests"

cd "${TMPDIR}/nccl-tests"

make -j$(nproc) \
    MPI=1 \
    MPI_HOME="${HPCX_MPI_DIR}" \
    CUDA_HOME=/usr/local/cuda

# ──────── Step 4: Install ────────
mkdir -p "${INSTALL_DIR}"
cp -r build "${INSTALL_DIR}/"
echo "${NCCL_TESTS_VERSION}" > "${INSTALL_DIR}/VERSION"

echo ""
echo "=== Build complete ==="
echo ""
echo "Binaries:"
ls -1 "${INSTALL_DIR}/build/"*_perf
echo ""
echo "Quick test (single node, 8 GPUs):"
echo "  srun -N1 --ntasks-per-node=8 --gpus-per-node=8 ${INSTALL_DIR}/build/all_reduce_perf -b 1M -e 1G -g 1"
echo ""
echo "Or use the wrapper:"
echo "  ./run_nccl_native.sh -n 1 -t all_reduce"
