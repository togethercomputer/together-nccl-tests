#!/bin/bash
#
# run.sh — Submit FA4 benchmark across all cluster nodes via Slurm
#
# Runs fa4_bench_node.py on every node (1 task/node, 8 GPUs/node).
# Each node benchmarks all 8 GPUs sequentially.
#
# Usage:
#   ./run.sh                     # all nodes in partition
#   ./run.sh 8                   # specific number of nodes
#   PARTITION=batch ./run.sh 16  # custom partition
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/mnt/vast/dgxc-benchmarking-auto/fa4-venv}"
PYTHON="${VENV_DIR}/bin/python"
BENCH_SCRIPT="${SCRIPT_DIR}/fa4_bench_node.py"

PARTITION="${PARTITION:-batch}"
GPUS_PER_NODE=8
NNODES="${1:-}"

RESULTS_BASE="${RESULTS_BASE:-/mnt/vast/dgxc-benchmarking-auto/nccl-results}"
GPU_TYPE="${GPU_TYPE:-B200}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_BASE}/${GPU_TYPE}/fa4_bench_${TIMESTAMP}"

# Validate
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Venv not found at ${VENV_DIR}"
    echo "Run ./setup.sh first"
    exit 1
fi

if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "ERROR: Benchmark script not found: ${BENCH_SCRIPT}"
    exit 1
fi

# Copy bench script to shared fs (compute nodes can't see /home)
SHARED_SCRIPT="/mnt/vast/dgxc-benchmarking-auto/fa4-bench-node.py"
cp "$BENCH_SCRIPT" "$SHARED_SCRIPT"

mkdir -p "$RESULTS_DIR"

# Build sbatch args
SBATCH_ARGS=(
    --job-name=fa4-bench
    --partition="${PARTITION}"
    --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task=16
    --mem=0
    --time=00:30:00
    --exclusive
    --output="${RESULTS_DIR}/fa4_bench_%j.out"
)

if [[ -n "$NNODES" ]]; then
    SBATCH_ARGS+=(--nodes="${NNODES}")
fi

echo "=== FA4 Benchmark Submission ==="
echo "  Partition: ${PARTITION}"
echo "  Nodes:     ${NNODES:-all available}"
echo "  Results:   ${RESULTS_DIR}"
echo ""

JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" --wrap "
export HOME=/tmp
export TRITON_CACHE_DIR=/tmp/triton_cache
export XDG_CACHE_HOME=/tmp/xdg_cache
export TORCH_HOME=/tmp/torch_home

echo 'node,gpu,time_ms,tflops,mfu'
srun ${PYTHON} ${SHARED_SCRIPT}
")

echo "  Submitted job ${JOB_ID}"
echo "  Monitor:  squeue -j ${JOB_ID}"
echo "  Results:  ${RESULTS_DIR}/fa4_bench_${JOB_ID}.out"
