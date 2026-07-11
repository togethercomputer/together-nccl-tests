#!/bin/bash
#
# run_all_benchmarks.sh — Run full cluster health + compute validation suite
#
# Submits all benchmarks as independent Slurm jobs:
#   1. GPU health (clocks, ECC, throttle)
#   2. FA4 attention (BF16, FlashAttention-4)
#   3. FP8 GEMM (cublasLt E4M3)
#   4. NVFP4 GEMM (cublasLt E2M1)
#   5. NCCL scale-out (optional, with --nccl)
#
# Usage:
#   ./run_all_benchmarks.sh              # all compute benchmarks, all nodes
#   ./run_all_benchmarks.sh 16           # 16 nodes
#   ./run_all_benchmarks.sh --nccl 2 4 8 # also run NCCL at 2,4,8 nodes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PARTITION="${PARTITION:-batch}"
export GPU_TYPE="${GPU_TYPE:-B200}"
export RESULTS_BASE="${RESULTS_BASE:-/mnt/vast/dgxc-benchmarking-auto/nccl-results}"

RUN_NCCL=false
NCCL_NODE_COUNTS=()
NNODES=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl)
            RUN_NCCL=true
            shift
            # Remaining args are node counts for NCCL
            while [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9]+$ ]]; do
                NCCL_NODE_COUNTS+=("$1")
                shift
            done
            ;;
        *)
            NNODES="$1"
            shift
            ;;
    esac
done

echo "======================================================"
echo "  B200 Cluster Benchmark Suite"
echo "======================================================"
echo "  Partition:  ${PARTITION}"
echo "  GPU type:   ${GPU_TYPE}"
echo "  Nodes:      ${NNODES:-all available}"
echo "  NCCL:       ${RUN_NCCL}"
echo "  Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
echo ""

# 1. GPU Health Check
echo "--- GPU Health ---"
"${SCRIPT_DIR}/gpu-health/run.sh" ${NNODES}
echo ""

# 2. FA4 Benchmark
echo "--- FA4 (FlashAttention-4 BF16) ---"
"${SCRIPT_DIR}/fa4-bench/run.sh" ${NNODES}
echo ""

# 3. FP8 + NVFP4 GEMM Benchmarks
echo "--- FP8 + NVFP4 GEMM ---"
"${SCRIPT_DIR}/fp8-gemm-bench/run.sh" ${NNODES}
echo ""

# 4. NCCL Scale-Out (optional)
if [[ "$RUN_NCCL" == true ]]; then
    echo "--- NCCL Scale-Out ---"
    if [[ ${#NCCL_NODE_COUNTS[@]} -gt 0 ]]; then
        "${SCRIPT_DIR}/submit_scaleout.sh" "${NCCL_NODE_COUNTS[@]}"
    else
        "${SCRIPT_DIR}/submit_scaleout.sh"
    fi
    echo ""
fi

echo "======================================================"
echo "  All jobs submitted. Monitor: squeue -u $USER"
echo "  Results:  ${RESULTS_BASE}/${GPU_TYPE}/"
echo "======================================================"
