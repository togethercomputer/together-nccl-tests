#!/bin/bash
# NCCL AllReduce performance test: compare Ring vs NVLS vs CollNet(SHARP)
#
# Key concepts:
#   NCCL_NVLS_ENABLE    — single-node NVLink SHARP (NVSwitch); enabled by default
#                         but silently suppressed by NCCL_ALGO=RING
#   NCCL_COLLNET_ENABLE — multi-node InfiniBand SHARP; orthogonal to NVLS
#
# NOTE: This cluster has NCCL_ALGO=RING in /etc/environment.
#       Configs needing NVLS/CollNet unset it via "env -u NCCL_ALGO".
#
# Usage:
#   Single-node  : bash run_nccl_sharp_test.sh
#   Multi-node   : bash run_nccl_sharp_test.sh --nodes gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs
#   Inside sbatch: bash run_nccl_sharp_test.sh --nodes $SLURM_NODELIST
#
# Results are saved under: benchmarks/<GPU_TYPE>/results/<timestamp>/

set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_TESTS_DIR="$(dirname "$BENCHMARKS_DIR")"
BUILD_DIR="$NCCL_TESTS_DIR/build"

NGPUS_PER_NODE=8
MSG_SIZE_BEGIN="1M"
MSG_SIZE_END="8G"
STEP_FACTOR="2"
WARMUP_ITERS="5"
RUN_ITERS="20"
NODELIST=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes|-N)  NODELIST="$2"; shift 2 ;;
        --nodes=*)   NODELIST="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect GPU type (e.g. "H100", "H200", "A100")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NR==1{print; exit}')
GPU_NAME="${GPU_NAME:-unknown}"
if [[ "$GPU_NAME" == *"H200"* ]]; then
    GPU_TYPE="H200"
elif [[ "$GPU_NAME" == *"H100"* ]]; then
    GPU_TYPE="H100"
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    GPU_TYPE="A100"
elif [[ "$GPU_NAME" == *"B200"* ]]; then
    GPU_TYPE="B200"
elif [[ "$GPU_NAME" == *"GB200"* ]]; then
    GPU_TYPE="GB200"
else
    GPU_TYPE="$(echo "$GPU_NAME" | tr ' ' '_')"
fi

RESULTS_DIR="$BENCHMARKS_DIR/$GPU_TYPE/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Determine mode: single-node or multi-node via srun
if [[ -z "$NODELIST" ]]; then
    NNODES=1
    TOTAL_GPUS=$NGPUS_PER_NODE
    MODE="local"
    LAUNCHER=""
else
    NNODES=$(scontrol show hostnames "$NODELIST" | wc -l)
    TOTAL_GPUS=$(( NNODES * NGPUS_PER_NODE ))
    MODE="cluster"
    LAUNCHER="srun --nodelist=$NODELIST \
                   --ntasks=$NNODES \
                   --ntasks-per-node=1 \
                   --gres=gpu:$NGPUS_PER_NODE \
                   --cpus-per-task=64 \
                   --mem=0"
fi

echo "============================================================"
echo "NCCL AllReduce Benchmark — $(date)"
echo "GPU Type    : $GPU_TYPE ($GPU_NAME)"
echo "Mode        : $MODE"
if [[ $MODE == "cluster" ]]; then
echo "Nodes       : $NNODES ($NODELIST)"
fi
echo "Total GPUs  : $TOTAL_GPUS | GPUs/node: $NGPUS_PER_NODE"
echo "Msg range   : ${MSG_SIZE_BEGIN}..${MSG_SIZE_END}"
echo "Results dir : $RESULTS_DIR"
echo "============================================================"

run_test() {
    local label="$1"
    local test_name="$2"
    local binary="$BUILD_DIR/${test_name}_perf"
    local outfile="$RESULTS_DIR/${test_name}_${label}.out"
    shift 2

    if [[ ! -x "$binary" ]]; then
        echo "  [SKIP] $binary not found"
        return
    fi

    echo ""
    echo "--- ${test_name} | config=${label} | mode=${MODE} ---"

    "$@" $LAUNCHER "$binary" \
        -b "$MSG_SIZE_BEGIN" \
        -e "$MSG_SIZE_END" \
        -f "$STEP_FACTOR" \
        -g "$NGPUS_PER_NODE" \
        -w "$WARMUP_ITERS" \
        -n "$RUN_ITERS" \
        2>&1 | tee "$outfile"
}

# Config 1: Ring — baseline (mirrors /etc/environment)
run_test "ring" "all_reduce" \
    env NCCL_ALGO=RING NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=0

# Config 2: NVLS only — single-node NVLink SHARP (NVSwitch)
run_test "nvls" "all_reduce" \
    env -u NCCL_ALGO NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=0

# Config 3: CollNet only — multi-node IB SHARP (no effect single-node)
run_test "collnet_sharp" "all_reduce" \
    env -u NCCL_ALGO NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=1

# Config 4: NVLS + CollNet — NVLS intra-node, IB SHARP inter-node
run_test "nvls_collnet" "all_reduce" \
    env -u NCCL_ALGO NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=1

echo ""
echo "============================================================"
echo "SUMMARY — Peak busBW at largest message size"
echo "============================================================"
printf "%-22s %-18s %14s\n" "Test" "Config" "busBW (GB/s)"
printf "%-22s %-18s %14s\n" "----" "------" "------------"

for outfile in "$RESULTS_DIR"/*.out; do
    fname=$(basename "$outfile" .out)
    label="${fname#all_reduce_}"
    last_line=$(grep -E "^[[:space:]]+[0-9]" "$outfile" 2>/dev/null | tail -1 || true)
    if [[ -n "$last_line" ]]; then
        busbw=$(echo "$last_line" | awk '{print $8}')
        printf "%-22s %-18s %14s\n" "all_reduce" "$label" "$busbw"
    else
        printf "%-22s %-18s %14s\n" "all_reduce" "$label" "N/A (failed)"
    fi
done

echo ""
echo "Notes:"
echo "  ring         — NCCL_ALGO=RING, NVLS=0, CollNet=0  (cluster default)"
echo "  nvls         — NVLS=1, CollNet=0  (single-node NVLink SHARP)"
echo "  collnet_sharp— NVLS=0, CollNet=1  (multi-node IB SHARP)"
echo "  nvls_collnet — NVLS=1, CollNet=1  (NVLS intra-node + IB SHARP inter-node)"
echo ""
echo "Full results: $RESULTS_DIR"
