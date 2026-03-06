#!/bin/bash
#SBATCH --job-name=nccl_benchmark
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00

# NCCL benchmark runner for Slurm clusters.
#
# Run directly for interactive output, or submit via sbatch for batch use.
# When inside a Slurm job (SLURM_JOB_ID + NCCL_BENCH_CONFIG set), runs one config.

#
# Node selection modes:
#   --node  <name>      Single node
#   --nodes <list>      Comma-separated nodelist
#   --all               All nodes in cluster (sinfo)
#   --exclude <list>    All cluster nodes except given list
#
# Usage:
#   bash slurm_nccl_benchmark.sh --node   gpu-dp-2jvzl-pqq7p
#   bash slurm_nccl_benchmark.sh --nodes  gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs
#   bash slurm_nccl_benchmark.sh --all
#   bash slurm_nccl_benchmark.sh --exclude gpu-dp-2jvzl-ddjjs

set -euo pipefail

BENCHMARKS_DIR="${BENCHMARKS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NCCL_TESTS_DIR="$(dirname "$BENCHMARKS_DIR")"
BUILD_DIR="$NCCL_TESTS_DIR/build"
NGPUS_PER_NODE=8
PARTITION="${PARTITION:-production}"
ACCOUNT="${ACCOUNT:-johnson}"
LOG_DIR="${LOG_DIR:-/data/home/johnson/logs}"

# ── Argument parsing ─────────────────────────────────────────────────────────
MODE="local"
NODELIST=""
EXCLUDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)      MODE="single";  NODELIST="$2"; shift 2 ;;
        --node=*)    MODE="single";  NODELIST="${1#*=}"; shift ;;
        --nodes)     MODE="list";    NODELIST="$2"; shift 2 ;;
        --nodes=*)   MODE="list";    NODELIST="${1#*=}"; shift ;;
        --all)       MODE="all";     shift ;;
        --exclude)   MODE="exclude"; EXCLUDE="$2"; shift 2 ;;
        --exclude=*) MODE="exclude"; EXCLUDE="${1#*=}"; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account)   ACCOUNT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Node resolution ──────────────────────────────────────────────────────────
resolve_nodes() {
    case "$MODE" in
        single|list) echo "$NODELIST" ;;
        all)
            sinfo -h -o "%N" -p "$PARTITION" | head -1 ;;
        exclude)
            ALL=$(sinfo -h -o "%n" -p "$PARTITION" | sort -u)
            EXCL=$(scontrol show hostnames "$EXCLUDE" 2>/dev/null | sort -u)
            REMAINING=$(comm -23 <(echo "$ALL") <(echo "$EXCL"))
            [[ -z "$REMAINING" ]] && { echo "Error: no nodes remain after exclusion." >&2; exit 1; }
            echo "$REMAINING" | paste -sd',' - ;;
        local) echo "$(hostname)" ;;
        "") echo "Error: specify --node, --nodes, --all, or --exclude." &>2; exit 1 ;;
    esac
}

# ── Single-config execution (inside a Slurm job) ────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" && -n "${NCCL_BENCH_CONFIG:-}" ]]; then
    CONFIG="$NCCL_BENCH_CONFIG"
    RESULTS_DIR="$NCCL_BENCH_RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NR==1{print; exit}')
    GPU_NAME="${GPU_NAME:-unknown}"

    OUTFILE="$RESULTS_DIR/all_reduce_${CONFIG}.out"

    SRUN="srun --nodelist=$SLURM_NODELIST \
               --ntasks=$SLURM_NNODES \
               --ntasks-per-node=1 \
               --gres=gpu:$NGPUS_PER_NODE \
               --cpus-per-task=64 \
               --mem=0"

    echo "[$(date)] NCCL benchmark (CLUSTER) started"
    echo "  Nodes       : $SLURM_NNODES ($SLURM_NODELIST)"
    echo "  GPU Type    : $GPU_NAME"
    echo "  Total GPUs  : $(( SLURM_NNODES * NGPUS_PER_NODE ))"
    echo "  Config      : $CONFIG"
    echo "  Output      : $OUTFILE"
    echo ""

    case "$CONFIG" in
        ring)
            export NCCL_ALGO=RING NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=0 ;;
        nvls)
            unset NCCL_ALGO 2>/dev/null || true
            export NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=0 ;;
        collnet_sharp)
            unset NCCL_ALGO 2>/dev/null || true
            export NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=1 ;;
        nvls_collnet)
            unset NCCL_ALGO 2>/dev/null || true
            export NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=1 ;;
        *) echo "Unknown config: $CONFIG"; exit 1 ;;
    esac

    $SRUN "$BUILD_DIR/all_reduce_perf" \
        -b 1M -e 8G -f 2 \
        -g "$NGPUS_PER_NODE" \
        -w 5 -n 20 \
        2>&1 | tee "$OUTFILE"

    # Peak busBW summary for this config
    last=$(grep -E "^[[:space:]]+[0-9]" "$OUTFILE" 2>/dev/null | tail -1 || true)
    [[ -n "$last" ]] && printf "\n>>> Peak busBW [%s]: %s GB/s\n" "$CONFIG" "$(echo "$last" | awk '{print $8}')"
    echo "[$(date)] config=${CONFIG} done"

    # Print overall summary if this is the last config
    if [[ "$CONFIG" == "nvls_collnet" ]]; then
        echo ""
        echo "=== SUMMARY — Peak busBW at largest message size ==="
        echo "============================================================"
        printf "%-22s %-18s %14s\n" "Test" "Config" "busBW (GB/s)"
        printf "%-22s %-18s %14s\n" "----" "------" "------------"
        for f in "$RESULTS_DIR"/all_reduce_*.out; do
            lbl="${f##*/all_reduce_}"; lbl="${lbl%.out}"
            ll=$(grep -E "^[[:space:]]+[0-9]" "$f" 2>/dev/null | tail -1 || true)
            [[ -n "$ll" ]] \
                && printf "%-22s %-18s %14s\n" "all_reduce" "$lbl" "$(echo "$ll" | awk '{print $8}')" \
                || printf "%-22s %-18s %14s\n" "all_reduce" "$lbl" "N/A"
        done
        echo ""
        echo "Full results: $RESULTS_DIR"
    fi
    exit 0
fi

# ── Direct execution: resolve nodes and run each config via srun ─────────────
NODELIST_RESOLVED=$(resolve_nodes)
NNODES=$(scontrol show hostnames "$NODELIST_RESOLVED" | wc -l)
TOTAL_GPUS=$(( NNODES * NGPUS_PER_NODE ))

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NR==1{print; exit}')
GPU_NAME="${GPU_NAME:-unknown}"
if   [[ "$GPU_NAME" == *"H200"* ]];  then GPU_TYPE="H200"
elif [[ "$GPU_NAME" == *"H100"* ]];  then GPU_TYPE="H100"
elif [[ "$GPU_NAME" == *"A100"* ]];  then GPU_TYPE="A100"
elif [[ "$GPU_NAME" == *"B200"* ]];  then GPU_TYPE="B200"
elif [[ "$GPU_NAME" == *"GB200"* ]]; then GPU_TYPE="GB200"
else GPU_TYPE="$(echo "$GPU_NAME" | tr ' ' '_')"; fi

RESULTS_DIR="$BENCHMARKS_DIR/$GPU_TYPE/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [[ "$MODE" == "local" ]]; then
    LAUNCHER=""
else
    LAUNCHER="srun --nodelist=$NODELIST_RESOLVED \
               --ntasks=$NNODES \
               --ntasks-per-node=1 \
               --gres=gpu:$NGPUS_PER_NODE \
               --cpus-per-task=64 \
               --mem=0"
fi

MASTER_ADDR=$(scontrol show hostnames "$NODELIST_RESOLVED" 2>/dev/null | head -1 || hostname)

if [[ "$MODE" == "local" ]]; then
    echo "[$(date)] NCCL benchmark (LOCAL) started"
    echo "  Node        : $(hostname)"
    echo "  GPU Type    : $GPU_TYPE"
    echo "  Total GPUs  : $TOTAL_GPUS"
else
    echo "[$(date)] NCCL benchmark (CLUSTER) started"
    echo "  Nodes       : $NNODES ($NODELIST_RESOLVED)"
    echo "  GPU Type    : $GPU_TYPE"
    echo "  Total GPUs  : $TOTAL_GPUS"
    echo "  Master      : $MASTER_ADDR"
fi
echo "  Results dir : $RESULTS_DIR"
echo ""

run_test() {
    local label="$1"
    local test_name="$2"
    local binary="$BUILD_DIR/${test_name}_perf"
    local outfile="$RESULTS_DIR/${test_name}_${label}.out"
    shift 2

    if [[ ! -x "$binary" ]]; then
        echo "  [SKIP] $binary not found"; return
    fi

    echo ""
    echo "=== ${test_name}_${label} ==="
    # Apply NCCL env vars then run srun directly (not as child of env)
    for kv in "$@"; do export "$kv"; done
    $LAUNCHER "$binary" \
        -b 1M -e 8G -f 2 \
        -g "$NGPUS_PER_NODE" \
        -w 5 -n 20 \
        2>&1 | tee "$outfile"
    # Unset NCCL vars to avoid leaking into next config
    unset NCCL_ALGO NCCL_NVLS_ENABLE NCCL_COLLNET_ENABLE 2>/dev/null || true
}

run_test "ring"          "all_reduce"  NCCL_ALGO=RING NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=0
run_test "nvls"          "all_reduce"  NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=0
run_test "collnet_sharp" "all_reduce"  NCCL_NVLS_ENABLE=0 NCCL_COLLNET_ENABLE=1
run_test "nvls_collnet"  "all_reduce"  NCCL_NVLS_ENABLE=1 NCCL_COLLNET_ENABLE=1

echo ""
echo "=== SUMMARY — Peak busBW at largest message size ==="
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
echo ""
echo "  Full results: $RESULTS_DIR"
echo "[$(date)] NCCL benchmark finished"
