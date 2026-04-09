#!/bin/bash
#
# run_nccl_native.sh — Run NCCL tests via Slurm WITHOUT containers
#                       Uses host HPC-X + NCCL from /opt/hpcx or shared build
#
# Usage:
#   ./run_nccl_native.sh                          # 1 node, all_reduce
#   ./run_nccl_native.sh -n 2 -t all_reduce
#   ./run_nccl_native.sh -n 4 -t all_gather --dry-run
#
set -euo pipefail

# ──────────────────── defaults ────────────────────
NODES=1
GPUS_PER_NODE=8
TEST="all_reduce"
PARTITION="batch"
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
NCCL_TESTS_DIR="/mnt/vast/exemplar/llmb/nccl-tests/build"
EXTRA_ARGS=""
DRY_RUN=0
TIME_LIMIT="00:30:00"
OUTPUT_DIR=""

# ──────────────────── parse args ────────────────────
usage() {
    cat <<'USAGE'
Usage: run_nccl_native.sh [OPTIONS]

Options:
  -n, --nodes NUM          Number of nodes (default: 1)
  -g, --gpus NUM           GPUs per node (default: 8)
  -t, --test NAME          Test name (default: all_reduce)
  -p, --partition NAME     Slurm partition (default: batch)
  -b, --min-bytes SIZE     Min message size (default: 1M)
  -B, --max-bytes SIZE     Max message size (default: 8G)
  -i, --iters NUM          Iterations (default: 20)
  -d, --dir PATH           nccl-tests build dir (default: /mnt/vast/exemplar/llmb/nccl-tests/build)
  -o, --output-dir DIR     Save results to DIR
  -T, --time LIMIT         Slurm time limit (default: 00:30:00)
  -e, --extra ARGS         Extra args for *_perf binary
  --dry-run                Print sbatch script without submitting
  -h, --help               Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--nodes)       NODES="$2";           shift 2 ;;
        -g|--gpus)        GPUS_PER_NODE="$2";   shift 2 ;;
        -t|--test)        TEST="$2";            shift 2 ;;
        -p|--partition)   PARTITION="$2";       shift 2 ;;
        -b|--min-bytes)   MIN_BYTES="$2";       shift 2 ;;
        -B|--max-bytes)   MAX_BYTES="$2";       shift 2 ;;
        -i|--iters)       ITERS="$2";           shift 2 ;;
        -d|--dir)         NCCL_TESTS_DIR="$2";  shift 2 ;;
        -o|--output-dir)  OUTPUT_DIR="$2";      shift 2 ;;
        -T|--time)        TIME_LIMIT="$2";      shift 2 ;;
        -e|--extra)       EXTRA_ARGS="$2";      shift 2 ;;
        --dry-run)        DRY_RUN=1;            shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ──────────────────── derived values ────────────────────
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
BINARY="${NCCL_TESTS_DIR}/${TEST}_perf"
JOB_NAME="nccl-${TEST}-${NODES}n${GPUS_PER_NODE}g"

if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="${OUTPUT_DIR}/${JOB_NAME}_%j.log"
else
    OUTPUT_FILE="${JOB_NAME}_%j.log"
fi

# ──────────────────── build sbatch script ────────────────────
SBATCH_SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${OUTPUT_FILE}
#SBATCH --exclusive

echo "=== NCCL Test: ${TEST} ==="
echo "Nodes: ${NODES}, GPUs/node: ${GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS}"
echo "Date: \$(date)"
echo "Nodelist: \${SLURM_NODELIST}"
echo "==============================="

# Source HPC-X for MPI + UCX
source /opt/hpcx/hpcx-init.sh
hpcx_load

# NCCL environment
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

srun --mpi=pmix \\
    ${BINARY} \\
        -b ${MIN_BYTES} \\
        -e ${MAX_BYTES} \\
        -f ${STEP_FACTOR} \\
        -g 1 \\
        -n ${ITERS} \\
        -w ${WARMUP} \\
        ${EXTRA_ARGS}

echo "=== Done: \$(date) ==="
EOF
)

# ──────────────────── submit or print ────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    echo "--- DRY RUN: sbatch script ---"
    echo "$SBATCH_SCRIPT"
    echo "--- end ---"
else
    TMPFILE=$(mktemp /tmp/nccl-test-XXXXXX.sbatch)
    echo "$SBATCH_SCRIPT" > "$TMPFILE"
    echo "Submitting: $JOB_NAME ($NODES nodes, $TOTAL_GPUS GPUs, test=$TEST)"
    sbatch "$TMPFILE"
    echo "Script saved to: $TMPFILE"
fi
