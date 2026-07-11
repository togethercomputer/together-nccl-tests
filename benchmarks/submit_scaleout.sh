#!/bin/bash
#
# submit_scaleout.sh — Submit all NCCL scale-out benchmarks in parallel
#
# Submits 9 tests × N node-counts as independent sbatch jobs (no --wait).
# Slurm handles scheduling; smaller jobs run immediately, larger ones queue.
#
set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARKS_DIR}/lib/common.sh"

# ── config ───────────────────────────────────────────────────────────────────
NODE_COUNTS=(${@:-2 4 8 16 32 64})
GPUS_PER_NODE=8
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="01:00:00"
BINARY_PATH="/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB="/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT="/opt/hpcx/hpcx-init.sh"
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5

# ── GPU type & results base ─────────────────────────────────────────────────
GPU_TYPE=$(detect_gpu_type)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE="${RESULTS_BASE:-/mnt/vast/dgxc-benchmarking-auto/nccl-results}"

echo "======================================================"
echo "  NCCL Scale-Out Benchmark — Batch Submit"
echo "======================================================"
echo "  GPU:         ${GPU_TYPE}"
echo "  Node counts: ${NODE_COUNTS[*]}"
echo "  Tests/scale: ${#NCCL_TEST_MATRIX[@]}"
echo "  Total jobs:  $(( ${#NODE_COUNTS[@]} * ${#NCCL_TEST_MATRIX[@]} ))"
echo "  Sizes:       ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:       ${WARMUP} warmup + ${ITERS} measured"
echo "  Timestamp:   ${TIMESTAMP}"
echo "======================================================"

TOTAL_SUBMITTED=0

for NNODES in "${NODE_COUNTS[@]}"; do
    RESULTS_DIR="${RESULTS_BASE}/${GPU_TYPE}/${TIMESTAMP}_slurm_${NNODES}nodes"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "=== ${NNODES} nodes (${GPUS_PER_NODE} GPUs/node = $(( NNODES * GPUS_PER_NODE )) GPUs) ==="
    echo "    Results: ${RESULTS_DIR}"

    for entry in "${NCCL_TEST_MATRIX[@]}"; do
        IFS=':' read -r test_name label nccl_algo nvls_enable collnet_enable <<< "$entry"

        outfile="${RESULTS_DIR}/${test_name}_${label}.out"
        binary="${BINARY_PATH}/${test_name}_perf"
        job_name="nccl-${NNODES}n-${test_name//_/-}-${label//_/-}"

        # NCCL env
        local_nccl_env=""
        if [[ -n "$nccl_algo" ]]; then
            local_nccl_env="export NCCL_ALGO=${nccl_algo}"
        else
            local_nccl_env="unset NCCL_ALGO 2>/dev/null || true"
        fi
        local_nccl_env+=$'\n'"export NCCL_NVLS_ENABLE=${nvls_enable}"
        local_nccl_env+=$'\n'"export NCCL_COLLNET_ENABLE=${collnet_enable}"
        local_nccl_env+=$'\n'"export NCCL_DEBUG=WARN"

        tmp_script=$(mktemp /tmp/nccl-sbatch-XXXXXX.sh)
        cat > "$tmp_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=${PARTITION}
#SBATCH --output=${outfile}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclusive
#SBATCH --chdir=/tmp

source ${HPCX_INIT} && hpcx_load
export LD_LIBRARY_PATH=${NCCL_LIB}:\${LD_LIBRARY_PATH}

${local_nccl_env}

srun --mpi=pmix ${binary} \\
    -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} \\
    -g 1 -n ${ITERS} -w ${WARMUP}
EOF

        job_id=$(sbatch --parsable "$tmp_script")
        rm -f "$tmp_script"
        echo "    [${test_name}/${label}] Job ${job_id}"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    done
done

echo ""
echo "======================================================"
echo "  Submitted ${TOTAL_SUBMITTED} jobs"
echo "  Monitor:  squeue -u $USER"
echo "  Results:  ${RESULTS_BASE}/${GPU_TYPE}/${TIMESTAMP}_slurm_*"
echo "======================================================"
echo ""
echo "TIMESTAMP=${TIMESTAMP}"
