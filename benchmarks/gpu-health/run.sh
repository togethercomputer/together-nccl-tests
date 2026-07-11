#!/bin/bash
#
# run.sh — Submit GPU health check across all cluster nodes
#
# Usage:
#   ./run.sh           # all nodes
#   ./run.sh 16        # 16 nodes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-batch}"
NNODES="${1:-}"

RESULTS_BASE="${RESULTS_BASE:-/mnt/vast/dgxc-benchmarking-auto/nccl-results}"
GPU_TYPE="${GPU_TYPE:-B200}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_BASE}/${GPU_TYPE}/gpu_health_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Copy script to shared fs (compute nodes can't see /home)
SHARED_SCRIPT="/mnt/vast/dgxc-benchmarking-auto/check_gpu_health.sh"
cp "${SCRIPT_DIR}/check_gpu_health.sh" "$SHARED_SCRIPT"
chmod +x "$SHARED_SCRIPT"

SBATCH_ARGS=(
    --job-name=gpu-health
    --partition="${PARTITION}"
    --ntasks-per-node=1
    --gpus-per-node=8
    --mem=0
    --time=00:10:00
    --exclusive
    --output="${RESULTS_DIR}/gpu_health_%j.out"
)
if [[ -n "$NNODES" ]]; then
    SBATCH_ARGS+=(--nodes="${NNODES}")
fi

echo "=== GPU Health Check ==="
echo "  Partition: ${PARTITION}"
echo "  Nodes:     ${NNODES:-all available}"
echo "  Results:   ${RESULTS_DIR}"

JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" --wrap "
echo 'node,gpu,gfx_mhz,gfx_max,mem_mhz,mem_max,ecc,ecc_corr,ecc_uncorr,temp_c,power_w,flags'
srun bash ${SHARED_SCRIPT}
")

echo "  Submitted job ${JOB_ID}"
echo "  Monitor:  squeue -j ${JOB_ID}"
echo "  Results:  ${RESULTS_DIR}/gpu_health_${JOB_ID}.out"
