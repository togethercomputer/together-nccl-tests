#!/bin/bash
#
# run_slurm.sh — NCCL benchmark suite via Slurm
#
# Submits one sbatch job per test config; waits for completion and collects results.
# GPU type is auto-detected and determines the results subdirectory:
#   benchmarks/<GPU_TYPE>/results/<timestamp>_slurm_<N>nodes/
#
# Supports native binaries (default) or Pyxis/enroot containers (--container).
#
# Usage:
#   ./benchmarks/run_slurm.sh --node gpu-node-1
#   ./benchmarks/run_slurm.sh --nodes gpu-node-1,gpu-node-2
#   ./benchmarks/run_slurm.sh --all
#   ./benchmarks/run_slurm.sh --container /path/to/image.sqsh --nodes gpu-node-1,gpu-node-2
#   ./benchmarks/run_slurm.sh --dry-run
#
set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARKS_DIR}/lib/common.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
GPU_TYPE=""
MODE="local"
NODES_ARG=""
GPUS_PER_NODE=8
PARTITION="${PARTITION:-batch}"
ACCOUNT="${ACCOUNT:-}"
TIME_LIMIT="01:00:00"
BINARY_PATH="/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
CONTAINER=""          # empty = native mode (default); set to sqsh/docker URI for Pyxis
NCCL_LIB="/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT="/opt/hpcx/hpcx-init.sh"
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
DRY_RUN=0
BASELINE_DIR=""
SET_BASELINE=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: run_slurm.sh [OPTIONS]

Node selection (pick one, or omit for local/current allocation):
      --node NAME           Single node
      --nodes LIST          Comma-separated nodelist
      --all                 All nodes in partition
      --exclude LIST        All nodes in partition except given list

Options:
      --gpu-type TYPE       GPU type label (default: auto-detect)
      --gpus-per-node N     GPUs per node (default: 8)
      --partition NAME      Slurm partition (default: gpu)
      --account NAME        Slurm account (optional)
      --time HH:MM:SS       Job time limit (default: 01:00:00)
      --binary-path DIR     Directory containing *_perf binaries
      --container PATH      Container image (sqsh or docker URI) for Pyxis/enroot
      --hpcx PATH           Path to hpcx-init.sh (e.g. /opt/hpcx/hpcx-init.sh)
  -b, --min SIZE            Min message size (default: 1M)
  -B, --max SIZE            Max message size (default: 8G)
  -f, --factor NUM          Step factor      (default: 2)
  -i, --iters NUM           Iterations       (default: 20)
  -w, --warmup NUM          Warmup iters     (default: 5)
      --baseline DIR        Baseline dir for comparison
      --set-baseline        Mark this run as new baseline
      --dry-run             Print sbatch scripts without submitting
  -h, --help                Show this help
USAGE
    exit 0
}

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node)          MODE="single";  NODES_ARG="$2";    shift 2 ;;
        --nodes)         MODE="list";    NODES_ARG="$2";    shift 2 ;;
        --all)           MODE="all";                        shift ;;
        --exclude)       MODE="exclude"; NODES_ARG="$2";    shift 2 ;;
        --gpu-type)      GPU_TYPE="$2";                     shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";                shift 2 ;;
        --partition)     PARTITION="$2";                    shift 2 ;;
        --account)       ACCOUNT="$2";                      shift 2 ;;
        --time)          TIME_LIMIT="$2";                   shift 2 ;;
        --binary-path)   BINARY_PATH="$2";                  shift 2 ;;
        --container)     CONTAINER="$2";                    shift 2 ;;
        --hpcx)          HPCX_INIT="$2";                    shift 2 ;;
        -b|--min)        MIN_BYTES="$2";                    shift 2 ;;
        -B|--max)        MAX_BYTES="$2";                    shift 2 ;;
        -f|--factor)     STEP_FACTOR="$2";                  shift 2 ;;
        -i|--iters)      ITERS="$2";                        shift 2 ;;
        -w|--warmup)     WARMUP="$2";                       shift 2 ;;
        --baseline)      BASELINE_DIR="$2";                 shift 2 ;;
        --set-baseline)  SET_BASELINE=1;                    shift ;;
        --dry-run)       DRY_RUN=1;                         shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── resolve nodelist ──────────────────────────────────────────────────────────
resolve_nodes() {
    case "$MODE" in
        single|list) echo "$NODES_ARG" ;;
        all)
            # %n prints one hostname per line across all partition rows/states;
            # sort -u deduplicates, paste joins into a comma-separated nodelist.
            sinfo -h -o "%n" -p "$PARTITION" | sort -u | paste -sd',' ;;
        exclude)
            local all excl remaining
            all=$(sinfo -h -o "%n" -p "$PARTITION" | sort -u)
            excl=$(scontrol show hostnames "$NODES_ARG" 2>/dev/null | sort -u)
            remaining=$(comm -23 <(echo "$all") <(echo "$excl"))
            [[ -z "$remaining" ]] && { echo "ERROR: no nodes remain after exclusion" >&2; exit 1; }
            echo "$remaining" | paste -sd',' - ;;
        local) hostname ;;
    esac
}

NODELIST=$(resolve_nodes)
if [[ "$MODE" == "local" ]]; then
    NNODES=1
else
    NNODES=$(scontrol show hostnames "$NODELIST" 2>/dev/null | wc -l)
fi
TOTAL_PROCS=$(( NNODES * GPUS_PER_NODE ))

# ── GPU type & results dir ────────────────────────────────────────────────────
[[ -z "$GPU_TYPE" ]] && GPU_TYPE=$(detect_gpu_type)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Use shared filesystem for results so compute nodes can write output files
RESULTS_BASE="${RESULTS_BASE:-/mnt/vast/dgxc-benchmarking-auto/nccl-results}"
RESULTS_DIR="${RESULTS_BASE}/${GPU_TYPE}/${TIMESTAMP}_slurm_${NNODES}nodes"
[[ -z "$BASELINE_DIR" ]] && BASELINE_DIR="${RESULTS_BASE}/${GPU_TYPE}/baseline_slurm"
mkdir -p "$RESULTS_DIR"

# ── binary prefix ─────────────────────────────────────────────────────────────
BIN_PREFIX="${BINARY_PATH:+${BINARY_PATH}/}"

echo "======================================================"
echo "  NCCL Benchmark Suite — Slurm"
echo "======================================================"
echo "  GPU:       ${GPU_TYPE}"
local_suffix=""; [[ "$MODE" != "local" ]] && local_suffix=" (${NODELIST})"
echo "  Mode:      ${MODE}${local_suffix}"
echo "  Nodes:     ${NNODES}  |  GPUs/node: ${GPUS_PER_NODE}  |  Total: ${TOTAL_PROCS}"
echo "  Partition: ${PARTITION}"
[[ -n "$CONTAINER" ]] && echo "  Container: ${CONTAINER}" || echo "  Runtime:   native"
echo "  Sizes:     ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:     ${WARMUP} warmup + ${ITERS} measured"
echo "  Results:   ${RESULTS_DIR}"
echo "======================================================"

# For container mode with a docker URI (not a local sqsh file), ensure the
# local host is authenticated so enroot can pull the image.
[[ -n "$CONTAINER" && "$CONTAINER" != *.sqsh ]] && registry_auth_local

TOTAL=${#NCCL_TEST_MATRIX[@]}
PASSED=0
FAILED=0

# ── run one test ──────────────────────────────────────────────────────────────
run_test() {
    local test_name="$1" label="$2" nccl_algo="$3" nvls_enable="$4" collnet_enable="$5"
    local outfile="${RESULTS_DIR}/${test_name}_${label}.out"
    local binary="${BIN_PREFIX}${test_name}_perf"
    local job_name="nccl-${test_name//_/-}-${label//_/-}"

    echo ""
    echo "--- ${test_name} | config=${label} ---"

    # NCCL env lines for the sbatch script
    # Explicitly unset NCCL_ALGO for non-ring configs (cluster may set it in /etc/environment)
    local nccl_env
    if [[ -n "$nccl_algo" ]]; then
        nccl_env="export NCCL_ALGO=${nccl_algo}"
    else
        nccl_env="unset NCCL_ALGO 2>/dev/null || true"
    fi
    nccl_env+=$'\n'"export NCCL_NVLS_ENABLE=${nvls_enable}"
    nccl_env+=$'\n'"export NCCL_COLLNET_ENABLE=${collnet_enable}"
    nccl_env+=$'\n'"export NCCL_DEBUG=WARN"

    # Optional directives
    local nodelist_line="" account_line="" hpcx_line="" container_args=""
    [[ "$MODE" != "local" ]] && nodelist_line="#SBATCH --nodelist=${NODELIST}"
    [[ -n "$ACCOUNT" ]]      && account_line="#SBATCH --account=${ACCOUNT}"
    [[ -n "$HPCX_INIT" ]]    && hpcx_line="source ${HPCX_INIT} && hpcx_load"
    if [[ -n "$CONTAINER" ]]; then
        container_args="--container-image=${CONTAINER} --container-mounts=/run/mellanox:/run/mellanox --no-container-mount-home"
    fi

    local sbatch_script
    sbatch_script=$(cat <<EOF
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
${nodelist_line}
${account_line}

${hpcx_line}
export LD_LIBRARY_PATH=${NCCL_LIB}:\${LD_LIBRARY_PATH}
# Unlock IB memory registration — required for NCCL IB/NVLS at scale.
# No-op if the cluster enforces the limit; works automatically once admin sets
# LimitMEMLOCK=infinity in the Slurm cgroup config.
ulimit -l unlimited 2>/dev/null || true
# Route MPI/UCX bootstrap through shared-mem + TCP (IB requires ulimit -l unlimited,
# blocked in K8s pods). NCCL owns actual data movement via NVLink/IB directly.
export UCX_TLS=self,sm,cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_btl_tcp_if_include=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=300
export CUDA_DEVICE_MAX_CONNECTIONS=32
# Restrict to IB/SHARP HCAs only; mlx5_8 absent, mlx5_13 is RoCE — both excluded.
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1"

${nccl_env}

srun --mpi=pmix ${container_args} ${binary} \
    -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} \
    -g 1 -n ${ITERS} -w ${WARMUP}
EOF
)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "$sbatch_script"
        echo "(dry-run: output would go to ${outfile})"
        return
    fi

    local tmp_script
    tmp_script=$(mktemp /tmp/nccl-sbatch-XXXXXX.sh)
    echo "$sbatch_script" > "$tmp_script"

    echo "  Submitting..."
    local job_id
    job_id=$(sbatch --wait --parsable "$tmp_script")
    rm -f "$tmp_script"
    echo "  Job ${job_id} finished."

    if grep -q "Out of bounds values : 0 OK" "${outfile}" 2>/dev/null; then
        echo "  => Peak busBW: $(peak_busbw "${outfile}") GB/s"
        PASSED=$((PASSED + 1))
    else
        echo "  ERROR: Job failed or produced no valid output"
        FAILED=$((FAILED + 1))
    fi
}

# ── run all tests ─────────────────────────────────────────────────────────────
IDX=0
for entry in "${NCCL_TEST_MATRIX[@]}"; do
    IDX=$((IDX + 1))
    IFS=':' read -r test_name label nccl_algo nvls_enable collnet_enable <<< "$entry"
    echo ""
    echo "[${IDX}/${TOTAL}] ${test_name} / ${label}"
    run_test "$test_name" "$label" "$nccl_algo" "$nvls_enable" "$collnet_enable" || true
done

# ── summary ───────────────────────────────────────────────────────────────────
[[ $DRY_RUN -eq 0 ]] && print_summary "${RESULTS_DIR}" "${BASELINE_DIR}"

echo ""
echo "Passed: ${PASSED} / ${TOTAL}  |  Failed: ${FAILED}"
echo "Results: ${RESULTS_DIR}"
[[ -d "${BASELINE_DIR}" ]] && echo "Baseline: ${BASELINE_DIR}"

if [[ $DRY_RUN -eq 0 && $SET_BASELINE -eq 1 ]]; then
    ln -sfn "$(basename "${RESULTS_DIR}")" \
        "${RESULTS_BASE}/${GPU_TYPE}/baseline_slurm"
    echo "Baseline updated -> ${RESULTS_DIR}"
fi
