#!/bin/bash
#
# run_mpi.sh — NCCL benchmark suite via direct MPI (no job scheduler required)
#
# Runs all tests using mpirun across one or more nodes.
# GPU type is auto-detected and determines the results subdirectory:
#   benchmarks/<GPU_TYPE>/results/<timestamp>_mpi_<N>nodes/
#
# Usage:
#   ./benchmarks/run_mpi.sh                          # single-node localhost
#   ./benchmarks/run_mpi.sh --hostfile hosts.txt     # multi-node
#   ./benchmarks/run_mpi.sh --dry-run
#
# Hostfile format (OpenMPI):
#   node1 slots=8
#   node2 slots=8
#
set -euo pipefail

BENCHMARKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${BENCHMARKS_DIR}/lib/common.sh"

# ── defaults ──────────────────────────────────────────────────────────────────
GPU_TYPE=""
HOSTFILE=""
NODES=1
GPUS_PER_NODE=8
MIN_BYTES="1M"
MAX_BYTES="8G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
MPIRUN_EXTRA=""
DRY_RUN=0
BASELINE_DIR=""
SET_BASELINE=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: run_mpi.sh [OPTIONS]

Options:
      --gpu-type TYPE       GPU type label (default: auto-detect)
      --hostfile FILE       MPI hostfile (omit for single-node localhost run)
      --gpus-per-node N     GPUs per node (default: 8)
  -b, --min SIZE            Min message size (default: 1M)
  -B, --max SIZE            Max message size (default: 8G)
  -f, --factor NUM          Step factor      (default: 2)
  -i, --iters NUM           Iterations       (default: 20)
  -w, --warmup NUM          Warmup iters     (default: 5)
      --mpirun-extra X      Extra args passed to mpirun
      --baseline DIR        Baseline dir for comparison
      --set-baseline        Mark this run as new baseline
      --dry-run             Print commands without running
  -h, --help                Show this help

Hostfile format (OpenMPI):
  node1 slots=8
  node2 slots=8
USAGE
    exit 0
}

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-type)      GPU_TYPE="$2";       shift 2 ;;
        --hostfile)      HOSTFILE="$2";       shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
        -b|--min)        MIN_BYTES="$2";      shift 2 ;;
        -B|--max)        MAX_BYTES="$2";      shift 2 ;;
        -f|--factor)     STEP_FACTOR="$2";    shift 2 ;;
        -i|--iters)      ITERS="$2";          shift 2 ;;
        -w|--warmup)     WARMUP="$2";         shift 2 ;;
        --mpirun-extra)  MPIRUN_EXTRA="$2";   shift 2 ;;
        --baseline)      BASELINE_DIR="$2";   shift 2 ;;
        --set-baseline)  SET_BASELINE=1;      shift ;;
        --dry-run)       DRY_RUN=1;           shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── validate hostfile & derive node count ────────────────────────────────────
if [[ -n "$HOSTFILE" ]]; then
    [[ ! -f "$HOSTFILE" ]] && { echo "ERROR: Hostfile not found: ${HOSTFILE}"; exit 1; }
    NODES=$(grep -cE '^[^#[:space:]]' "$HOSTFILE" || true)
    [[ "$NODES" -eq 0 ]] && { echo "ERROR: No valid hosts in ${HOSTFILE}"; exit 1; }
fi

TOTAL_PROCS=$(( NODES * GPUS_PER_NODE ))

# ── Registry auth on all nodes (multi-node only) ─────────────────────────────
if [[ -n "$HOSTFILE" && $DRY_RUN -eq 0 ]]; then
    registry_auth_remote "$HOSTFILE"
fi

# ── SSH connectivity check (multi-node only) ─────────────────────────────────
check_ssh() {
    local failed_hosts=()
    local host
    while IFS= read -r line; do
        host=$(echo "$line" | awk '{print $1}')
        [[ -z "$host" || "$host" == \#* ]] && continue
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" exit 2>/dev/null \
            || failed_hosts+=("$host")
    done < "$HOSTFILE"
    if [[ ${#failed_hosts[@]} -gt 0 ]]; then
        echo "WARNING: SSH failed for: ${failed_hosts[*]}"
        echo "         Ensure passwordless SSH is configured."
        echo "         Proceeding in 5s — Ctrl-C to abort."
        sleep 5
    else
        echo "  All hosts reachable."
    fi
}

if [[ -n "$HOSTFILE" && $DRY_RUN -eq 0 ]]; then
    echo "Checking SSH connectivity..."
    check_ssh
fi

# ── GPU type & results dir ────────────────────────────────────────────────────
[[ -z "$GPU_TYPE" ]] && GPU_TYPE=$(detect_gpu_type)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${BENCHMARKS_DIR}/${GPU_TYPE}/results/${TIMESTAMP}_mpi_${NODES}nodes"
[[ -z "$BASELINE_DIR" ]] && BASELINE_DIR="${BENCHMARKS_DIR}/${GPU_TYPE}/results/baseline_mpi"
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "  NCCL Benchmark Suite — MPI"
echo "======================================================"
echo "  GPU:       ${GPU_TYPE}"
echo "  Nodes:     ${NODES}  |  GPUs/node: ${GPUS_PER_NODE}  |  Total: ${TOTAL_PROCS} ranks"
[[ -n "$HOSTFILE" ]] && echo "  Hostfile:  ${HOSTFILE}" || echo "  Hostfile:  (none — localhost)"
echo "  Sizes:     ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:     ${WARMUP} warmup + ${ITERS} measured"
echo "  Results:   ${RESULTS_DIR}"
echo "======================================================"

TOTAL=${#NCCL_TEST_MATRIX[@]}
PASSED=0
FAILED=0

# ── run one test ──────────────────────────────────────────────────────────────
run_test() {
    local test_name="$1" label="$2" nccl_algo="$3" nvls_enable="$4" collnet_enable="$5"
    local outfile="${RESULTS_DIR}/${test_name}_${label}.out"
    local binary="${test_name}_perf"

    echo ""
    echo "--- ${test_name} | config=${label} ---"

    # For non-ring configs, prefix with env -u NCCL_ALGO so that a globally-set
    # NCCL_ALGO=RING (e.g. from /etc/environment) does not leak into mpirun ranks.
    local env_prefix=""
    [[ -z "$nccl_algo" ]] && env_prefix="env -u NCCL_ALGO "

    local cmd="${env_prefix}mpirun -np ${TOTAL_PROCS} -N ${GPUS_PER_NODE}"
    [[ -n "$HOSTFILE" ]]  && cmd+=" --hostfile ${HOSTFILE}"
    [[ -n "$nccl_algo" ]] && cmd+=" -x NCCL_ALGO=${nccl_algo}"
    cmd+=" -x NCCL_NVLS_ENABLE=${nvls_enable}"
    cmd+=" -x NCCL_COLLNET_ENABLE=${collnet_enable}"
    cmd+=" -x NCCL_DEBUG=WARN"
    [[ -n "$MPIRUN_EXTRA" ]] && cmd+=" ${MPIRUN_EXTRA}"
    cmd+=" ${binary} -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR}"
    cmd+=" -g 1 -n ${ITERS} -w ${WARMUP}"

    echo "  CMD: ${cmd}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry-run: would save to ${outfile})"
        return
    fi

    if eval "$cmd" 2>&1 | tee "${outfile}"; then
        if grep -q "Out of bounds values : 0 OK" "${outfile}" 2>/dev/null; then
            echo "  => Peak busBW: $(peak_busbw "${outfile}") GB/s"
            PASSED=$((PASSED + 1))
        else
            echo "  ERROR: Test produced no valid output"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ERROR: mpirun exited non-zero"
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
        "${BENCHMARKS_DIR}/${GPU_TYPE}/results/baseline_mpi"
    echo "Baseline updated -> ${RESULTS_DIR}"
fi
