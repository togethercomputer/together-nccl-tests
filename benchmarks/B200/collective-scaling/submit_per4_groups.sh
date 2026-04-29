#!/bin/bash
#
# submit_per4_groups.sh — NCCL all_reduce on disjoint 4-node groups, in parallel.
#
# Pulls idle nodes from the batch partition (or accepts an explicit --nodes
# hostlist) and partitions them deterministically into groups of GROUP_SIZE=4.
# One sbatch job per group, pinned via --nodelist, so every group is a known
# 4-node sub-fabric. Goal: stress every 4-node group in parallel; identify any
# group that underperforms (and therefore which node(s) within it).
#
# Usage:
#   ./submit_per4_groups.sh                       # 16 groups from current idle batch nodes
#   ./submit_per4_groups.sh --algo nvls           # alternate NCCL algo (ring|nvls|tree|nvls_tree|collnet_sharp|auto)
#   ./submit_per4_groups.sh --nodes <hostlist>    # explicit Slurm hostlist
#   ./submit_per4_groups.sh --exclude <hostlist>  # subtract these from idle pool
#   ./submit_per4_groups.sh --sizes "1M 8G 2"     # MIN MAX FACTOR
#   ./submit_per4_groups.sh --dry-run
#
# Mirrors env / HPC-X / NCCL_IB_HCA / bond0 setup from submit_all.sh in this dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config ───────────────────────────────────────────────────────────────────
PARTITION="batch"
GPUS_PER_NODE=8
GROUP_SIZE=4
TIME_LIMIT="00:20:00"
BINARY_PATH="/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB="/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT="/var/tmp/hpcx-2.18/hpcx-init.sh"  # local-disk copy; falls back to /mnt/vast then /opt/hpcx
MIN_BYTES="2G"
MAX_BYTES="16G"
STEP_FACTOR=2
ITERS=20
WARMUP=5

ALGO="ring"           # ring is the default — most fabric-sensitive, no SHARP dep
NODES_ARG=""
EXCLUDE_ARG=""
DRY_RUN=0

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1;        shift ;;
        --algo)        ALGO="$2";        shift 2 ;;
        --nodes)       NODES_ARG="$2";   shift 2 ;;
        --exclude)     EXCLUDE_ARG="$2"; shift 2 ;;
        --group-size)  GROUP_SIZE="$2";  shift 2 ;;
        --time)        TIME_LIMIT="$2";  shift 2 ;;
        --sizes)       read -r MIN_BYTES MAX_BYTES STEP_FACTOR <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Resolve nodelist ─────────────────────────────────────────────────────────
if [[ -n "$NODES_ARG" ]]; then
    RAW_NODELIST="$NODES_ARG"
    mapfile -t HOSTS < <(scontrol show hostnames "$RAW_NODELIST")
else
    # `sinfo -t idle` lumps drained nodes into the idle bucket; filter on exact STATE column.
    mapfile -t HOSTS < <(sinfo -N -p "${PARTITION}" -h -o "%N %t" | awk '$2=="idle"{print $1}' | sort -u)
    if (( ${#HOSTS[@]} == 0 )); then
        echo "ERROR: no truly-idle nodes in partition ${PARTITION}" >&2; exit 1
    fi
fi

# Apply --exclude
if [[ -n "$EXCLUDE_ARG" ]]; then
    mapfile -t EXCL < <(scontrol show hostnames "$EXCLUDE_ARG")
    declare -A EXCLMAP=()
    for h in "${EXCL[@]}"; do EXCLMAP[$h]=1; done
    FILTERED=()
    for h in "${HOSTS[@]}"; do
        [[ -n "${EXCLMAP[$h]:-}" ]] || FILTERED+=("$h")
    done
    HOSTS=("${FILTERED[@]}")
fi

TOTAL_NODES=${#HOSTS[@]}
if (( TOTAL_NODES < GROUP_SIZE )); then
    echo "ERROR: only ${TOTAL_NODES} nodes available (need >= ${GROUP_SIZE})" >&2; exit 1
fi

NGROUPS=$(( TOTAL_NODES / GROUP_SIZE ))
USED=$(( NGROUPS * GROUP_SIZE ))
LEFTOVER=$(( TOTAL_NODES - USED ))

# ── Map --algo → NCCL env ────────────────────────────────────────────────────
case "$ALGO" in
    ring)          NCCL_ALGO_ENV="Ring";     NVLS=0; COLLNET=0 ;;
    nvls)          NCCL_ALGO_ENV="";         NVLS=1; COLLNET=0 ;;
    tree)          NCCL_ALGO_ENV="Tree";     NVLS=0; COLLNET=0 ;;
    nvls_tree)    NCCL_ALGO_ENV="NVLSTree"; NVLS=1; COLLNET=0 ;;
    collnet_sharp) NCCL_ALGO_ENV="";         NVLS=0; COLLNET=1 ;;
    auto)          NCCL_ALGO_ENV="";         NVLS=1; COLLNET=1 ;;
    *) echo "Unknown --algo: ${ALGO} (valid: ring|nvls|tree|nvls_tree|collnet_sharp|auto)" >&2; exit 1 ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/per${GROUP_SIZE}-groups/${TIMESTAMP}_${ALGO}"
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "  NCCL All-Reduce — disjoint ${GROUP_SIZE}-node groups in parallel"
echo "======================================================"
echo "  Algo:        ${ALGO}"
echo "  Total nodes: ${TOTAL_NODES}"
echo "  Groups:      ${NGROUPS} of ${GROUP_SIZE} (leftover: ${LEFTOVER})"
echo "  Sizes:       ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:       ${WARMUP}w + ${ITERS}m"
echo "  Time limit:  ${TIME_LIMIT}"
echo "  Results:     ${RESULTS_DIR}"
echo "  Dry run:     ${DRY_RUN}"
echo "======================================================"
echo ""

JOB_IDS=()
GROUP_NODES_LIST=()
SUBMITTED=0

for ((g=0; g<NGROUPS; g++)); do
    start=$(( g * GROUP_SIZE ))
    GROUP_HOSTS=( "${HOSTS[@]:$start:$GROUP_SIZE}" )
    GROUP_LIST=$(IFS=','; echo "${GROUP_HOSTS[*]}")
    GROUP_NODES_LIST+=("$GROUP_LIST")

    GID=$(printf "%02d" $((g + 1)))
    job_name="nccl-ar-${ALGO}-grp${GID}"
    outfile="${RESULTS_DIR}/group${GID}_allreduce_${ALGO}.out"
    binary="${BINARY_PATH}/all_reduce_perf"

    # Build NCCL env block
    nccl_env=""
    if [[ -n "$NCCL_ALGO_ENV" ]]; then
        nccl_env+="export NCCL_ALGO=${NCCL_ALGO_ENV}"$'\n'
    else
        nccl_env+="unset NCCL_ALGO 2>/dev/null || true"$'\n'
    fi
    nccl_env+="export NCCL_NVLS_ENABLE=${NVLS}"$'\n'
    nccl_env+="export NCCL_COLLNET_ENABLE=${COLLNET}"$'\n'
    nccl_env+="export NCCL_DEBUG=WARN"$'\n'
    nccl_env+="export NCCL_TIMEOUT=300"$'\n'
    nccl_env+="export NCCL_SOCKET_IFNAME=bond0"$'\n'
    nccl_env+='export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"'$'\n'
    nccl_env+="export UCX_NET_DEVICES=bond0"$'\n'
    nccl_env+="export OMPI_MCA_btl_tcp_if_include=bond0"$'\n'
    nccl_env+="export CUDA_DEVICE_MAX_CONNECTIONS=32"

    sbatch_script="#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${GROUP_SIZE}
#SBATCH --ntasks-per-node=${GPUS_PER_NODE}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=${PARTITION}
#SBATCH --output=${outfile}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclusive
#SBATCH --chdir=/tmp
#SBATCH --nodelist=${GROUP_LIST}

if [ -f ${HPCX_INIT} ]; then
    source ${HPCX_INIT} && hpcx_load
elif [ -f /mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/hpcx-init.sh ]; then
    source /mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/hpcx-init.sh && hpcx_load
elif [ -f /opt/hpcx/hpcx-init.sh ]; then
    source /opt/hpcx/hpcx-init.sh && hpcx_load
else
    echo \"ERROR: no hpcx-init.sh found on \$(hostname)\" >&2; exit 127
fi
export LD_LIBRARY_PATH=${NCCL_LIB}:\${LD_LIBRARY_PATH:-}

${nccl_env}

echo \"=== NCCL all_reduce | algo=${ALGO} | group ${GID} (${GROUP_SIZE} nodes / $((GROUP_SIZE * GPUS_PER_NODE)) GPUs) ===\"
echo \"Job ID: \${SLURM_JOB_ID}\"
echo \"Nodes:  ${GROUP_LIST}\"
echo \"Actual: \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\\n' ' ')\"
echo \"Date:   \$(date)\"
echo \"NCCL_ALGO=\${NCCL_ALGO:-<unset>} NCCL_NVLS_ENABLE=${NVLS} NCCL_COLLNET_ENABLE=${COLLNET}\"
echo \"\"

srun --mpi=pmix ${binary} \\
    -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} \\
    -g 1 -n ${ITERS} -w ${WARMUP}

echo \"\"
echo \"=== Done: group ${GID} ===\"
"

    if (( DRY_RUN )); then
        echo "[grp ${GID}] ${GROUP_LIST}  (dry-run)"
    else
        tmp_script=$(mktemp /tmp/nccl-per4-XXXXXX.sh)
        echo "$sbatch_script" > "$tmp_script"
        job_id=$(sbatch --parsable "$tmp_script")
        rm -f "$tmp_script"
        JOB_IDS+=("$job_id")
        SUBMITTED=$((SUBMITTED + 1))
        echo "[grp ${GID}] ${GROUP_LIST} -> Job ${job_id}"
    fi
done

# ── Manifest ─────────────────────────────────────────────────────────────────
if (( DRY_RUN == 0 )); then
    manifest="${RESULTS_DIR}/manifest.txt"
    {
        echo "# NCCL All-Reduce per-${GROUP_SIZE}-node groups — ${TIMESTAMP}"
        echo "# Algo:       ${ALGO}"
        echo "# Group size: ${GROUP_SIZE}"
        echo "# Sizes:      ${MIN_BYTES}-${MAX_BYTES} factor=${STEP_FACTOR}"
        echo "# Iters:      ${WARMUP}w + ${ITERS}m"
        echo "# Total idle: ${TOTAL_NODES} (used ${USED}, leftover ${LEFTOVER})"
        if [[ -n "$EXCLUDE_ARG" ]]; then
            echo "# Exclude:    ${EXCLUDE_ARG}"
        fi
        echo "#"
        echo "# group  job_id  nodes                       outfile"
        for ((g=0; g<NGROUPS; g++)); do
            GID=$(printf "%02d" $((g + 1)))
            echo "${GID}  ${JOB_IDS[$g]}  ${GROUP_NODES_LIST[$g]}  group${GID}_allreduce_${ALGO}.out"
        done
    } > "$manifest"

    if (( LEFTOVER > 0 )); then
        echo ""
        echo "Note: ${LEFTOVER} node(s) not used (not divisible by ${GROUP_SIZE}):"
        echo "  ${HOSTS[@]:$USED:$LEFTOVER}"
    fi

    echo ""
    echo "======================================================"
    echo "  Submitted: ${SUBMITTED} jobs"
    if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
        echo "  Job IDs:   ${JOB_IDS[0]}..${JOB_IDS[-1]}"
    fi
    echo "  Results:   ${RESULTS_DIR}"
    echo "  Manifest:  ${manifest}"
    echo ""
    echo "  Monitor:   squeue -u $USER -n nccl-ar-${ALGO}"
    echo "  Wait:      until [[ \$(squeue -u $USER -n nccl-ar-${ALGO} -h | wc -l) -eq 0 ]]; do sleep 30; done"
    echo "======================================================"
fi
