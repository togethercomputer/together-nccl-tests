#!/bin/bash
#
# submit_all.sh — Submit NCCL collective scaling benchmarks
#
# Test matrix:
#   Nodes:      1, 2, 4, 8, 16, 32, 64
#   Operations: all_reduce, all_gather, sendrecv (P2P)
#   Algorithms: ring, nvls, tree, nvls_tree, collnet_sharp (sendrecv uses defaults only)
#   Messages:   2G, 4G, 8G, 16G (min=2G, max=16G, factor=2)
#
# 1-node jobs skip tree, nvls_tree, collnet_sharp (inter-node only).
# All jobs submitted without --wait; Slurm schedules them.
#
# Usage:
#   ./submit_all.sh                     # submit all 64 jobs
#   ./submit_all.sh --dry-run           # print sbatch scripts only
#   ./submit_all.sh --nodes "2 4 8"     # subset of node counts
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Config ───────────────────────────────────────────────────────────────────
PARTITION="batch"
EXCLUDE="use3a-ss-b200-gpu-[190,197,199,201,211,233,239]"
GPUS_PER_NODE=8
TIME_LIMIT="00:20:00"
BINARY_PATH="/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB="/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT="/var/tmp/hpcx-2.18/hpcx-init.sh"  # local-disk copy; stage with /tmp/stage-hpcx.sh first. Falls back to /mnt/vast copy at runtime.
MIN_BYTES="2G"
MAX_BYTES="16G"
STEP_FACTOR=2
ITERS=20
WARMUP=5

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/${TIMESTAMP}"

NODE_COUNTS=(1 2 4 8 16 32 64)
OPS=(all_reduce all_gather sendrecv)

# algo definitions: "label:NCCL_ALGO_VALUE:NCCL_NVLS_ENABLE:NCCL_COLLNET_ENABLE:multi_node_only"
ALGOS=(
    "ring:Ring:0:0:no"
    "nvls::1:0:no"
    "tree:Tree:0:0:yes"
    "nvls_tree:NVLSTree:1:0:yes"
    "collnet_sharp::0:1:yes"
)

DRY_RUN=0
CUSTOM_NODES=""

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;       shift ;;
        --nodes)     CUSTOM_NODES="$2"; shift 2 ;;
        --exclude)   EXCLUDE="$2";     shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -n "$CUSTOM_NODES" ]]; then
    read -ra NODE_COUNTS <<< "$CUSTOM_NODES"
fi

mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "  NCCL Collective Scaling Benchmark"
echo "======================================================"
echo "  Nodes:     ${NODE_COUNTS[*]}"
echo "  Ops:       ${OPS[*]}"
echo "  Algos:     ring nvls tree nvls_tree collnet_sharp"
echo "  Sizes:     ${MIN_BYTES} -> ${MAX_BYTES} (factor ${STEP_FACTOR})"
echo "  Iters:     ${WARMUP} warmup + ${ITERS} measured"
echo "  Exclude:   ${EXCLUDE}"
echo "  Results:   ${RESULTS_DIR}"
echo "  Dry run:   ${DRY_RUN}"
echo "======================================================"
echo ""

TOTAL=0
SUBMITTED=0
SKIPPED=0
JOB_IDS=()

for NNODES in "${NODE_COUNTS[@]}"; do
    for op in "${OPS[@]}"; do
        for algo_entry in "${ALGOS[@]}"; do
            IFS=':' read -r label nccl_algo nvls_enable collnet_enable multi_only <<< "$algo_entry"

            # sendrecv P2P: skip algo loop, run once with defaults
            if [[ "$op" == "sendrecv" && "$label" != "ring" ]]; then
                continue
            fi

            # Skip inter-node-only algorithms on single node
            if [[ "$NNODES" -eq 1 && "$multi_only" == "yes" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            TOTAL=$((TOTAL + 1))

            if [[ "$op" == "sendrecv" ]]; then
                job_name="nccl-sendrecv-p2p-${NNODES}n"
                outfile="${RESULTS_DIR}/sendrecv_p2p_${NNODES}nodes.out"
                binary="${BINARY_PATH}/sendrecv_perf"
            else
                job_name="nccl-${op}-${label}-${NNODES}n"
                outfile="${RESULTS_DIR}/${op}_${label}_${NNODES}nodes.out"
                binary="${BINARY_PATH}/${op}_perf"
            fi

            # Build NCCL env block
            nccl_env=""
            if [[ "$op" == "sendrecv" ]]; then
                nccl_env+="unset NCCL_ALGO 2>/dev/null || true"$'\n'
                nccl_env+="export NCCL_NVLS_ENABLE=0"$'\n'
                nccl_env+="export NCCL_COLLNET_ENABLE=0"$'\n'
            elif [[ -n "$nccl_algo" ]]; then
                nccl_env+="export NCCL_ALGO=${nccl_algo}"$'\n'
                nccl_env+="export NCCL_NVLS_ENABLE=${nvls_enable}"$'\n'
                nccl_env+="export NCCL_COLLNET_ENABLE=${collnet_enable}"$'\n'
            else
                nccl_env+="unset NCCL_ALGO 2>/dev/null || true"$'\n'
                nccl_env+="export NCCL_NVLS_ENABLE=${nvls_enable}"$'\n'
                nccl_env+="export NCCL_COLLNET_ENABLE=${collnet_enable}"$'\n'
            fi
            nccl_env+="export NCCL_DEBUG=WARN"$'\n'
            nccl_env+="export NCCL_TIMEOUT=300"$'\n'
            nccl_env+="export NCCL_SOCKET_IFNAME=bond0"$'\n'
            nccl_env+='export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"'$'\n'
            nccl_env+="export UCX_NET_DEVICES=bond0"$'\n'
            nccl_env+="export OMPI_MCA_btl_tcp_if_include=bond0"$'\n'
            nccl_env+="export CUDA_DEVICE_MAX_CONNECTIONS=32"

            sbatch_script="#!/bin/bash
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
#SBATCH --exclude=${EXCLUDE}

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

echo \"=== NCCL ${op} | algo=${label:-p2p} | ${NNODES} nodes ($((NNODES * GPUS_PER_NODE)) GPUs) ===\"
echo \"Job ID: \${SLURM_JOB_ID}\"
echo \"Nodes: \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\\n' ' ')\"
echo \"Date: \$(date)\"
echo \"NCCL_ALGO=\${NCCL_ALGO:-<unset>} NCCL_NVLS_ENABLE=${nvls_enable} NCCL_COLLNET_ENABLE=${collnet_enable}\"
echo \"\"

srun --mpi=pmix ${binary} \\
    -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} \\
    -g 1 -n ${ITERS} -w ${WARMUP}

echo \"\"
echo \"=== Done: ${op} ${label} ${NNODES}n ===\"
"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[${TOTAL}] ${job_name} (dry-run)"
                echo "$sbatch_script"
                echo "---"
            else
                tmp_script=$(mktemp /tmp/nccl-sbatch-XXXXXX.sh)
                echo "$sbatch_script" > "$tmp_script"
                job_id=$(sbatch --parsable "$tmp_script")
                rm -f "$tmp_script"
                JOB_IDS+=("$job_id")
                SUBMITTED=$((SUBMITTED + 1))
                echo "[${SUBMITTED}/${TOTAL}] ${job_name} -> Job ${job_id}"
            fi
        done
    done
done

echo ""
echo "======================================================"
echo "  Submitted: ${SUBMITTED}  |  Skipped: ${SKIPPED} (1-node inter-node algos)"
echo "  Total jobs: ${TOTAL}"
echo "  Results:    ${RESULTS_DIR}"
if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    echo "  Job IDs:    ${JOB_IDS[0]}..${JOB_IDS[-1]}"
    echo ""
    echo "  Monitor:  squeue -u $USER"
    echo "  Wait all: squeue -u $USER -h | wc -l"
fi
echo "======================================================"

# Save job manifest for later analysis
if [[ $DRY_RUN -eq 0 ]]; then
    manifest="${RESULTS_DIR}/manifest.txt"
    echo "# NCCL Collective Scaling Benchmark — ${TIMESTAMP}" > "$manifest"
    echo "# Nodes: ${NODE_COUNTS[*]}" >> "$manifest"
    echo "# Ops: ${OPS[*]}" >> "$manifest"
    echo "# Algos: ring nvls tree nvls_tree collnet_sharp" >> "$manifest"
    echo "# Sizes: ${MIN_BYTES}-${MAX_BYTES} factor=${STEP_FACTOR}" >> "$manifest"
    echo "# Iters: ${WARMUP}w + ${ITERS}i" >> "$manifest"
    echo "# Exclude: ${EXCLUDE}" >> "$manifest"
    echo "#" >> "$manifest"
    echo "# job_id  op  algo  nodes  outfile" >> "$manifest"
    idx=0
    for NNODES in "${NODE_COUNTS[@]}"; do
        for op in "${OPS[@]}"; do
            for algo_entry in "${ALGOS[@]}"; do
                IFS=':' read -r label _ _ _ multi_only <<< "$algo_entry"
                [[ "$op" == "sendrecv" && "$label" != "ring" ]] && continue
                [[ "$NNODES" -eq 1 && "$multi_only" == "yes" ]] && continue
                if [[ "$op" == "sendrecv" ]]; then
                    echo "${JOB_IDS[$idx]}  ${op}  p2p  ${NNODES}  sendrecv_p2p_${NNODES}nodes.out" >> "$manifest"
                else
                    echo "${JOB_IDS[$idx]}  ${op}  ${label}  ${NNODES}  ${op}_${label}_${NNODES}nodes.out" >> "$manifest"
                fi
                idx=$((idx + 1))
            done
        done
    done
    echo "  Manifest: ${manifest}"
fi
