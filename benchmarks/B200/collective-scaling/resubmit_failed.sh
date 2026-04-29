#!/bin/bash
# Resubmit the 7 unexpected failures from run 20260420_124543
# into the same results directory.
set -euo pipefail

PARTITION="batch"
EXCLUDE="use3a-ss-b200-gpu-[130,190,197,199,201,211,233,239]"
GPUS_PER_NODE=8
TIME_LIMIT="00:20:00"
BINARY_PATH="/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB="/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT="/opt/hpcx/hpcx-init.sh"
MIN_BYTES="2G"
MAX_BYTES="16G"
STEP_FACTOR=2
ITERS=20
WARMUP=5
RESULTS_DIR="/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260420_124543"

# Each entry: op algo nodes
FAILED=(
    "all_reduce ring 1"
    "all_reduce ring 8"
    "all_reduce ring 32"
    "all_reduce tree 8"
)

get_algo_env() {
    local algo="$1"
    case "$algo" in
        ring)          printf 'export NCCL_ALGO=Ring\nexport NCCL_NVLS_ENABLE=0\nexport NCCL_COLLNET_ENABLE=0\n' ;;
        nvls)          printf 'unset NCCL_ALGO 2>/dev/null || true\nexport NCCL_NVLS_ENABLE=1\nexport NCCL_COLLNET_ENABLE=0\n' ;;
        tree)          printf 'export NCCL_ALGO=Tree\nexport NCCL_NVLS_ENABLE=0\nexport NCCL_COLLNET_ENABLE=0\n' ;;
        nvls_tree)     printf 'export NCCL_ALGO=NVLSTree\nexport NCCL_NVLS_ENABLE=1\nexport NCCL_COLLNET_ENABLE=0\n' ;;
        collnet_sharp) printf 'unset NCCL_ALGO 2>/dev/null || true\nexport NCCL_NVLS_ENABLE=0\nexport NCCL_COLLNET_ENABLE=1\n' ;;
        p2p)           printf 'unset NCCL_ALGO 2>/dev/null || true\nexport NCCL_NVLS_ENABLE=0\nexport NCCL_COLLNET_ENABLE=0\n' ;;
    esac
}

JOB_IDS=()
for entry in "${FAILED[@]}"; do
    read -r op algo nnodes <<< "$entry"
    if [[ "$op" == "sendrecv" ]]; then
        job_name="nccl-sendrecv-p2p-${nnodes}n"
        outfile="${RESULTS_DIR}/sendrecv_p2p_${nnodes}nodes.out"
        binary="${BINARY_PATH}/sendrecv_perf"
    else
        job_name="nccl-${op}-${algo}-${nnodes}n"
        outfile="${RESULTS_DIR}/${op}_${algo}_${nnodes}nodes.out"
        binary="${BINARY_PATH}/${op}_perf"
    fi

    algo_env="$(get_algo_env "$algo")"

    tmp=$(mktemp /tmp/nccl-resub-XXXXXX.sh)
    cat > "$tmp" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${nnodes}
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

source ${HPCX_INIT} && hpcx_load
export LD_LIBRARY_PATH=${NCCL_LIB}:\${LD_LIBRARY_PATH:-}

${algo_env}
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export CUDA_DEVICE_MAX_CONNECTIONS=32

echo "=== NCCL ${op} | algo=${algo} | ${nnodes} nodes ($((nnodes * GPUS_PER_NODE)) GPUs) [RESUB] ==="
echo "Job ID: \${SLURM_JOB_ID}"
echo "Nodes: \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "Date: \$(date)"
echo ""

srun --mpi=pmix ${binary} \\
    -b ${MIN_BYTES} -e ${MAX_BYTES} -f ${STEP_FACTOR} \\
    -g 1 -n ${ITERS} -w ${WARMUP}

echo ""
echo "=== Done: ${op} ${algo} ${nnodes}n ==="
EOF
    jid=$(sbatch --parsable "$tmp")
    rm -f "$tmp"
    JOB_IDS+=("$jid")
    echo "${job_name} -> Job ${jid}"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} resubmit jobs: ${JOB_IDS[*]}"
