#!/bin/bash
#
# NCCL SendRecv P2P benchmark — parameterized by SLURM_NNODES
# Native mode: HPCx MPI + shared NCCL libs (no container)
#
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
#SBATCH --exclude=use3a-ss-b200-gpu-143,use3a-ss-b200-gpu-158,use3a-ss-b200-gpu-159

set -euo pipefail

NNODES=${SLURM_NNODES}
NGPUS=$((NNODES * 8))

# HPCx for MPI
source /opt/hpcx/hpcx-init.sh && hpcx_load

# NCCL lib on shared storage
export LD_LIBRARY_PATH=/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# NCCL environment — match training job settings
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export CUDA_DEVICE_MAX_CONNECTIONS=32

SENDRECV=/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build/sendrecv_perf
OUTDIR=/mnt/vast/dgxc-benchmarking-auto/nvbandwidth_benchmarks/results
mkdir -p ${OUTDIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== NCCL SendRecv P2P Benchmark — ${NNODES} Nodes (${NGPUS} GPUs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "Date: $(date)"
echo ""

# --- SendRecv sweep (1KB to 256MB) ---
echo ">>> SendRecv bandwidth sweep (1KB to 256MB) — ${NGPUS} GPUs..."
srun --mpi=pmix ${SENDRECV} -b 1K -e 256M -f 2 -g 1 -n 20 -w 5
echo ""

# --- SendRecv at 64MB (activation tensor size for 405B TP=4) ---
echo ">>> SendRecv at 64MB (405B activation tensor) — ${NGPUS} GPUs..."
srun --mpi=pmix ${SENDRECV} -b 64M -e 64M -g 1 -n 50 -w 5
echo ""

echo "=== ${NNODES}-Node SendRecv Complete ==="
