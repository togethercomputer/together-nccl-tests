# NCCL Benchmarks

Benchmark scripts and results for comparing NCCL communication algorithms
(Ring, NVLS, CollNet/SHARP) across GPU types and cluster scales.

## Directory Structure

```
benchmarks/
├── run_nccl_sharp_test.sh    # Interactive: single/multi-node, output to screen
├── slurm_nccl_benchmark.sh   # Slurm: flexible node selection, output to screen
├── H100/
│   └── results/              # Timestamped result directories (auto-created)
├── H200/                     # Created automatically when run on H200
└── ...
```

GPU type is auto-detected from `nvidia-smi`. Results are saved to
`benchmarks/<GPU_TYPE>/results/<timestamp>/`.

---

## Scripts

### `run_nccl_sharp_test.sh`

Simple interactive script. Auto-detects GPU type, runs 4 configs, saves results.

```bash
# Local — current node, 8 GPUs
bash benchmarks/run_nccl_sharp_test.sh

# Multi-node
bash benchmarks/run_nccl_sharp_test.sh --nodes <node1>,<node2>

# Inside sbatch
bash benchmarks/run_nccl_sharp_test.sh --nodes $SLURM_NODELIST
```

---

### `slurm_nccl_benchmark.sh`

Slurm-aware script with flexible node selection. All modes print to screen in real time.

```bash
# Local (no args) — current node, 8 GPUs, binary run directly (no srun)
bash benchmarks/slurm_nccl_benchmark.sh

# Single specific node
bash benchmarks/slurm_nccl_benchmark.sh --node <nodename>

# Explicit node list
bash benchmarks/slurm_nccl_benchmark.sh --nodes <node1>,<node2>

# All nodes in cluster (auto-detected via sinfo)
bash benchmarks/slurm_nccl_benchmark.sh --all

# All cluster nodes EXCEPT specified
bash benchmarks/slurm_nccl_benchmark.sh --exclude <node1>,<node2>
```

> **Note for K8s clusters (slurmd=PID1)**: multiple sequential `srun` steps
> in one Slurm job cause `slurmstepd` zombie processes that block subsequent
> steps. This script avoids the issue — when run directly, each `srun` creates
> its own independent allocation; when submitted via sbatch, it chains one job
> per config with `--dependency=afterok`.

---

## NCCL Configs Tested

| Config | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE | Description |
|--------|-----------|-----------------|---------------------|-------------|
| `ring` | RING | 0 | 0 | Cluster default (forced by `/etc/environment`) |
| `nvls` | (unset) | 1 | 0 | NVSwitch SHARP — intra-node NVLink reduction |
| `collnet_sharp` | (unset) | 0 | 1 | IB SHARP — inter-node InfiniBand reduction |
| `nvls_collnet` | (unset) | 1 | 1 | NVLS intra-node + IB SHARP inter-node |

---

## Results: H100 80GB HBM3

**Cluster**: 2× nodes (gpu-dp-2jvzl-pqq7p, gpu-dp-2jvzl-ddjjs), 8× H100 per node  
**NCCL**: 2.28.3+cuda13.0 | **Driver**: 570.195.03  
**Test**: AllReduce, float32 sum, 1M–8G, 5 warmup + 20 iters

### 1-Node (8 GPUs)

```bash
bash benchmarks/run_nccl_sharp_test.sh
# Results: benchmarks/H100/results/20260305_233108/
```

| Config | Peak busBW (GB/s) | vs Ring |
|--------|:-----------------:|:-------:|
| ring | 368.55 | baseline |
| **nvls** | **477.80** | **+29.7%** |
| collnet_sharp | 367.95 | ~0% |
| nvls_collnet | 478.72 | +29.9% |

### 2-Node (16 GPUs)

```bash
bash benchmarks/run_nccl_sharp_test.sh --nodes gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs
# Results: benchmarks/H100/results/20260305_233250/
```

| Config | Peak busBW (GB/s) | vs Ring |
|--------|:-----------------:|:-------:|
| ring | 368.24 | baseline |
| **nvls** | **479.81** | **+30.3%** |
| collnet_sharp | 368.27 | ~0% |
| nvls_collnet | 479.03 | +30.1% |

### Key Findings

- **NVLS gives +30%** on both 1-node and 2-node. NVSwitch reduces all 8 GPUs
  locally before transmitting over IB, cutting inter-node traffic significantly.
- **IB SHARP (CollNet) is inactive** on this cluster. The SHARP plugin
  (`libnccl-net.so` v10) is present but does not support `float32` AllReduce.
  Forcing `NCCL_ALGO=CollNetDirect` crashes with `ncclInvalidUsage`.
- **Recommendation**: set `NCCL_NVLS_ENABLE=1` and leave `NCCL_ALGO` unset.
  `NCCL_COLLNET_ENABLE` has no effect on this cluster.
