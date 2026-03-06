# NCCL Benchmarks

Structured benchmark results for comparing NCCL communication algorithms across GPU types.

## Directory Structure

```
benchmarks/
├── run_nccl_sharp_test.sh   # Shared benchmark script (auto-detects GPU type)
├── H100/
│   └── results/             # Timestamped result directories
├── H200/                    # Created automatically when run on H200
└── ...
```

## Usage

```bash
# Single-node (8 GPUs on current host)
bash benchmarks/run_nccl_sharp_test.sh

# Multi-node via Slurm srun
bash benchmarks/run_nccl_sharp_test.sh --nodes <node1>,<node2>

# Inside sbatch
bash benchmarks/run_nccl_sharp_test.sh --nodes $SLURM_NODELIST
```

GPU type is auto-detected from `nvidia-smi`. Results are saved to
`benchmarks/<GPU_TYPE>/results/<timestamp>/`.

## Configs Tested

| Config | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE | Description |
|--------|-----------|-----------------|---------------------|-------------|
| ring | RING | 0 | 0 | Cluster default (forced by /etc/environment) |
| nvls | (unset) | 1 | 0 | NVSwitch SHARP (intra-node NVLink) |
| collnet_sharp | (unset) | 0 | 1 | IB SHARP (inter-node InfiniBand) |
| nvls_collnet | (unset) | 1 | 1 | NVLS intra-node + IB SHARP inter-node |

---

## Results: H100 80GB HBM3

**Cluster**: 2x nodes (gpu-dp-2jvzl-pqq7p, gpu-dp-2jvzl-ddjjs), 8x H100 per node  
**NCCL**: 2.28.3+cuda13.0 | **Driver**: 570.195.03  
**Test**: AllReduce, float32 sum, 1M–8G, 5 warmup + 20 iters

### 1-Node (8 GPUs) — Peak busBW at 8G message

```bash
bash benchmarks/run_nccl_sharp_test.sh
# Results: benchmarks/H100/results/20260305_233108/
```

| Config | Peak busBW (GB/s) | vs Ring |
|--------|-------------------|---------|
| ring | 368.55 | baseline |
| **nvls** | **477.80** | **+29.7%** |
| collnet_sharp | 367.95 | ~0% (IB SHARP inactive) |
| nvls_collnet | 478.72 | +29.9% |

### 2-Node (16 GPUs) — Peak busBW at 8G message

```bash
bash benchmarks/run_nccl_sharp_test.sh --nodes gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs
# Results: benchmarks/H100/results/20260305_233250/
```

| Config | Peak busBW (GB/s) | vs Ring |
|--------|-------------------|---------|
| ring | 368.24 | baseline |
| **nvls** | **479.81** | **+30.3%** |
| collnet_sharp | 368.27 | ~0% (IB SHARP inactive) |
| nvls_collnet | 479.03 | +30.1% |

### Key Findings

- **NVLS (+30%)**: NVSwitch SHARP reduces 8 GPUs locally before sending over IB — effective for
  both single-node and multi-node workloads.
- **CollNet/IB SHARP (0%)**: No improvement observed. SHARP plugin (`libnccl-net.so` v10) is
  present but IB SHARP is not active/supported for float32 AllReduce on this cluster.
  (`NCCL_ALGO=CollNetDirect` crashes with `ncclInvalidUsage`.)
- **Recommendation**: Use `NCCL_NVLS_ENABLE=1` (unset `NCCL_ALGO`) for best performance.
  `NCCL_COLLNET_ENABLE` has no effect on this cluster.

---

## Scripts

### `run_nccl_sharp_test.sh` — Interactive benchmark

Runs all 4 NCCL configs sequentially, prints output to screen. Each config
uses its own `srun` allocation so there are no Slurm step conflicts.

```bash
# Local (current node, 8 GPUs)
bash benchmarks/run_nccl_sharp_test.sh

# Multi-node
bash benchmarks/run_nccl_sharp_test.sh --nodes gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs

# Inside sbatch
bash benchmarks/run_nccl_sharp_test.sh --nodes $SLURM_NODELIST
```

---

### `slurm_nccl_benchmark.sh` — Slurm benchmark with node selection

Flexible node targeting with 5 modes. Prints output to screen in real time.

```bash
# Local (no args) — current node, 8 GPUs, binary run directly (no srun)
bash benchmarks/slurm_nccl_benchmark.sh

# Single specific node
bash benchmarks/slurm_nccl_benchmark.sh --node gpu-dp-2jvzl-pqq7p

# Explicit node list
bash benchmarks/slurm_nccl_benchmark.sh --nodes gpu-dp-2jvzl-pqq7p,gpu-dp-2jvzl-ddjjs

# All nodes in cluster (auto-detected via sinfo)
bash benchmarks/slurm_nccl_benchmark.sh --all

# All cluster nodes EXCEPT specified
bash benchmarks/slurm_nccl_benchmark.sh --exclude gpu-dp-2jvzl-ddjjs
```

**Output format** (matches `job_nccl_benchmark.sh` style):
```
[date] NCCL benchmark (LOCAL/CLUSTER) started
  Node/Nodes  : ...
  GPU Type    : H100
  Total GPUs  : 8/16
  Results dir : benchmarks/H100/results/<timestamp>/

=== all_reduce_ring ===
[nccl output...]

=== all_reduce_nvls ===
[nccl output...]

=== SUMMARY — Peak busBW at largest message size ===
Test                   Config               busBW (GB/s)
----                   ------               ------------
all_reduce             ring                       368.17
all_reduce             nvls                       478.87
all_reduce             collnet_sharp              367.85
all_reduce             nvls_collnet               479.32

[date] NCCL benchmark finished
```

**Implementation note**: On K8s clusters where `slurmd` runs as PID 1,
multiple sequential `srun` steps within the same Slurm job cause
`slurmstepd` zombie processes that block subsequent steps. This script
avoids the issue by either:
- Running directly (outside sbatch): each `srun` creates its own independent allocation
- Using sbatch chained mode: one job per config via `--dependency=afterok`
