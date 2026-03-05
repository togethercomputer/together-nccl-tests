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
