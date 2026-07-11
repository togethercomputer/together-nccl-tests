---
name: NCCL multi-node working config (slinky cluster, 2026-05-09)
description: Verified working sbatch config for NCCL all_reduce_perf at 8/16/32/60 nodes on slinky/B200. TCP BTL, hcoll/ucc disabled, 12 HCAs, vendor tuning vars.
type: project
originSessionId: 05b58e05-7df9-496b-aeaf-eaa70c54af83
---
## Working sbatch (2026-05-09, jobs 1955/1980-1992)

```bash
#!/bin/bash
#SBATCH --partition=slinky
#SBATCH --nodes=N         # works at 8, 16, 32, 60
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --exclude=slinky-43,slinky-62  # IB-broken at 2026-05-09

source /opt/hpcx/hpcx-init.sh && hpcx_load

# NCCL tuning extracted from passed 62n K8s log (~/fa-nccl-multi-node)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_AR_THRESHOLD=0
# For ring-only at scale (this memory's 2026-05-09 measurements): 12 HCAs OK.
# For SHARP / run_slurm.sh production use: ONLY 8 RDMA HCAs. mlx5_0-3 are
# NVSwitch/non-RDMA on slinky B200 nodes and break SHARP — see commit 12a864a
# (2026-05-13) in together-nccl-tests.
export NCCL_IB_HCA="=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1"

# CRITICAL: disable HPC-X collectives — they cause MPI_Init hangs at scale on this cluster
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll=^hcoll,ucc
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=eth0

srun --mpi=pmix \
  /home/johnson/nccl-tests-upstream/build/all_reduce_perf \
  -b 32G -e 32G -f 2 -g 8 -n 20 -w 5 -t 1 -c 0
```

## Scaling baseline (32 GiB allreduce, busbw avg)
| Nodes | busbw GB/s | job | date |
|---|---|---|---|
| 8 | 390.01 (4× σ≈0.2) | 1955/1956/1957/1977 | 2026-05-09 |
| 16 | 389.17 | 1991 | 2026-05-09 |
| 32 | 388.30 | 1984 | 2026-05-09 |
| 60 (excl 43,62) | 367.36 | 1992 | 2026-05-09 |
| **64 (NO exclusions)** | **388.50** | **2413** | **2026-05-11** |
| 62 (with bad nodes, K8s ref) | 360.35 | ~/fa-nccl-multi-node | (historical) |

**Why:** 64n result on 2026-05-11 matches the 32n flat-scaling line (~388 GB/s) and significantly beats the prior 60n-with-exclusions (367 GB/s) — slinky-43 and slinky-62 are healed (verified in jobs 2410/2411 at 716/714 GB/s 2n). `--exclude=slinky-43,slinky-62` is no longer needed.

**SHARP still inactive** (job 2412, 2026-05-11): 4n with `NCCL_COLLNET_ENABLE=1` got 390 GB/s = ring fallback, vs use3a-ss SHARP ref ~542 GB/s.

**How to apply:** Use this exact env block + `srun --mpi=pmix` for any multi-node NCCL allreduce on slinky/B200. Do NOT use `mpirun` — HPC-X mpirun hangs at ≥12 nodes due to ORTE OOB issues. Do NOT use `--mpi=pmi2` — HPC-X OpenMPI 4.1.2 lacks PMI2 build.

## Things that did NOT work (avoid retrying)
- `mpirun -np N --map-by ppr:1:node` (HPC-X) — hangs in ORTE OOB at ≥12 nodes; works at 8 nodes only
- `srun --mpi=pmi2` — `OPAL ERROR: Unreachable in ext3x_client.c`, OMPI not built with PMI2
- `srun --mpi=pmix` WITHOUT disabling hcoll/ucc — hangs in MPI_Init at scale
- Setting `NCCL_IB_HCA="=mlx5_..."` (with `=` prefix) — works but auto-list (no `=`) is what codex used and is fine
- Together-nccl-tests binary at `/data/home/johnson/together-nccl-tests/build/` — works, but the upstream build at `/home/johnson/nccl-tests-upstream/build/all_reduce_perf` is what codex used successfully

## Key files
- Working sbatch templates: `/home/johnson/nccl-ar-scale-b200.sbatch`, `/home/johnson/nccl-ar-60n-b200-c0.sbatch`
- Reference passed log (K8s, 62n, 360 GB/s): `/home/johnson/fa-nccl-multi-node`
- Result logs: `/home/johnson/nccl-ar-{8n,16n,32n,60n}-c0-*.log`
