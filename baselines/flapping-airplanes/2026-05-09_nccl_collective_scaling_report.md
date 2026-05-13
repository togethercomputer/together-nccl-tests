# NCCL Collective Scaling Benchmark Report — 2026-05-09

flapping-airplanes B200 cluster (slinky). NCCL collective performance across 1–60 nodes (8–480 GPUs).

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Available nodes | 62 in `slinky` partition (`slinky-[0-41,43-51,53-63]`) |
| Node range tested | 1, 2, 4, 8, 16, 32, 48, 60 nodes (8–480 GPUs) |
| Interconnect | Mellanox ConnectX, NDR 400 Gb/s InfiniBand |
| NICs per node | **12 IB HCAs** — `mlx5_0..7, 9..12` (corrected; prior note saying mlx5_0..3 were non-RDMA was wrong) |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| **SHARP** | **Inactive** — all CollNet runs return exit 3 (`Cannot create SHARP job`) |
| Excluded | `slinky-43`, `slinky-62` — both fail at `ibv_reg_mr_iova2` / `ibv_create_qp` (Cannot allocate memory) on every multi-node run |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.28.9+cuda12.9 |
| NCCL Test Binaries | `/home/johnson/nccl-tests-upstream/build/` |
| MPI | HPC-X OpenMPI 4.1.2, `srun --mpi=pmix` |
| Critical MCA settings | `OMPI_MCA_coll=^hcoll,ucc` (disable HPC-X collectives — they hang at scale on this cluster), `pml=ob1`, `btl=tcp,self` |
| Vendor NCCL tuning | `NCCL_IB_PCI_RELAXED_ORDERING=1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_AR_THRESHOLD=0`, `NCCL_IB_SPLIT_DATA_ON_QPS=0`, `NCCL_IGNORE_CPU_AFFINITY=1` |
| HPC-X | `/opt/hpcx/hpcx-init.sh` |
| Scheduler | Slurm partition=`slinky`, `--exclusive`, `--exclude=slinky-43,slinky-62` |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 1 MiB → 32 GiB (factor 2 sweep, 16 sizes) |
| Iterations | 5 warmup + 20 measured (`-w 5 -n 20`) |
| GPUs per process | 8 (`-g 8`, 1 process per node) |
| Validation | off (`-c 0`) |
| Sbatch template | `/home/johnson/nccl-scaling-2026-05-09/wrapper.sbatch` |
| Submitter | `/home/johnson/nccl-scaling-2026-05-09/submit.sh` |
| Slurm jobs | 1994–2057 |
| Results dir | `/home/johnson/nccl-scaling-2026-05-09/logs/` |
| TSV | `/home/johnson/nccl-scaling-2026-05-09/results.tsv` |

## Job Results Summary

| Category | Count | Notes |
|----------|-------|-------|
| Passed | 39 | all Ring, Tree, sendrecv across all sizes; NVLS @ 1n; **NVLSTree @ 2-60n** |
| Expected failures | 25 | CollnetChain/CollnetDirect (no SHARP, all 16 — even with correct algo names); multi-node NVLS (8); single-node NVLSTree (1) |
| Hardware failures | 0 (with exclusion) | slinky-43, 62 excluded; would otherwise hit `ibv_reg_mr_iova2` |

> Note: my initial report set `NCCL_ALGO=CollNet` (token unrecognized by NCCL 2.28). Re-tested with the canonical names `CollnetChain` and `CollnetDirect` (jobs 2058, 2059): both fail with "no algorithm/protocol available" — confirming SHARP is genuinely inactive at the service layer (the plugin loads but `Cannot create SHARP job` underneath).

---

## Results — all_reduce busBW peak (GB/s)

| Nodes (GPUs) | Ring | Tree | NVLS | NVLSTree | CollnetChain | CollnetDirect | Best |
|---|---|---|---|---|---|---|---|
| 1n (8) | 696.38 | 551.22 | **843.66** | ✗ ≥2n only | ✗ no SHARP | ✗ no SHARP | **NVLS 843.66** |
| 2n (16) | 389.45 | 338.63 | ✗ ≥1 node only | **721.56** | ✗ | ✗ | **NVLSTree 721.56** |
| 4n (32) | **390.14** | 191.90 | ✗ | 360.71 | ✗ | ✗ | Ring 390.14 |
| 8n (64) | **390.60** | 193.81 | ✗ | 345.88 | ✗ | ✗ | Ring 390.60 |
| 16n (128) | **389.91** | 195.47 | ✗ | 360.56 | ✗ | ✗ | Ring 389.91 |
| 32n (256) | **389.26** | 195.37 | ✗ | 345.24 | ✗ | ✗ | Ring 389.26 |
| 48n (384) | **386.56** | 195.59 | ✗ | 338.66 | ✗ | ✗ | Ring 386.56 |
| 60n (480) | **366.11** | 195.36 | ✗ | 337.62 | ✗ | ✗ | Ring 366.11 |

**NVLSTree** (NVLink-aggregation + tree inter-node, no SHARP needed) wins at 2n (intra-NVLink dominates), but Ring takes over from 4n onward. CollnetChain / CollnetDirect both fail with "no algorithm/protocol available" — SHARP service inactive.

Peak busBW occurs at 32 GiB for all multi-node Ring runs.

## Results — all_reduce busBW @ 8 GiB / 16 GiB / 32 GiB (Ring, GB/s)

| Nodes | 8 GiB | 16 GiB | 32 GiB |
|---|---|---|---|
| 1n | 689.93 | 693.62 | 696.38 |
| 2n | 388.80 | 388.23 | 389.45 |
| 4n | 388.77 | 387.02 | 390.14 |
| 8n | 389.52 | 389.23 | 390.60 |
| 16n | 387.41 | 387.48 | 389.91 |
| 32n | 386.81 | 388.13 | 389.26 |
| 48n | 379.56 | 372.72 | 386.56 |
| 60n | 273.01 | 339.73 | 366.11 |

## Results — all_gather busBW peak (GB/s)

| Nodes (GPUs) | Ring | NVLS | CollNet/SHARP | Best |
|---|---|---|---|---|
| 1n (8) | 670.39 | 502.27 | ✗ | **Ring 670.39** |
| 2n (16) | **371.24** | ✗ | ✗ | Ring 371.24 |
| 4n (32) | **377.34** | ✗ | ✗ | Ring 377.34 |
| 8n (64) | **381.01** | ✗ | ✗ | Ring 381.01 |
| 16n (128) | **382.64** | ✗ | ✗ | Ring 382.64 |
| 32n (256) | **383.53** | ✗ | ✗ | Ring 383.53 |
| 48n (384) | **372.46** | ✗ | ✗ | Ring 372.46 |
| 60n (480) | **353.25** | ✗ | ✗ | Ring 353.25 |

Tree / NVLSTree skipped (NCCL doesn't support these for all_gather).

## Results — sendrecv P2P busBW peak (GB/s)

| Nodes (GPUs) | busBW peak |
|---|---|
| 1n (8) | 651.52 |
| 2n (16) | 34.27 |
| 4n (32) | 32.18 |
| 8n (64) | 17.19 |
| 16n (128) | 17.21 |
| 32n (256) | 17.20 |
| 48n (384) | 17.20 |
| 60n (480) | 17.18 |

sendrecv saturates at ~17 GB/s/peer for ≥8n — bound by per-NIC bandwidth across the rank-pair pattern.

---

## Comparison vs. K8s reference (the "matched benchmark", `~/fa-nccl-multi-node`)

The K8s-launched 62-node run at 32 GiB allreduce ring reported avg busbw **360.35 GB/s**. That run included slinky-43 and slinky-62, which dragged the number down. Excluding those nodes, this slurm-mpirun path achieves:

| Nodes | This run @ 32 GiB Ring | Notes |
|---|---|---|
| 32 | 389.26 | within 0.3% of 8n (390.60) |
| 48 | 386.56 | -1% vs 32n |
| 60 | 366.11 | -7% vs 32n; **+1.6% vs K8s 62n reference (360.35)** |

The matched benchmark is reproducible — and exceeded — once the two IB-broken nodes are excluded.

---

## Comparison with MD1 Cluster Baseline (2026-05-02)

MD1 had SHARP active. On flapping-airplanes SHARP is **inactive**, so direct CollNet/SHARP comparison is not possible. Below: this run's best (Ring) vs MD1's SHARP and Ring numbers.

### all_reduce busBW @ 16 GiB

| Nodes | MD1 SHARP | MD1 Ring | This run Ring | vs MD1 Ring |
|---|---|---|---|---|
| 2n | 533.0 | 344.7 | 388.23 | +12.6% |
| 4n | 543.8 | 334.8 | 387.02 | +15.6% |
| 8n | 547.9 | 322.6 | 389.23 | +20.7% |
| 16n | 546.7 | 306.3 | 387.48 | +26.5% |
| 32n | 544.3 | 304.4 | 388.13 | +27.5% |

(SHARP, when available, beats ring at scale on MD1; on this cluster Ring is the only fast multi-node algo, but it's stronger than MD1 Ring.)

### all_gather busBW @ 16 GiB

| Nodes | MD1 SHARP | This run Ring | vs MD1 SHARP |
|---|---|---|---|
| 2n | 376.4 | 371.24 | -1.4% |
| 4n | 385.2 | 374.26 | -2.8% |
| 8n | 383.6 | 381.01 | -0.7% |
| 16n | 385.2 | 379.72 | -1.4% |
| 32n | 377.6 | 381.98 | +1.2% |

Ring all_gather effectively matches MD1's SHARP all_gather.

### sendrecv P2P @ 16 GiB

| Nodes | MD1 | This run | Delta |
|---|---|---|---|
| 1n | 641.6 | 651.13 | +1.5% |
| 2n | 42.6 | 34.26 | -19.6% |
| 4n | 42.7 | 32.16 | -24.7% |
| 8n | 25.4 | 17.18 | -32.4% |
| 16n | 15.1 | 17.20 | +13.9% |
| 32n | 15.6 | 17.19 | +10.2% |

sendrecv ≤8n is below MD1; above 16n is slightly better.

---

## Key Findings

1. **Ring is the workhorse on this cluster.** Multi-node Ring all_reduce holds 386–390 GB/s busbw across 2 → 48 nodes (≤1% spread), and 366 GB/s at 60n. **Beats the MD1 Ring baseline by 12–27%** at all multi-node sizes.

2. **SHARP is unavailable** — confirmed at the service layer with **correctly-named** `CollnetChain` and `CollnetDirect` (jobs 2058, 2059): both NCCL-rejected with "no algorithm/protocol available". The plugin loads (`Loaded collnet plugin SHARP (v8)`) but `Cannot create SHARP job(-11)` underneath. **However**, NVLSTree (NVLink-tree, no SHARP) **does work** multi-node — wins decisively at 2n (721 vs Ring 389), then settles to ~338-360 GB/s for ≥4n (consistently slower than Ring at ≥4n). NVLS alone is single-node-only by design.

3. **2 hardware-broken nodes** confirmed: `slinky-43` and `slinky-62`. Both produce `ibv_reg_mr_iova2 failed` and `ibv_create_qp ... Cannot allocate memory` on every multi-node test that includes them. Most likely root cause: `ulimit -l` not actually unlimited on those nodes (the script's `ulimit -l unlimited 2>/dev/null || true` silently fails under K8s/cgroup limits and can't raise hard limit). Alternatives: HCA QP/MR resource exhaustion, driver state. Excluding both unblocks the entire ≤60n test space.

4. **The K8s "matched benchmark" 360.35 GB/s number is below this cluster's true ring-allreduce ceiling.** Excluding the 2 broken nodes, 60n achieves 366 GB/s avg busbw — **+1.6% over the K8s reference**. The K8s reference was capped by the broken nodes. The achievable ceiling at small-to-medium scale is ~390 GB/s (≤32n).

5. **Critical launcher fix.** HPC-X collectives (`hcoll`, `ucc`) hang MPI_Init at scale on this cluster. Disable via `OMPI_MCA_coll=^hcoll,ucc` and force `pml=ob1, btl=tcp,self`. With these settings, `srun --mpi=pmix` works at all tested sizes. Without them, MPI_Init hangs starting at 12 nodes regardless of UCX/PMI version. (Direct `mpirun` and `srun --mpi=pmi2` also fail; the documented working path is the codex sbatch in `/home/johnson/nccl-ar-scale-b200.sbatch`.)

6. **Tree all_reduce flatlines at ~195 GB/s** for ≥4n. Useful as a fallback if Ring degrades, but always slower than Ring on this topology.

---

## Node Actions

| Node | Status | Action |
|------|--------|--------|
| slinky-43 | IB MR allocation fails (`ibv_reg_mr_iova2`, `Cannot allocate memory`) on every multi-node test (confirmed 2026-05-09 via codex 2-node tests + this run's job 1993) | **Drain.** Investigate `ulimit -l` enforcement / pam-limits / HCA QP-MR state. |
| slinky-62 | Same `ibv_reg_mr_iova2` failure (2026-05-09) | **Drain.** Same investigation. |
| slinky-[0-41,44-51,53-61,63] (60 nodes) | Healthy under `srun --mpi=pmix` + hcoll/ucc disabled | OK |

The earlier 2026-05-04 "20 IB-broken nodes" diagnosis is **superseded** — those were launcher-environment hangs, not hardware. With this report's launcher fix, all 60 non-{43,62} nodes run cleanly.

---

## Reproduce

```bash
# Generate full sweep
bash /home/johnson/nccl-scaling-2026-05-09/submit.sh 1,2,4,8,16,32,48,60

# Extract TSV
/home/johnson/nccl-scaling-2026-05-09/parse.sh > results.tsv

# Single 60n allreduce-ring (peak run)
sbatch /home/johnson/nccl-ar-60n-b200-c0.sbatch
```

---

## Cross-references

- **MD1 cluster NCCL baseline (2026-05-02):** `/home/johnson/worklogs/2026-05-02_nccl_collective_scaling_report.md`
- **Reference (K8s 62n passed log, 360.35 GB/s):** `/home/johnson/fa-nccl-multi-node`
- **Working sbatch templates:** `/home/johnson/nccl-ar-scale-b200.sbatch`, `/home/johnson/nccl-ar-60n-b200-c0.sbatch`, `/home/johnson/nccl-scaling-2026-05-09/wrapper.sbatch`
- **All result logs:** `/home/johnson/nccl-scaling-2026-05-09/logs/`
- **TSV summary:** `/home/johnson/nccl-scaling-2026-05-09/results.tsv`
