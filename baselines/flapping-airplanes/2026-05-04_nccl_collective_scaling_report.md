# NCCL Collective Scaling Benchmark Report — 2026-05-04

flapping-airplanes B200 cluster (slinky). NCCL collective performance across 1–32 nodes (8–256 GPUs) on the B200 cluster.

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Available nodes | 32 (exeamplar-benchmark partition) |
| Node range tested | 1–32 nodes (8–256 GPUs) |
| Interconnect | Mellanox ConnectX (mlx5), NDR 400 Gb/s InfiniBand |
| NICs per node | 8 HCAs (mlx5_4, mlx5_5, mlx5_6, mlx5_7, mlx5_9, mlx5_10, mlx5_11, mlx5_12) |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| **SHARP** | **Inactive** — `Cannot create SHARP job(-11)` on all multi-node tests |
| Excluded | slinky-9,10,11,12,13,15,16,20,23,24,26,27,30,31,34,36,41,44,50,53 (IB hang) |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.28.9+cuda12.9 |
| NCCL Library | `/usr/lib/x86_64-linux-gnu/libnccl.so.2` |
| NCCL Test Binaries | `~/together-nccl-tests/build/` (nccl-tests v2.17.9) |
| MPI | PMIx (`srun --mpi=pmix`) |
| HPC-X | `/opt/hpcx/hpcx-init.sh` |
| Scheduler | Slurm, partition=`exeamplar-benchmark`, exclusive |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 1M → 8G (factor 2) |
| Iterations | 5 warmup + 20 measured |
| GPUs per rank | 1 (`-g 1`) |
| Results dir | `/data/home/johnson/nccl-results/flapping-airplanes/B200/` |
| Slurm jobs | 1643–1883 |

## Job Results Summary

| Category | Count |
|----------|-------|
| Passed (Out of bounds : 0 OK) | [N] |
| Failed | [N] |
| Expected failures (tree/nvls_tree unsupported) | [N] |

---

## Results — all_reduce Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | NVLSTree | Tree | CollNet SHARP | Best | MD1 ref (2026-05-02) SHARP |
|---|---|---|---|---|---|---|---|
| 1n (8) | — | — | — | — | — | — | — |
| 2n (16) | — | — | — | — | — | — | 533.0 |
| 4n (32) | — | — | — | — | — | — | 543.8 |
| 8n (64) | — | — | — | — | — | — | 547.9 |
| 16n (128) | — | — | — | — | — | — | 546.7 |
| 32n (256) | — | — | — | — | — | — | 544.3 |
| 46n (368) | — | — | — | — | — | — | 544.8 |

**MD1 cluster reference (ring, 2026-05-02):** 1n 679.0 · 2n 344.7 · 4n 334.8 · 8n 322.6 · 16n 306.3 · 32n 304.4 · 46n 291.4

**MD1 cluster reference (NVLS, 2026-05-02):** 1n 840.8 · 2n 717.2 · 4n 334.7 · 8n 306.8 · 16n 306.4 · 32n 305.2 · 46n 291.4

## Results — all_gather Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best | MD1 ref (2026-05-02) SHARP |
|---|---|---|---|---|---|
| 1n (8) | — | — | — | — | — |
| 2n (16) | — | — | — | — | 376.4 |
| 4n (32) | — | — | — | — | 385.2 |
| 8n (64) | — | — | — | — | 383.6 |
| 16n (128) | — | — | — | — | 385.2 |
| 32n (256) | — | — | — | — | 377.6 |
| 46n (368) | — | — | — | — | 385.1 |

Tree / NVLSTree: expected failures (exit code 3) for all_gather — NCCL limitation.

## Results — sendrecv P2P @ 16G (GB/s)

| Nodes (GPUs) | busBW | MD1 ref (2026-05-02) |
|---|---|---|
| 1n (8) | — | 641.6 |
| 2n (16) | — | 42.6 |
| 4n (32) | — | 42.7 |
| 8n (64) | — | 25.4 |
| 16n (128) | — | 15.1 |
| 32n (256) | — | 15.6 |
| 46n (368) | — | 15.6 |

---

## Comparison with MD1 Cluster Baseline (2026-05-02)

### all_reduce CollNet SHARP

| Nodes | MD1 (GB/s) | **This run (GB/s)** | Delta |
|---|---|---|---|
| 2n | 533.0 | — | — |
| 4n | 543.8 | — | — |
| 8n | 547.9 | — | — |
| 16n | 546.7 | — | — |
| 32n | 544.3 | — | — |
| 46n | 544.8 | — | — |

### all_reduce Ring

| Nodes | MD1 (GB/s) | **This run (GB/s)** | Delta |
|---|---|---|---|
| 1n | 679.0 | — | — |
| 2n | 344.7 | — | — |
| 4n | 334.8 | — | — |
| 8n | 322.6 | — | — |
| 16n | 306.3 | — | — |
| 32n | 304.4 | — | — |
| 46n | 291.4 | — | — |

### all_reduce NVLS

| Nodes | MD1 (GB/s) | **This run (GB/s)** | Delta |
|---|---|---|---|
| 1n | 840.8 | — | — |
| 2n | 717.2 | — | — |
| 4n | 334.7 | — | — |
| 8n | 306.8 | — | — |
| 16n | 306.4 | — | — |
| 32n | 305.2 | — | — |
| 46n | 291.4 | — | — |

### all_gather CollNet SHARP

| Nodes | MD1 (GB/s) | **This run (GB/s)** | Delta |
|---|---|---|---|
| 2n | 376.4 | — | — |
| 4n | 385.2 | — | — |
| 8n | 383.6 | — | — |
| 16n | 385.2 | — | — |
| 32n | 377.6 | — | — |
| 46n | 385.1 | — | — |

### sendrecv P2P

| Nodes | MD1 (GB/s) | **This run (GB/s)** | Delta |
|---|---|---|---|
| 1n | 641.6 | — | — |
| 2n | 42.6 | — | — |
| 4n | 42.7 | — | — |
| 8n | 25.4 | — | — |
| 16n | 15.1 | — | — |
| 32n | 15.6 | — | — |
| 46n | 15.6 | — | — |

---

## Key Findings

1. **SHARP:** [Active/inactive — summarize hardware vs software aggregation and busBW vs baseline]
2. **Ring/NVLS:** [Stable / regression / improvement vs MD1 (2026-05-02)]
3. **Node health:** [Any bad nodes, hang patterns, or unexpected failures]

---

## Node Actions

| Node | Status | Action |
|------|--------|--------|
| slinky-9, 10, 11, 12, 13, 15, 16, 20, 23, 24, 26, 27, 30, 31, 34, 36, 41, 44, 50, 53 | IB hang — NCCL all_reduce times out (ring/nvls/collnet-sharp) on every 2-node pair test across 5+ repeated runs (2026-05-04) | Drain and investigate IB connectivity |
| slinky-0, 1, 3, 7, 21, 22, 28, 29, 32, 33, 46, 47 | Healthy — all pair tests COMPLETED (~32–37 s) | OK |

---

## What to Re-run

### 2-Node Pairwise Sweep (IB health check)

Runs 16 disjoint 2-node pairs covering all 32 nodes in parallel. Each job does a NCCL ring all-reduce at 8 G and 16 G. Time limit: 3 min. A TIMEOUT indicates an IB hang on that pair.

```bash
bash /data/home/johnson/nccl-results/run_pair_sweep.sh
```

Results land in a timestamped directory under `/data/home/johnson/nccl-results/flapping-airplanes/pair_sweep_<YYYYMMDD_HHMMSS>/`. The script prints a pass/fail summary with busBW per pair when all jobs finish.

**Healthy pairs (PASS expected):**
`slinky-0+1`, `slinky-3+7`, `slinky-21+22`, `slinky-28+29`, `slinky-32+33`, `slinky-46+47`

**Failing pairs as of 2026-05-04 (TIMEOUT expected until nodes are fixed):**
`slinky-9+10`, `slinky-11+12`, `slinky-13+15`, `slinky-16+20`, `slinky-23+24`,
`slinky-26+27`, `slinky-30+31`, `slinky-34+36`, `slinky-41+44`, `slinky-50+53`

---

## Cross-references

- **MD1 cluster NCCL baseline (2026-05-02):** `2026-05-02_nccl_collective_scaling_report.md`
- **Results dir:** [path]
- **Submit script:** [path]
