# P2P Bandwidth Scaling: 2 to 64 Nodes (B200 Cluster)

**Date:** 2026-04-10
**Cluster:** Together AI B200 (DGXC) — 75 nodes, 8x B200 GPUs per node
**NCCL Version:** 2.29.7+cuda12.9
**Tested at:** 2, 4, 8, 16, 32, 64 nodes (16 to 512 GPUs)
**Runs:** 3 independent runs across different physical node sets

## Why This Test

Llama 3.1 405B NVFP4 training uses pipeline parallelism (PP=16) across 32 nodes.
Profiling showed P2P Send/Recv as the #1 bottleneck — 38-43% of GPU time.
This benchmark measures how P2P bandwidth degrades as we scale from 2 to 64 nodes,
to understand the network contention driving that bottleneck.

## How It Works

`nccl-tests/sendrecv_perf` creates a ring: each GPU sends to the next GPU and
receives from the previous one simultaneously. At 32 nodes (256 GPUs), that ring
has 224 intra-node links (fast NVLink) and 32 inter-node links (IB fabric).
The reported bandwidth is limited by the slowest links — the inter-node IB hops.

The key message size is **64MB** — the activation tensor size for 405B with TP=4:
`hidden=16384 / TP=4 * seq=8192 * MBS=1 * 2 bytes(bf16) = 64MB`

## Setup

- **Hardware:** 8x B200 (NV18 NVLink/NVSwitch) + 8x ConnectX-7 IB 400Gb/s per node
- **NCCL settings:** NVLS disabled, SHARP disabled (matches training config)
- **Sweep:** 1KB to 256MB (factor 2, 20 iterations)
- **Fixed test:** 64MB only (50 iterations)
- **Script:** `benchmarks/nccl_sendrecv_Nnode.sh`

---

## Key Results: Bandwidth at 64MB

### Sweep Test (20 iterations, busBW GB/s out-of-place)

| Nodes | GPUs | Run 1 | Run 2 | Run 3 | Average | Scaling Eff. |
|------:|-----:|------:|------:|------:|--------:|-------------:|
| 2 | 16 | 41.86 | 43.13 | 42.82 | 42.60 | 100% |
| 4 | 32 | 43.88 | 44.44 | 44.39 | 44.24 | 103.8% |
| 8 | 64 | 26.21 | 26.04 | 26.21 | 26.15 | 61.4% |
| 16 | 128 | 14.61 | 14.32 | 14.62 | 14.52 | 34.1% |
| 32 | 256 | 14.43 | 14.57 | 14.59 | 14.53 | 34.1% |
| 64 | 512 | — | — | 14.69 | 14.69 | 34.5% |

### Fixed 64MB Test (50 iterations, busBW GB/s out-of-place)

| Nodes | GPUs | Run 1 | Run 2 | Run 3 | Average |
|------:|-----:|------:|------:|------:|--------:|
| 2 | 16 | 41.62 | 41.66 | 41.78 | 41.69 |
| 4 | 32 | — | 43.35 | 43.03 | 43.19 |
| 8 | 64 | 25.53 | 25.52 | — | 25.53 |
| 16 | 128 | 14.16 | 13.64 | 14.18 | 13.99 |
| 32 | 256 | 13.82 | 14.24 | 14.23 | 14.10 |
| 64 | 512 | — | — | 14.27 | 14.27 |

("—" = data not collected due to NCCL teardown hang or not tested at that scale)

---

## Three Scaling Regimes

```
  busBW (GB/s)
  45 |  *----*
     |        \
  35 |         \        Regime 1: No contention
     |          \       (2-4 nodes, ~43 GB/s)
  25 |           *
     |            \     Regime 2: Contention cliff
     |             \    (4→8→16 nodes, -41% per step)
  15 |              *---*----*----*
     |                  Regime 3: Saturation plateau
     |                  (16-64 nodes, ~14.6 GB/s)
   0 +---+---+---+---+---+---+--→ nodes
       2   4   8  16  32  64
```

### Regime 1 — No contention (2-4 nodes): ~43 GB/s

The IB fabric has plenty of capacity for 2-4 inter-node links.
4 nodes is slightly faster than 2 because the ring has a better
intra-node/inter-node link ratio.

### Regime 2 — Contention cliff (4→8→16 nodes): 44→26→15 GB/s

- 4→8 nodes: **-41% bandwidth drop**
- 8→16 nodes: **-44% bandwidth drop**

Each doubling of nodes roughly halves the per-link bandwidth.
The IB fabric cannot sustain full line rate for this many
concurrent inter-node transfers.

### Regime 3 — Saturation plateau (16-64 nodes): ~14.6 GB/s

- 16→32 nodes: **-0.2%** (no change)
- 32→64 nodes: **+0.7%** (no change)

The fabric is fully saturated at 16 nodes. Doubling to 32 or
quadrupling to 64 nodes causes **zero additional degradation**.
This is the operating regime of the 405B training job.

---

## Reproducibility

All results are consistent within ±3% across 3 independent runs on different
physical node sets:

| Nodes | Max Delta Across Runs | Node Sets Used |
|------:|----------------------:|:---------------|
| 2 | 3.0% | gpu-149,233 / gpu-154-155 / gpu-145-146 |
| 4 | 1.3% | gpu-149-152 (all runs) |
| 8 | 0.7% | gpu-209-216 / gpu-149-157 / gpu-209-216 |
| 16 | 2.1% | gpu-160-175 / gpu-145,209-217,226-230 / gpu-160-175 |
| 32 | 1.1% | gpu-149-235 / gpu-160-190,195 / gpu-160-190,195 |

Contiguous vs scattered node placement has no measurable impact.
This confirms bandwidth is a **fabric-level property**, not node-specific.

---

## What This Means for 405B Training

| Config | PP Stages | Inter-node Links | P2P busBW | Time per 64MB |
|--------|----------:|------------------:|----------:|--------------:|
| PP=16 (current) | 16 | 16 (Regime 3) | ~14.5 GB/s | ~4.4 ms |
| PP=8 (validated) | 8 | 8 (Regime 2) | ~26.2 GB/s | ~2.4 ms |

PP=8 operates in Regime 2 instead of Regime 3:
- **1.8x faster per P2P transfer** (2.4ms vs 4.4ms)
- **Half the pipeline stages** (7 boundaries vs 15)
- **Result:** +38% TFLOP/s improvement (already validated: 1,868 vs 1,355 TFLOP/s/GPU)

---

## Comparison: Benchmark vs Training Profiler

| Metric | Benchmark (32-node) | Training Profiler |
|--------|--------------------:|------------------:|
| SendRecv at 64MB | ~4.5 ms | median 3 ms, max 200 ms |
| P2P busBW at 64MB | ~14.5 GB/s | ~21 GB/s (64MB/3ms) |

- Training is faster because PP sends between **specific adjacent stages**,
  not a full 256-rank ring. Many PP boundaries are intra-node.
- The 200ms outliers are **not bandwidth issues** — they reflect pipeline bubble
  synchronization delays and straggler effects.

---

## Full Bandwidth Sweep — Run 3 (busBW GB/s, out-of-place)

| Size | 2-Node | 4-Node | 8-Node | 16-Node | 32-Node | 64-Node |
|-----:|-------:|-------:|-------:|--------:|--------:|--------:|
| 1K | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 2K | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 |
| 4K | 0.15 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 |
| 8K | 0.26 | 0.30 | 0.27 | 0.27 | 0.28 | 0.28 |
| 16K | 0.70 | 0.49 | 0.52 | 0.51 | 0.50 | 0.51 |
| 32K | 1.28 | 1.06 | 0.97 | 0.96 | 0.97 | 0.98 |
| 64K | 1.84 | 1.67 | 1.68 | 1.63 | 1.61 | 1.60 |
| 128K | 3.17 | 2.97 | 2.99 | 2.97 | 2.96 | 3.05 |
| 256K | 5.59 | 5.80 | 5.74 | 4.74 | 4.59 | 4.68 |
| 512K | 10.02 | 10.44 | 9.32 | 6.41 | 6.41 | 6.48 |
| 1M | 18.01 | 18.03 | 15.52 | 10.45 | 10.40 | 10.53 |
| 2M | 26.51 | 27.73 | 20.12 | 12.20 | 12.20 | 12.19 |
| 4M | 34.51 | 34.63 | 23.55 | 13.42 | 13.44 | 13.46 |
| 8M | 39.05 | 40.10 | 25.47 | 14.10 | 14.07 | 14.19 |
| 16M | 41.85 | 43.58 | 26.16 | 14.41 | 14.37 | 14.49 |
| 32M | 42.38 | 43.57 | 26.19 | 14.55 | 14.51 | 14.63 |
| 64M | 42.82 | 44.39 | 26.21 | 14.62 | 14.59 | 14.69 |
| 128M | 43.15 | 44.37 | 26.16 | 14.66 | 14.62 | 14.76 |
| 256M | 43.14 | 44.33 | 26.13 | 14.75 | 14.65 | 14.79 |

## Full Bandwidth Sweep — Run 1 (busBW GB/s, out-of-place)

| Size | 2-Node | 4-Node | 8-Node | 16-Node | 32-Node |
|-----:|-------:|-------:|-------:|--------:|--------:|
| 1K | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 |
| 2K | 0.08 | 0.07 | 0.06 | 0.07 | 0.07 |
| 4K | 0.19 | 0.17 | 0.14 | 0.14 | 0.14 |
| 8K | 0.38 | 0.30 | 0.29 | 0.29 | 0.29 |
| 16K | 0.68 | 0.52 | 0.52 | 0.52 | 0.52 |
| 32K | 1.13 | 1.04 | 0.93 | 0.96 | 0.95 |
| 64K | 1.77 | 1.56 | 1.58 | 1.71 | 1.68 |
| 128K | 3.49 | 2.96 | 3.06 | 3.09 | 3.06 |
| 256K | 6.61 | 5.96 | 5.80 | 4.86 | 4.82 |
| 512K | 11.72 | 10.55 | 8.92 | 6.54 | 6.46 |
| 1M | 20.72 | 19.07 | 15.25 | 10.48 | 10.36 |
| 2M | 28.17 | 27.94 | 20.14 | 12.29 | 12.15 |
| 4M | 34.49 | 34.60 | 23.20 | 13.39 | 13.22 |
| 8M | 39.45 | 40.28 | 25.48 | 14.10 | 13.92 |
| 16M | 41.67 | 42.92 | 26.14 | 14.37 | 14.18 |
| 32M | 41.37 | 43.72 | 26.25 | 14.53 | 14.36 |
| 64M | 41.86 | 43.88 | 26.21 | 14.61 | 14.43 |
| 128M | 41.95 | 44.17 | 26.19 | 14.67 | 14.50 |
| 256M | 41.98 | 44.40 | 26.15 | 14.67 | 14.50 |

## Full Bandwidth Sweep — Run 2 (busBW GB/s, out-of-place)

| Size | 2-Node | 4-Node | 8-Node | 16-Node | 32-Node |
|-----:|-------:|-------:|-------:|--------:|--------:|
| 1K | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 |
| 2K | 0.09 | 0.08 | 0.07 | 0.07 | 0.07 |
| 4K | 0.20 | 0.16 | 0.14 | 0.14 | 0.14 |
| 8K | 0.38 | 0.32 | 0.28 | 0.26 | 0.27 |
| 16K | 0.69 | 0.56 | 0.50 | 0.51 | 0.50 |
| 32K | 1.15 | 1.09 | 0.97 | 0.97 | 0.96 |
| 64K | 1.82 | 1.65 | 1.59 | 1.58 | 1.62 |
| 128K | 3.59 | 3.20 | 3.04 | 3.00 | 3.03 |
| 256K | 6.86 | 6.14 | 5.53 | 4.62 | 4.68 |
| 512K | 12.08 | 10.77 | 8.70 | 6.32 | 6.43 |
| 1M | 21.15 | 19.22 | 15.17 | 10.39 | 10.46 |
| 2M | 28.88 | 27.36 | 19.89 | 11.92 | 12.14 |
| 4M | 35.55 | 34.59 | 23.08 | 13.14 | 13.39 |
| 8M | 40.17 | 40.72 | 25.22 | 13.79 | 14.06 |
| 16M | 42.70 | 43.45 | 25.94 | 14.09 | 14.35 |
| 32M | 42.90 | 44.19 | 26.08 | 14.24 | 14.51 |
| 64M | 43.13 | 44.44 | 26.04 | 14.32 | 14.57 |
| 128M | 43.29 | 44.74 | 26.02 | 14.36 | 14.61 |
| 256M | 43.36 | 44.94 | 26.00 | 14.37 | 14.63 |

---

## Raw Data Files

### Run 1 (2-32 nodes)
| Nodes | File | Physical Nodes |
|------:|:-----|:---------------|
| 2 | `run1/nccl-sendrecv-2node-79912.out` | gpu-149,233 |
| 4 | `run1/nccl-sendrecv-4node-79915.out` | gpu-149-152 |
| 8 | `run1/nccl-sendrecv-8node-79916.out` | gpu-209-216 |
| 16 | `run1/nccl-sendrecv-16node-79917.out` | gpu-160-175 |
| 32 | `run1/nccl-sendrecv-32node-79914.out` | gpu-149-235 |

### Run 2 (2-32 nodes, reproducibility)
| Nodes | File | Physical Nodes |
|------:|:-----|:---------------|
| 2 | `run2/nccl-sendrecv-2node-v2-79921.out` | gpu-154-155 |
| 4 | `run2/nccl-sendrecv-4node-v2-79922.out` | gpu-149-152 |
| 8 | `run2/nccl-sendrecv-8node-v2-79923.out` | gpu-149-157 |
| 16 | `run2/nccl-sendrecv-16node-v2-79924.out` | gpu-145,146,209-217,226-230 |
| 32 | `run2/nccl-sendrecv-32node-v2-79925.out` | gpu-160-190,195 |

### Run 3 (2-64 nodes, with 64-node new)
| Nodes | File | Physical Nodes |
|------:|:-----|:---------------|
| 2 | `run3/nccl-sendrecv-2node-79939.out` | gpu-145-146 |
| 4 | `run3/nccl-sendrecv-4node-79940.out` | gpu-149-152 |
| 8 | `run3/nccl-sendrecv-8node-79941.out` | gpu-209-216 |
| 16 | `run3/nccl-sendrecv-16node-79945.out` | gpu-160-175 |
| 32 | `run3/nccl-sendrecv-32node-79946.out` | gpu-160-190,195 |
| 64 | `run3/nccl-sendrecv-64node-79947.out` | gpu-145-146,160-190,195-217,226-235,238-239,256 |

## Recommendations

1. **PP=8 over PP=16:** Moves P2P from Regime 3 (~14.5 GB/s) to Regime 2 (~26 GB/s) — 1.8x bandwidth gain plus half the pipeline stages. Already validated at +38% TFLOP/s.

2. **Enable SHARP for AllReduce:** `NCCL_COLLNET_ENABLE=1` would boost DP=4 gradient sync from ~107 to ~385 GB/s (3.6x).

3. **Scaling beyond 16 nodes is free for P2P:** The fabric saturates at 16 nodes. Going to 32 or 64 nodes adds no additional P2P penalty — the bottleneck is pipeline depth, not node count.
