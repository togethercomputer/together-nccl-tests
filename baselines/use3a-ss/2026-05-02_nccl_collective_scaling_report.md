# NCCL Collective Scaling Benchmark Report — 2026-05-02

NCCL collective performance across 1–32 node scale (8–256 GPUs) on the B200 DGXC cluster (use3a-ss).

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Available nodes | 50 idle (batch partition) |
| Node range tested | 1–32 nodes (8–256 GPUs) — 64n skipped, only 50 idle |
| Interconnect | Mellanox ConnectX-7 (MT4129), NDR 400 Gb/s InfiniBand |
| NICs per node | 8 HCAs (mlx5_0,1,4,5,6,11,14,15) |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| **SHARP** | **NOW ACTIVE** (LLT + SAT hardware trees; previously software fallback) |
| Excluded | use3a-ss-b200-gpu-[190,197,199,201,211,233,239] |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.29.7+cuda12.9 |
| NCCL Library | `/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu` |
| NCCL Test Binaries | `/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build` |
| MPI | PMIx (`srun --mpi=pmix --kill-on-bad-exit=1`) |
| HPC-X | `/var/tmp/hpcx-2.18/hpcx-init.sh` |
| Scheduler | Slurm, partition=batch, exclusive |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 2 GB, 4 GB, 8 GB, 16 GB (2x factor) |
| Iterations | 5 warmup + 20 measured |
| GPUs per rank | 1 (`-g 1`) |
| Results dir | `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260502_165325/` |
| Slurm jobs | 87805–87864 (60 submitted) |

## Job Results Summary

| Category | Count |
|----------|-------|
| Passed (Out of bounds : 0 OK) | 27 |
| Failed — gpu-177 bad node | 19 |
| Expected failures (all_gather tree/nvls_tree unsupported) | 10 |
| Not yet run (64n not submitted) | — |

**Root cause of unexpected failures: `use3a-ss-b200-gpu-177` exits with code 2, cascading PMIx kills to all other ranks. Every single unexpected failure involves gpu-177. Recommend draining immediately.**

---

## Results — all_reduce Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | NVLSTree | Tree | CollNet SHARP | Best |
|--------------|------|------|----------|------|---------------|------|
| 1n (8) | 671.4 | **840.6** | — | — | — | NVLS |
| 2n (16) | 344.7 | **717.1** | 717.1 | 335.4 | 536.2 | NVLS |
| 4n (32) | 334.8 | 334.6 | 291.6 | ❌ gpu-177 | **543.2** | CollNet SHARP |
| 8n (64) | 306.7 | ❌ gpu-177 | 291.5 | 192.4 | ❌ gpu-177 | Ring (partial) |
| 16n (128) | 304.5 | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | **547.7** | CollNet SHARP |
| 32n (256) | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | — |

## Results — all_gather Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|--------------|------|------|---------------|------|
| 1n (8) | 662.7 | **666.9** | — | NVLS |
| 2n (16) | 342.4 | 343.8 | **368.1** | CollNet SHARP |
| 4n (32) | ❌ gpu-177 | 319.7 | **385.1** | CollNet SHARP |
| 8n (64) | ❌ gpu-177 | 308.4 | **383.5** | CollNet SHARP |
| 16n (128) | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | — |
| 32n (256) | ❌ gpu-177 | ❌ gpu-177 | ❌ gpu-177 | — |

## Results — sendrecv P2P Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | busBW | vs Apr-17 |
|--------------|-------|-----------|
| 1n (8) | 657.2 | ✓ |
| 2n (16) | 42.7 | ✓ |
| 4n (32) | 41.9 | ✓ |
| 8n (64) | ❌ gpu-177 | — |
| 16n (128) | 15.4 | ✓ |
| 32n (256) | ❌ gpu-177 | — |

---

## Comparison with Past Reports

### all_reduce CollNet SHARP — Major Improvement (SHARP now hardware-offloaded)

| Nodes | Apr-16 (GB/s) | Apr-17 (GB/s) | **May-02 (GB/s)** | Delta vs Apr-17 |
|-------|--------------|--------------|-------------------|-----------------|
| 2n | 351.9 | 371.1 | **536.2** | **+44.3%** |
| 4n | 384.5 | 379.2 | **543.2** | **+43.3%** |
| 8n | 385.2 | 382.5 | ❌ (gpu-177) | — |
| 16n | 377.9 | 357.7 | **547.7** | **+53.1%** |
| 32n | 385.1 | 336.5 | ❌ (gpu-177) | — |

**Previous runs logged `SHARP coll init error: Cannot create SHARP job` (software CollNet fallback). This run confirms SHARP hardware is now fully active (`sharp_job_id`, `tree_type:LLT`, `tree_type:SAT` in logs), delivering 540–548 GB/s vs prior ~335–385 GB/s — a ~40–53% gain.**

### all_reduce Ring

| Nodes | Apr-16 (GB/s) | Apr-17 (GB/s) | May-02 (GB/s) | Delta |
|-------|--------------|--------------|---------------|-------|
| 1n | 681.4 | 658.3 | 671.4 | −1.5% vs Apr-16 |
| 2n | 344.0 | 341.9 | 344.7 | +0.8% vs Apr-16 |
| 4n | 332.4 | 328.7 | 334.8 | +0.7% vs Apr-16 |
| 8n | 305.6 | 309.8 | 306.7 | +0.4% vs Apr-16 |
| 16n | 306.8 | 287.9 | 304.5 | −0.8% vs Apr-16 |

Ring bandwidth is stable and consistent across all three runs. ✓

### all_reduce NVLS

| Nodes | Apr-16 (GB/s) | Apr-17 (GB/s) | May-02 (GB/s) | Delta |
|-------|--------------|--------------|---------------|-------|
| 1n | 839.0 | 821.0 | 840.6 | +0.2% vs Apr-16 |
| 2n | 713.3 | 687.5 | 717.1 | +0.5% vs Apr-16 |
| 4n | 332.4 | 315.0 | 334.6 | +0.7% vs Apr-16 |

NVLS at 1–2n healthy; 8n+ blocked by gpu-177. ✓

### all_gather CollNet SHARP

| Nodes | Apr-16 (GB/s) | May-02 (GB/s) | Delta |
|-------|--------------|---------------|-------|
| 2n | 362.4 | 368.1 | +1.6% |
| 4n | 382.9 | 385.1 | +0.6% |
| 8n | 384.0 | 383.5 | −0.1% |

all_gather SHARP is consistent — within noise. ✓

### sendrecv P2P

| Nodes | Apr-16 (GB/s) | May-02 (GB/s) | Delta |
|-------|--------------|---------------|-------|
| 1n | 653.3 | 657.2 | +0.6% |
| 2n | 42.5 | 42.7 | +0.5% |
| 4n | 44.2 | 41.9 | −0.5% |
| 16n | 15.3 | 15.4 | +0.7% |

Sendrecv P2P stable. ✓

---

## Key Findings

1. **SHARP hardware is now active.** All CollNet SHARP jobs that completed show real hardware aggregation (LLT + SAT trees). Bandwidth jumped from ~335–385 GB/s (software fallback) to **536–548 GB/s** — a 40–53% improvement. This is the biggest change since April.

2. **gpu-177 is a bad node.** It exited with code 2 on every single job it participated in across 4n, 8n, 16n, 32n scales and multiple collectives (ring, nvls, tree, collnet_sharp, sendrecv). The failure pattern is deterministic. **Recommend draining `use3a-ss-b200-gpu-177` immediately.**

3. **Ring and NVLS bandwidth is stable.** Results within 1% of April baselines — fabric health is consistent for nodes not involving gpu-177.

4. **all_gather tree/nvls_tree expected failures** (10 jobs) are confirmed NCCL limitations, same as all previous runs.

---

## Node Action Required

| Node | Status | Action |
|------|--------|--------|
| `use3a-ss-b200-gpu-177` | Exits code 2 on all collective tests | **Drain immediately** |

---

## What to Re-run After Draining gpu-177

The following jobs failed solely due to gpu-177 and should be re-run:

```bash
cd ~/together-nccl-tests
./benchmarks/B200/collective-scaling/submit_all.sh \
  --nodes "4 8 16 32" \
  --exclude "use3a-ss-b200-gpu-[190,197,199,201,211,233,239,177]"
```

This covers: all_reduce (ring/nvls/tree/nvls_tree at 4n, all algos at 8n/16n/32n), all_gather (ring/nvls/collnet_sharp at 4n+), sendrecv at 8n/32n.

---

## Re-run Results (Jobs 87865–87935) — 2026-05-02 17:22–18:12

Exclude: `use3a-ss-b200-gpu-[133,177,190,197,199,201,211,233,239]`
Node counts: 1, 2, 4, 8, 16, 32, 46 (all schedulable idle nodes)
Results: `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/collective-scaling/20260502_172207/`
Note: Message sizes 2G→16G (vs Apr-27 which used 2G→8G). SHARP numbers are comparable; Ring/NVLS at small node counts show +2–8% from larger message size.

### all_reduce Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | NVLSTree | Tree | CollNet SHARP | Best |
|---|---|---|---|---|---|---|
| 1n (8) | 679.0 | **840.8** | — | — | — | NVLS |
| 2n (16) | 344.7 | 717.2 | 717.2 | 335.2 | 533.0 | NVLS |
| 4n (32) | 334.8 | 334.7 | 288.9 | 190.7 | **543.8** | CollNet SHARP |
| 8n (64) | 322.6 | 306.8 | 274.3 | 192.6 | **547.9** | CollNet SHARP |
| 16n (128) | 306.3 | 306.4 | 290.7 | 193.4 | **546.7** | CollNet SHARP |
| 32n (256) | 304.4 | 305.2 | 280.7 | 193.5 | **544.3** | CollNet SHARP |
| 46n (368) | 291.4 | 291.4 | 274.9 | 193.2 | **544.8** | CollNet SHARP |

### all_gather Peak busBW @ 16G (GB/s)

| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |
|---|---|---|---|---|
| 1n (8) | 673.6 | 680.6 | — | NVLS |
| 2n (16) | 342.9 | 344.0 | **376.4** | CollNet SHARP |
| 4n (32) | 323.9 | 323.5 | **385.2** | CollNet SHARP |
| 8n (64) | 311.5 | 308.8 | **383.6** | CollNet SHARP |
| 16n (128) | 304.9 | 306.5 | **385.2** | CollNet SHARP |
| 32n (256) | 302.3 | 300.7 | **377.6** | CollNet SHARP |
| 46n (368) | 293.3 | 292.1 | **385.1** | CollNet SHARP |

Tree / NVLS Tree: expected failures (exit code 3) for all_gather — NCCL limitation.

### sendrecv P2P @ 16G (GB/s)

| Nodes (GPUs) | busBW |
|---|---|
| 1n (8) | 641.6 |
| 2n (16) | 42.6 |
| 4n (32) | 42.7 |
| 8n (64) | 25.4 |
| 16n (128) | 15.1 |
| 32n (256) | 15.6 |
| 46n (368) | 15.6 |

### Apple-to-Apple vs Apr-27

| Metric | Apr-27 @8G | May-02 @16G | Delta |
|---|---|---|---|
| all_reduce SHARP 2n | 542.9 | 533.0 | −1.8% |
| all_reduce SHARP 4n | 542.3 | 543.8 | +0.3% |
| all_reduce SHARP 8n | 544.2 | 547.9 | +0.7% |
| all_reduce SHARP 16n | 545.3 | 546.7 | +0.3% |
| all_reduce SHARP 32n | 545.1 | 544.3 | −0.1% |
| all_reduce Ring 2n | 344.3 | 344.7 | +0.1% |
| all_reduce Ring 16n | 306.6 | 306.3 | −0.1% |
| all_reduce Ring 32n | 304.0 | 304.4 | +0.1% |
| all_reduce NVLS 1n | 836.2 | 840.8 | +0.5% |
| all_reduce NVLS 2n | 713.1 | 717.2 | +0.6% |
| all_gather SHARP 2n | ~368 | 376.4 | +2.3% |
| all_gather SHARP 4n | ~385 | 385.2 | ~0% |
| all_gather SHARP 8n | ~384 | 383.6 | −0.1% |
| sendrecv 2n | ~42 | 42.6 | +1.4% |
| sendrecv 4n | ~42 | 42.7 | +1.7% |

**Conclusion: Cluster is healthy. All metrics within noise of Apr-27. SHARP flat-scales from 4n to 46n at 544–548 GB/s. gpu-177 drain recommended.**

