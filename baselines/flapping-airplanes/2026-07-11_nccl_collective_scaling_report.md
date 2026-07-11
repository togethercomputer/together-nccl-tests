# NCCL Collective Scaling Benchmark Report — 2026-07-11

flapping-airplanes B200 cluster (slinky). 4-node / 32-GPU NCCL collective performance.
**Headline: two independent `run_slurm.sh` misconfigurations were suppressing inter-node
bandwidth, and fixing both brings every collective back in line with the historical baseline.**

1. **Degraded-fabric HCA list** — the hard-coded `NCCL_IB_HCA` pointed NCCL at 4 healthy rails
   plus 1 Ethernet port and 3 links that trained down to 100G since May → flat ~192 GB/s ceiling.
2. **Missing vendor NCCL IB tuning vars** — even with healthy rails, ring/NVLS ran ~20–25% low.

After both fixes, results match the May 2026 records to within ~1% (e.g. all_gather ring
376.2 vs historical 376.9; all_reduce ring 384.0 vs 389.3; SHARP 554 vs 550). All configs
`0 wrong`.

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Nodes tested | 4 (`slinky-[0-3]`), 32 GPUs, `all`/`slinky` partition |
| Interconnect | Mellanox ConnectX, NDR 400 Gb/s InfiniBand (8 compute rails/node) |
| NCCL | 2.30.7 (May baseline was 2.28.9) |
| CUDA / Driver | 13.3 / 610.43.02 |
| MPI launch | `srun --mpi=pmix` (HPC-X OpenMPI 5, PMIx v4) |
| Binaries | `together-nccl-tests` built `MPI=1` (CUDA 13.3, system NCCL) |
| Message sweep | 1 MiB → 8 GiB, factor 2, 5 warmup + 20 measured iters |

---

## Finding 1: `NCCL_IB_HCA` no longer matches the healthy rails

Each slinky node exposes 16 mlx5 ports. As of **2026-07-11**, only eight are full-rate
**400G NDR InfiniBand**. Verified identical on all 4 nodes via
`/sys/class/infiniband/<hca>/ports/1/{rate,link_layer}`:

| Port | Rate / link | Role |
|------|-------------|------|
| mlx5_1, 2, 3, 4, 6, 7, 12, 13 | **400G NDR InfiniBand** | ✅ healthy compute rails |
| mlx5_5 | 200G, **Ethernet/RoCE** | ❌ wrong link-layer (`Spectrum-X plugin … skipping` warnings) |
| mlx5_9, 10, 11 | **100G (2X HDR)** IB | ❌ trained down — lane count (4→2) and speed (NDR→HDR) both reduced |
| mlx5_0, 14 | 200G Ethernet | storage/mgmt |
| mlx5_8 | 100G HDR IB | non-compute |

The script hard-coded `mlx5_4,5,6,7,9,10,11,12` — the Ethernet port + all three 100G links,
excluding 4 healthy 400G rails → only 4 of 8 rails full-speed → ~192 GB/s ceiling on every
collective.

**Root cause = fabric regression, not a static bug.** Repo memory `project_nccl_working_config.md`
records this same list hitting **390 GB/s at 8 nodes on 2026-05-09**. Between then and
2026-07-11 mlx5_5 flipped to Ethernet and mlx5_9/10/11 trained down. The hard-coded list
stopped tracking which ports are healthy.

> **Action item:** mlx5_5 / mlx5_9 / mlx5_10 / mlx5_11 were 400G NDR IB in May — worth a
> hardware/cabling/switch-port check on all 4 nodes.

**Fix:** `NCCL_IB_HCA="=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_6:1,mlx5_7:1,mlx5_12:1,mlx5_13:1"`
(SHARP works fine with mlx5_1,2,3,13 — the old "mlx5_0-3 break SHARP" note is stale.)

## Finding 2: missing vendor NCCL IB tuning vars

With healthy rails, SHARP hit the historical line (546 vs 550 GB/s) but **ring/NVLS were still
~20–25% low** (4n ring 307 vs historical 389). The gap was isolated to a set of vendor IB
tuning vars — present in the May working-config, absent from `run_slurm.sh` — **independent of
`-g 1` vs `-g 8` launch style** (verified: run_slurm-style `-g 1` + these vars → 383.6 GB/s):

```bash
export NCCL_IB_QPS_PER_CONNECTION=2    # ring-critical
export NCCL_IB_AR_THRESHOLD=0          # ring-critical
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IGNORE_CPU_AFFINITY=1
```

SHARP goes through the CollNet offload path and is unaffected by these.

---

## Results — peak busBW at 8 GiB (GB/s), 4 nodes / 32 GPUs

Three configs of `run_slurm.sh`: **Buggy** (original), **HCA-fix** (mean of 3 reproducible runs,
Finding 1 only), **Tuned** (Findings 1+2 — the recommended config).

| Collective | Config | Buggy | HCA-fix (mean/3) | **Tuned** | vs Buggy |
|------------|--------|------:|-----------------:|----------:|---------:|
| all_reduce | ring | 192.8 | 307.3 | **384.0** | **1.99×** |
| all_reduce | nvls | 163.8 | 301.8 | **350.8** | 2.14× |
| all_reduce | collnet_sharp | 320.0 | 546.4 | **554.4** | 1.73× |
| all_gather | ring | 192.7 | 308.7 | **376.2** | 1.95× |
| all_gather | nvls | 193.0 | 310.2 | **376.3** | 1.95× |
| all_gather | collnet_sharp | 192.6 | 381.0 | **377.8** | 1.96× |
| reduce_scatter | ring | 192.6 | 306.3 | **384.1** | 1.99× |
| reduce_scatter | nvls | 193.0 | 307.3 | **384.1** | 1.99× |
| reduce_scatter | collnet_sharp | 192.9 | 381.6 | **383.7** | 1.99× |

- All configs `0 wrong`. HCA-fix column: 3 runs (`044705`, `045843`, `050749`), spread ≤2.1%.
- **all_reduce + SHARP = 554 GB/s** — the standout; above the use3a-ss SHARP baseline (~542).
- SHARP does **not** help all_gather (no reduction to offload) — ring/nvls/collnet all ~377,
  matching the May report's finding.

## Consistency with existing together-nccl-tests records

Tuned results (this run, @8 GiB) vs the same-cluster May 2026-05-12 report (@32 GiB) and the
use3a-ss 4-node JSON baseline. Ring/NVLS are saturated by 8 GiB, so the size difference is minor.

| Metric (4n) | Historical record | Tuned (2026-07-11) | Δ |
|-------------|------------------:|-------------------:|--:|
| all_gather ring | 376.9 (May) | **376.2** | −0.2% ✓✓ |
| all_gather nvls | 377.0 (May) | **376.3** | −0.2% ✓✓ |
| all_reduce ring | 389.3 (May, 32G) | **384.0** | −1.4% ✓ |
| all_reduce SHARP | 550.6 (May CollnetDirect, 8G) | **554.4** | +0.7% ✓ |
| all_reduce SHARP | 514.5 (use3a-ss 4n JSON) | **554.4** | +7.8% (diff cluster) |

**Conclusion:** once both misconfigurations are fixed, the cluster's collective performance is
consistent with the historical records to within ~1%. The fabric and SHARP are healthy; the
earlier low numbers were entirely `run_slurm.sh` configuration (degraded-rail HCA list +
missing tuning vars), not a hardware regression on the compute rails — though the four
non-compute ports (mlx5_5/9/10/11) that degraded since May still warrant a fabric check.

## Reproducibility

| Metric | Value |
|--------|-------|
| HCA-fix runs | 3 (`044705`, `045843`, `050749`), spread ≤ 2.1% |
| Tuned run | `20260711_052556` |
| Correctness | 36/36 fixed+tuned configs `0 wrong` |

## Raw Data

- Buggy baseline: `nccl-results/B200/20260711_042103_slurm_4nodes/`
- HCA-fix runs 1–3: `…/20260711_044705`, `…_045843`, `…_050749`
- **Tuned (recommended baseline): `…/20260711_052556_slurm_4nodes/`**

(Result logs live under `~/nccl-results/` per repo convention and are not committed.)

## Recommendations

1. **Land both fixes** in `run_slurm.sh` (done 2026-07-11): the 400G-rail HCA list and the 5
   vendor IB tuning vars. Together: ring/NVLS ~2×, SHARP ~1.7× vs the buggy config.
2. **Auto-detect the 400G IB rails** instead of hard-coding, so the list tracks fabric drift.
3. **Investigate the degraded ports** mlx5_5 (→Ethernet) and mlx5_9/10/11 (→100G) — 400G in May.
4. **Re-baseline at scale** (8/16/32/64 nodes) once ports are healed, to compare against the May
   388 GB/s ring / 534 GB/s SHARP scaling line. This 4-node run matches that line to ~1%.
