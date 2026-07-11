---
name: NCCL benchmark scripts and workflow
description: How to run NCCL benchmarks on Slurm clusters using together-nccl-tests — scripts, test matrix, straggler detection
type: project
originSessionId: 959acd8b-aa46-4d21-b222-be7352c270dc
---
## Repository Locations
- `~/together-nccl-tests` (branch: johnson-dev) — benchmark scripts
- `~/together-dgxc-benchmarking` (branch: johnson_dev) — worklogs and history

## Main Entry Point
`~/together-nccl-tests/benchmarks/run_slurm.sh`

Generates and submits one `sbatch --wait` job per test config. Waits for each to complete, collects results, prints summary with optional baseline comparison.

## Test Matrix (9 tests)
Defined in `benchmarks/lib/common.sh` — NCCL_TEST_MATRIX:
| test_name | label | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE |
|-----------|-------|-----------|------------------|---------------------|
| all_reduce | ring | RING | 0 | 0 |
| all_reduce | nvls | (unset) | 1 | 0 |
| all_reduce | collnet_sharp | (unset) | 0 | 1 |
| all_gather | ring | RING | 0 | 0 |
| all_gather | nvls | (unset) | 1 | 0 |
| all_gather | collnet_sharp | (unset) | 0 | 1 |
| reduce_scatter | ring | RING | 0 | 0 |
| reduce_scatter | nvls | (unset) | 1 | 0 |
| reduce_scatter | collnet_sharp | (unset) | 0 | 1 |

**Important**: For non-ring configs, script explicitly `unset NCCL_ALGO` in case cluster sets it globally.

## Results Directory
Pattern: `${RESULTS_BASE}/${GPU_TYPE}/${TIMESTAMP}_slurm_${NNODES}nodes/`  
Output files: `all_reduce_ring.out`, `all_reduce_nvls.out`, `all_reduce_collnet_sharp.out`, etc.

## Baseline Comparison
- Set baseline: `--set-baseline` flag creates symlink `baseline_slurm → latest run`
- Compare: pass `--baseline DIR` or it auto-discovers `${RESULTS_BASE}/${GPU_TYPE}/baseline_slurm`
- Summary shows: Baseline(GB/s) | Current(GB/s) | Delta(GB/s) | Delta(%)

## Straggler Detection Script
`~/together-nccl-tests/stragglers/find_stragglers.py`

Multi-phase approach:
1. **Phase 1**: Disjoint 4-node groups, ring + collnet_sharp, classify with z-scores
2. **Phase 3**: Localization — pair each suspect with 3 healthy nodes
3. **Phase 4**: Cross-pair check — test 2 healthy 4n groups as 8n to find inter-link issues
4. **Phase 5**: Confirmation retest for marginal candidates
5. **Phase 6**: IB per-HCA sweep with ib_write_bw (opt-in, `--with-ib-sweep`)

Thresholds: z < -2 = suspect, z < -5 = severe

Hard-coded to use3a-ss cluster paths — needs adaptation for slinky cluster.

## B200 Performance Baselines (flapping-airplanes/slinky, 2026-05-01)
All at 8GB message size, all_reduce ring (only scale tested so far):
| Scale | Ring | Notes |
|-------|------|-------|
| 2n (16 GPU) | **375 GB/s** | slinky-0,1 — healthy, confirmed working |
| 4n+ | ❌ HANG | ibv_reg_mr blocked by Slurm cgroup memlock limit |

**Blocking issue**: `ulimit -l unlimited` silently fails on slinky (Slurm cgroup enforces LimitMEMLOCK).
NCCL init hangs in ibv_reg_mr at 4+ nodes. Zero output, hits 1h time limit.
Fix: ops ticket — set `LimitMEMLOCK=infinity` in Slurm stepd cgroup config.

**Healthy nodes (33, from 3× straggler sweep 2026-05-01)**:
slinky-0,1,2,3,4,5,10,11,12,15,16,19,20,22,23,25,27,28,29,30,31,32,35,36,39,40,44,45,46,53,54,56,58

**Bad nodes (20, drain candidates)**:
slinky-6,7,8,9,13,21,24,26,33,34,37,38,41,42,43,48,50,52,60,61

Scaling sweep script (ready to re-run once memlock fixed):
`/data/home/johnson/nccl-results/run_scaling_sweep.sh`
Daily report: `/data/home/johnson/nccl-results/flapping-airplanes/daily-report-20260501.md`

## B200 Performance Baselines (use3a-ss, 2026-04-27 canonical)
All at 8GB message size, all_reduce:
| Scale | Ring | NVLS | CollNet SHARP | Best |
|-------|------|------|---------------|------|
| 1n (8 GPU) | 686.6 | **836.2** | — | NVLS |
| 2n (16 GPU) | 344.3 | **713.1** | 542.9 | NVLS≈NVLSTree |
| 4n (32 GPU) | 311.2 | 333.3 | **542.3** | SHARP |
| 8n (64 GPU) | 308.6 | 307.6 | **544.2** | SHARP |
| 16n (128 GPU) | 306.6 | 303.8 | **545.3** | SHARP |
| 32n (256 GPU) | 304.0 | 304.5 | **545.1** | SHARP |
| 64n (512 GPU) | 287.9 | 274.2 | **542.1** | SHARP |

## Algorithm Recommendations
- **1-2 nodes**: NCCL_NVLS_ENABLE=1 (NVLS wins via NVSwitch intra-node reduction)
- **4+ nodes**: NCCL_COLLNET_ENABLE=1 (SHARP/CollNet flat scaling, Ring degrades badly)
- **Never force NCCL_ALGO=RING at scale** — ~50% bandwidth loss vs SHARP at 4+ nodes
- **NVLS collapses at 64+ nodes** — don't use NVLS_ENABLE at large scale

## Key Worklogs
- `2026-04-27_nccl_rebaseline_report.md` — **canonical baseline** after SHARP enabled
- `2026-04-29_straggler_report.md` — 8 bad nodes on use3a-ss, methodology
- `2026-04-09_nccl_benchmark.md` — original comprehensive study

## History: SHARP on use3a-ss
- Before 04-27: CollNet fallback only (sharpd not running), ~383-385 GB/s at 4-64n
- After 04-27: Real SHARP offload, +40-49% gain to ~542-545 GB/s at 4-64n
- Status on slinky: unknown — needs investigation
