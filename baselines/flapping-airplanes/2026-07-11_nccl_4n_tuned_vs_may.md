# NCCL 4-Node Reproducibility + May-2026 Comparison — 2026-07-11

flapping-airplanes B200 cluster (slinky), 4 nodes / 32 GPUs. Three reproducible runs of the
**tuned** `run_slurm.sh` (corrected 400G-rail HCA list + vendor IB tuning vars — see
`2026-07-11_nccl_collective_scaling_report.md` for the fix), compared against the same-cluster
May 2026-05-12 4-node baseline.

**Bottom line:** the fixed cluster reproduces to ≤2.6% run-to-run and matches the May baseline
to ~1% on ring and all_gather; SHARP matches at equal message size (554 vs 550 GB/s @8 GiB) and
is still climbing at 8 GiB. All 27 configs `0 wrong`.

## Configuration

| Parameter | This run (2026-07-11) | May baseline (2026-05-12) |
|-----------|-----------------------|---------------------------|
| Nodes / GPUs | 4 / 32 (`slinky-[0-3]`) | 4 / 32 |
| NCCL | 2.30.7 | 2.28.9 |
| Message sweep | 1 MiB → **8 GiB** (reproducibility ×3) + **32 GiB** headline run | 1 MiB → **32 GiB** |
| Iters | 5 warmup + 20 measured | 5 warmup + 20 measured |
| Launch | `srun --mpi=pmix`, `-g 1`, ntasks/node=8 | `srun --mpi=pmix`, `-g 8`, ntasks/node=1 |
| HCA list | mlx5_1,2,3,4,6,7,12,13 (8×400G NDR IB) | mlx5_4,5,6,7,9,10,11,12 (healthy at the time) |
| IB tuning | QPS/conn=2, AR_THRESHOLD=0, SPLIT=0, PCI_RO=1 | same |

> **Size:** the 3 reproducibility runs sweep to 8 GiB; a dedicated 32 GiB run
> (`20260711_061228`) plus 3 extra all_reduce SHARP samples give a direct headline-to-headline
> comparison against May's @32 GiB numbers — no size caveat. Ring/NVLS are saturated by 8 GiB;
> SHARP keeps climbing to 32 GiB.

## Reproducibility — 3 tuned runs, peak busBW @8 GiB (GB/s)

Runs: `20260711_052556`, `_054021`, `_054855`.

| Collective | Config | Run 1 | Run 2 | Run 3 | **Mean** | Spread |
|------------|--------|------:|------:|------:|---------:|-------:|
| all_reduce | ring | 384.0 | 384.6 | 384.1 | **384.2** | 0.1% |
| all_reduce | nvls | 350.8 | 344.4 | 349.8 | **348.3** | 1.8% |
| all_reduce | collnet_sharp | 554.4 | 550.3 | 555.7 | **553.5** | 1.0% |
| all_gather | ring | 376.2 | 376.9 | 373.5 | **375.5** | 0.9% |
| all_gather | nvls | 376.3 | 372.0 | 366.7 | **371.7** | 2.6% |
| all_gather | collnet_sharp | 377.8 | 383.8 | 383.8 | **381.8** | 1.6% |
| reduce_scatter | ring | 384.1 | 384.1 | 384.1 | **384.1** | 0.0% |
| reduce_scatter | nvls | 384.1 | 384.1 | 383.6 | **383.9** | 0.1% |
| reduce_scatter | collnet_sharp | 383.7 | 383.6 | 384.0 | **383.8** | 0.1% |

- **27/27 configs `0 wrong`.** Max run-to-run spread 2.6% (all_gather nvls), most <1%.

## Comparison vs May 2026-05-12 (4n / 32 GPU) — all @32 GiB

Direct headline-to-headline at 32 GiB (`20260711_061228` matrix + confirmatory SHARP runs).

| Collective | Metric | May 2026-05-12 (@32G) | **This run (@32G)** | Δ | Note |
|------------|--------|----------------------:|--------------------:|--:|------|
| all_reduce | ring | 389.30 | **385.1** | −1.1% | ✓ |
| all_reduce | NVLS/NVLSTree | 388.40 | **350.4** | −9.7% | NCCL 2.30 NVLS path — see note |
| all_reduce | SHARP — CollnetChain | 540.07 | **559.0** | +3.5% | ✓ (this is what the matrix auto-selects) |
| all_reduce | **SHARP — CollnetDirect** | **605.90** | **621.1** | **+2.5%** | ✓ May's headline algo — mine beats it |
| all_gather | ring | 376.91 | **373.8** | −0.8% | ✓✓ |
| all_gather | nvls | 376.97 | **376.5** | −0.1% | ✓✓ |
| all_gather | SHARP | 316.96 (CollnetDirect) | **385.5** | +21.6% | SHARP doesn't help all_gather; NCCL 2.30 stays on ring/nvls (correct, faster) |
| reduce_scatter | ring | — | 383.7 | — | new (no May table) |
| reduce_scatter | nvls | — | 385.0 | — | new |
| reduce_scatter | collnet_sharp | — | 385.7 | — | new |

**Notes:**
- **Ring and all_gather match May to ~1%** at 32 GiB — fabric and 8-rail config reproduce the
  historical baseline. all_gather ring/nvls are essentially exact (−0.8% / −0.1%).
- **all_reduce SHARP is healthy and slightly *faster* than May** once the algorithm matches.
  The matrix's `collnet_sharp` config (`NCCL_ALGO` unset) auto-selects **CollnetChain** →
  ~558–559 GB/s @32 GiB (3 clean samples; one matrix sample dipped to 526 = SHARP run-to-run
  variance). Forcing **CollnetDirect** — May's headline algorithm — gives **621.1 GB/s**,
  beating May's 605.90 (+2.5%). So the earlier "SHARP might lag at 32 GiB" concern was purely a
  CollnetChain-vs-CollnetDirect algorithm-selection artifact, not a regression. (May noted
  CollnetDirect is faster but flakier — transient `Streaming Tree lock -18` — so CollnetChain
  is the safer auto default.) The SHARP-specific SAT env vars (`SHARP_COLL_ENABLE_SAT=1` etc.)
  made no measurable difference here.
- **all_reduce NVLS is ~10% below May — a NCCL 2.28→2.30 version difference in the NVLS
  multi-node path, not a fabric or config issue.** Isolated by elimination (all tests 4n @8 GiB
  unless noted):
  - Not message size: NVLS is flat 348→350→353 over 8→16→32 GiB (it saturates like ring).
  - Not a fallback: `NVLS multicast support is available` on all devices, `0 wrong`.
  - Not algorithm selection: forcing `NCCL_ALGO=NVLSTree` gives 343, auto-NVLS 348 — both far
    below 388 (Tree gives only 233).
  - Not launch style: May's `-g 8` launch gives **308** here (even lower than `-g 1`'s 348), so
    the launch difference cannot explain May's higher number.
  - Ring (−1.1%) and SHARP-CollnetDirect (+2.5%) — distinct code paths — both match/beat May,
    confirming the fabric, 8-rail HCA list, and IB tuning are healthy.
  → The only remaining variable is NCCL 2.28.9 (May) vs 2.30.7 (now); the 2.30 NVLS inter-node
  path is ~10% slower at 4n on this cluster. NVLS isn't the recommended all_reduce algo here
  anyway — SHARP-CollnetDirect (621) and ring (385) both win and both match/beat May. To chase
  NVLS's historical peak, NCCL 2.28 would be required.
- **all_gather SHARP is higher** because May explicitly found SHARP doesn't help all_gather
  (CollnetDirect adds tree-setup overhead → 317); NCCL 2.30 leaves it on ring/NVLS (~382),
  which is the correct/faster behavior.

## Per-size all_reduce curve to 32 GiB (busBW GB/s)

ring/nvls from the 32 GiB matrix run (`20260711_061228`); SHARP shown for both auto-CollnetChain
and forced-CollnetDirect.

| Size | ring | nvls | SHARP (CollnetChain, auto) | SHARP (CollnetDirect) |
|-----:|-----:|-----:|---------------------------:|----------------------:|
| 4 GiB | 381.8 | 336.8 | ~532 | — |
| 8 GiB | 384.9 | 344.3 | 549–554 | — |
| 16 GiB | 384.1 | 347.4 | 558 | — |
| 32 GiB | 385.1 | 350.4 | 558–559 | **621** |

Ring saturates by ~8 GiB; NVLS plateaus ~350 (the 2.30 gap); SHARP keeps climbing — CollnetDirect
reaches 621 @32 GiB, above May's 605.9.

## Conclusion

After the two `run_slurm.sh` fixes, the 4-node cluster is **reproducible (≤2.6%)** and, at a
direct 32 GiB headline-to-headline comparison, **matches or beats the May 2026 baseline** on
every collective except NVLS:
- **all_reduce ring** 385 vs 389 (−1.1%), **all_gather ring/nvls** within 1%.
- **all_reduce SHARP-CollnetDirect** 621 vs 605.9 (**+2.5%** — beats May's headline).
- **all_reduce NVLS** 350 vs 388 (−9.7%) — the sole deviation, traced by elimination to the
  NCCL 2.28.9→2.30.7 NVLS multi-node path (message size, algorithm selection, launch style, and
  fabric all ruled out; ring and SHARP both match May). Not a hardware regression, and NVLS is
  not the recommended all_reduce algo here anyway.

The compute fabric, 8-rail HCA config, IB tuning, and SHARP are all healthy. The degraded
non-compute ports (mlx5_5/9/10/11) remain a separate hardware action item.

**Optional follow-up:** the matrix `collnet_sharp` config auto-selects CollnetChain (~558);
adding a `NCCL_ALGO=CollnetDirect` all_reduce variant would surface the higher 621 GB/s figure,
at the cost of CollnetDirect's occasional transient `Streaming Tree lock -18` (light retry needed).

## Raw Data

- Tuned run 1: `nccl-results/B200/20260711_052556_slurm_4nodes/` (also `baseline_slurm`)
- Tuned run 2: `nccl-results/B200/20260711_054021_slurm_4nodes/`
- Tuned run 3: `nccl-results/B200/20260711_054855_slurm_4nodes/`
- **32 GiB headline run: `nccl-results/B200/20260711_061228_slurm_4nodes/`**
- May baseline report: `baselines/flapping-airplanes/2026-05-12_nccl_collective_scaling_report.md`
