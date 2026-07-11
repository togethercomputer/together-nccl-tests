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
| Message sweep | 1 MiB → **8 GiB** | 1 MiB → **32 GiB** |
| Iters | 5 warmup + 20 measured | 5 warmup + 20 measured |
| Launch | `srun --mpi=pmix`, `-g 1`, ntasks/node=8 | `srun --mpi=pmix`, `-g 8`, ntasks/node=1 |
| HCA list | mlx5_1,2,3,4,6,7,12,13 (8×400G NDR IB) | mlx5_4,5,6,7,9,10,11,12 (healthy at the time) |
| IB tuning | QPS/conn=2, AR_THRESHOLD=0, SPLIT=0, PCI_RO=1 | same |

> **Size caveat:** this sweep tops out at 8 GiB, the May headline is @32 GiB. Ring and NVLS
> are saturated by 8 GiB so the comparison is fair (verified: NVLS is flat 348→353 over
> 8→32 GiB); SHARP keeps climbing past 8 GiB, so its like-for-like comparison uses the May
> @8 GiB figure.

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

## Comparison vs May 2026-05-12 (4n / 32 GPU)

| Collective | Metric | May 2026-05-12 | **This run (mean)** | Δ | Note |
|------------|--------|---------------:|--------------------:|--:|------|
| all_reduce | ring | 389.30 (@32G) | **384.2** | −1.3% | saturated — fair compare ✓ |
| all_reduce | NVLS/NVLSTree | 388.40 (@32G) | **348.3** | −10.3% | see note below |
| all_reduce | SHARP @8 GiB | 550.55 (CollnetDirect) | **553.5** | +0.5% | matched size ✓ |
| all_reduce | SHARP @32 GiB | 605.90 (CollnetDirect) | — (sweep capped 8G) | — | mine still climbing at 8G |
| all_gather | ring | 376.91 (@32G) | **375.5** | −0.4% | ✓✓ |
| all_gather | nvls | 376.97 (@32G) | **371.7** | −1.4% | ✓ |
| all_gather | SHARP | 316.96 (CollnetDirect) | **381.8** | +20.5% | NCCL 2.30 falls back to ring/nvls (SHARP doesn't help all_gather); mine ≈ ring, better than May's CollnetDirect overhead |

**Notes:**
- **Ring and all_gather match May to ~1%** — the fabric and 8-rail config reproduce the
  historical baseline. all_gather ring 375.5 vs 376.9 is essentially exact.
- **all_reduce SHARP matches at equal size** (553.5 vs 550.55 @8 GiB) and is still rising at
  8 GiB (curve: 391→398→549→554 over 1→8 GiB), so a 32 GiB sweep would approach May's ~606.
- **all_reduce NVLS is ~10% below May — a NCCL 2.28→2.30 version difference in the NVLS
  multi-node path, not a fabric or config issue.** Isolated by elimination (all tests 4n @8 GiB
  unless noted):
  - Not message size: NVLS is flat 348→350→353 over 8→16→32 GiB (it saturates like ring).
  - Not a fallback: `NVLS multicast support is available` on all devices, `0 wrong`.
  - Not algorithm selection: forcing `NCCL_ALGO=NVLSTree` gives 343, auto-NVLS 348 — both far
    below 388 (Tree gives only 233).
  - Not launch style: May's `-g 8` launch gives **308** here (even lower than `-g 1`'s 348), so
    the launch difference cannot explain May's higher number.
  - Ring (−1.3%) and SHARP (+0.5%) — distinct code paths — both match May, confirming the
    fabric, 8-rail HCA list, and IB tuning are healthy.
  → The only remaining variable is NCCL 2.28.9 (May) vs 2.30.7 (now); the 2.30 NVLS inter-node
  path is ~10% slower at 4n on this cluster. NVLS isn't the recommended all_reduce algo here
  anyway — SHARP (553) and ring (384) both win and both match May. To chase NVLS's historical
  peak, NCCL 2.28 would be required.
- **all_gather SHARP is higher** because May explicitly found SHARP doesn't help all_gather
  (CollnetDirect adds tree-setup overhead → 317); NCCL 2.30 leaves it on ring/NVLS (~382),
  which is the correct/faster behavior.

## Per-size all_reduce curve (tuned run 1, busBW GB/s)

| Size | ring | nvls | collnet_sharp |
|-----:|-----:|-----:|--------------:|
| 512 MiB | 362.9 | 362.3 | 377.6 |
| 1 GiB | 379.0 | 325.5 | 391.4 |
| 2 GiB | 374.1 | 337.3 | 398.4 |
| 4 GiB | 376.1 | 347.2 | 548.6 |
| 8 GiB | 384.0 | 350.8 | 554.4 |

Ring plateaus by ~8 GiB; SHARP keeps climbing (→ would approach May's 605.9 @32 GiB).

## Conclusion

After the two `run_slurm.sh` fixes, the 4-node cluster is **reproducible (≤2.6%)** and
**consistent with the May 2026 baseline to ~1%** on ring/all_gather, with SHARP matching at
equal message size. The only >5% deviation (all_reduce NVLS, −10%) was traced by elimination
to the NCCL 2.28.9→2.30.7 NVLS multi-node path (ruling out message size, algorithm selection,
launch style, and fabric — ring and SHARP both match May), not a hardware regression. The
compute fabric and SHARP are healthy; the degraded non-compute ports (mlx5_5/9/10/11) remain a
separate hardware action item.

## Raw Data

- Tuned run 1: `nccl-results/B200/20260711_052556_slurm_4nodes/` (also `baseline_slurm`)
- Tuned run 2: `nccl-results/B200/20260711_054021_slurm_4nodes/`
- Tuned run 3: `nccl-results/B200/20260711_054855_slurm_4nodes/`
- May baseline report: `baselines/flapping-airplanes/2026-05-12_nccl_collective_scaling_report.md`
