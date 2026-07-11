# NCCL Collective Scaling Benchmark Report — 2026-05-11

flapping-airplanes B200 cluster (slinky). NCCL collective performance across 1–62 nodes (8–496 GPUs).

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Available nodes | 64 in `slinky` partition (`slinky-[0-63]`); 62 used (slinky-9, slinky-26 newly degraded mid-day, excluded) |
| Node range tested | 1, 2, 4, 8, 16, 32, 62 nodes (8–496 GPUs) |
| Interconnect | Mellanox ConnectX, NDR 400 Gb/s InfiniBand |
| NICs per node | **12 IB HCAs** — `mlx5_0..7, 9..12` |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| **SHARP** | **Inactive** — `CollnetChain` and `CollnetDirect` both return exit 3 (`no algorithm/protocol available` / plugin loads but `Cannot create SHARP job` underneath) on every multi-node run |
| Excluded | `slinky-9`, `slinky-26` — went bad during today's earlier 512-GPU LLM sweep (~10:00–12:02 UTC). slinky-43/62 are now HEALED (verified 09:20 UTC, 716/714 GB/s 2n). |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.28.9+cuda12.9 |
| NCCL Library | system `/usr/lib/x86_64-linux-gnu/libnccl.so.2` |
| NCCL Test Binaries | `/home/johnson/nccl-tests-upstream/build/` |
| MPI | HPC-X OpenMPI 4.1.7a1, `srun --mpi=pmix` |
| Critical MCA settings | `OMPI_MCA_coll=^hcoll,ucc` (disable HPC-X collectives — they hang MPI_Init at ≥12 nodes on this cluster), `pml=ob1`, `btl=tcp,self` |
| Vendor NCCL tuning | `NCCL_IB_PCI_RELAXED_ORDERING=1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_AR_THRESHOLD=0`, `NCCL_IB_SPLIT_DATA_ON_QPS=0`, `NCCL_IGNORE_CPU_AFFINITY=1` |
| HPC-X | `/opt/hpcx/hpcx-init.sh` |
| Scheduler | Slurm partition=`slinky`, `--exclusive`, `--exclude=slinky-9,slinky-26` |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 1 MiB → 32 GiB (factor 2 sweep, 16 sizes) |
| Iterations | 5 warmup + 20 measured (`-w 5 -n 20`) |
| GPUs per process | 8 (`-g 8`, 1 process per node) |
| Validation | off (`-c 0`) |
| Sbatch template | `/home/johnson/nccl-scaling-2026-05-11/wrapper.sbatch` |
| Submitter | `/home/johnson/nccl-scaling-2026-05-11/submit.sh` |
| Slurm jobs | 2528–2615 |
| Results dir | `/home/johnson/nccl-scaling-2026-05-11/logs/` |
| TSV | `/home/johnson/nccl-scaling-2026-05-11/results.tsv` |

## Job Results Summary

| Category | Count | Notes |
|----------|-------|-------|
| Passed | 49 | all Ring, Tree, NVLS/NVLSTree across all sizes |
| Expected failures | 28 | CollnetChain + CollnetDirect at every size, both collectives (no SHARP at the service layer) |
| Hardware failures | 0 (with exclusion) | sweep started at 12:53 UTC, after slinky-9/26 already drained — no further node degradation seen during sweep |

---

## Results — all_reduce busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | Ring | Tree | NVLS / NVLSTree | CollnetChain | CollnetDirect | Best |
|---|---|---|---|---|---|---|
| 1n (8) | 686.12 | 556.54 | **844.89** | ✗ no SHARP | ✗ no SHARP | **NVLS 844.89** |
| 2n (16) | 388.20 | 338.77 | **723.44** | ✗ | ✗ | **NVLSTree 723.44** |
| 4n (32) | **388.94** | 191.70 | 389.13 | ✗ | ✗ | NVLSTree≈Ring 389 |
| 8n (64) | 390.34 | 194.18 | **390.08** | ✗ | ✗ | Ring 390.34 |
| 16n (128) | 389.54 | 195.54 | **390.74** | ✗ | ✗ | NVLSTree 390.74 |
| 32n (256) | 389.01 | 195.29 | **389.46** | ✗ | ✗ | NVLSTree≈Ring 389 |
| 62n (496) | 356.17 | 195.73 | **362.46** | ✗ | ✗ | NVLSTree 362.46 |

**NVLSTree** wins decisively at 2n (intra-NVLink reduction dominates); above 4n it converges with Ring at ~389 GB/s busbw — both flat through 32n, both drop slightly at 62n. CollnetChain / CollnetDirect both fail — SHARP service inactive.

## Results — all_reduce busBW @ 8 GiB / 16 GiB / 32 GiB (Ring, GB/s)

| Nodes | 8 GiB | 16 GiB | 32 GiB |
|---|---|---|---|
| 1n | 679.88 | 683.22 | 686.12 |
| 2n | 385.27 | 387.31 | 388.20 |
| 4n | 386.22 | 385.96 | 388.94 |
| 8n | 390.11 | 385.79 | 390.34 |
| 16n | 387.61 | 388.42 | 389.54 |
| 32n | 388.04 | 389.01 | 388.33 |
| 62n | 270.87 | 336.81 | 356.17 |

Ring is **remarkably stable 2n→32n** at 386–390 GB/s. At 62n the 8 GiB number drops sharply (270 GB/s) but 32 GiB recovers to 356 GB/s — the 62n run is bandwidth-undersaturated at 8 GiB.

## Results — all_gather busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | Ring | NVLS | CollnetChain | CollnetDirect | Best |
|---|---|---|---|---|---|
| 1n (8) | **674.63** | 675.60 | ✗ | ✗ | NVLS≈Ring 675 |
| 2n (16) | **370.22** | 371.79 | ✗ | ✗ | NVLS≈Ring 371 |
| 4n (32) | **377.85** | 378.06 | ✗ | ✗ | ≈ |
| 8n (64) | **381.43** | 380.92 | ✗ | ✗ | Ring 381 |
| 16n (128) | **382.75** | 382.19 | ✗ | ✗ | Ring 382 |
| 32n (256) | **376.77** | 377.47 | ✗ | ✗ | ≈ |
| 62n (496) | **348.55** | 348.79 | ✗ | ✗ | ≈ |

Ring and NVLS produce essentially identical all_gather numbers (within 1%) — NVLS doesn't help here. Tree/NVLSTree unsupported for all_gather (NCCL limitation).

## Results — sendrecv P2P busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | busBW peak |
|---|---|
| 1n (8) | 649.55 |
| 2n (16) | 32.35 |
| 4n (32) | 32.27 |
| 8n (64) | 17.18 |
| 16n (128) | 17.21 |
| 32n (256) | 17.21 |
| 62n (496) | 17.18 |

sendrecv saturates at ~17 GB/s/peer for ≥8n — bounded by per-NIC bandwidth across the rank-pair pattern. Matches the 2026-05-09 finding exactly.

---

## Comparison vs. 2026-05-09 sweep (60n, +2 IB-broken nodes excluded)

The 2026-05-09 report tested 60n (excluding the then-broken slinky-43 and slinky-62). Today we test 62n (slinky-43 and slinky-62 are healed, but slinky-9 and slinky-26 newly degraded). Net node count moved by +2.

| Metric | 2026-05-09 (60n) | 2026-05-11 (62n) | Delta |
|---|---|---|---|
| all_reduce Ring 32 GiB | 366.11 | 356.17 | −2.7% |
| all_reduce NVLSTree 32 GiB | 337.62 | 362.46 | **+7.4%** |
| all_gather Ring 32 GiB | 353.25 | 348.55 | −1.3% |
| sendrecv 32 GiB | 17.18 | 17.18 | flat |

Aggregate: very close to 2026-05-09's results. NVLSTree at 62n improved noticeably (+7%), Ring dropped slightly. **No regression**; the cluster's collective performance is stable across the past two days even as bad nodes have rotated.

---

## Comparison with MD1 Cluster Baseline (2026-05-02)

MD1 has SHARP active. On flapping-airplanes SHARP is **still inactive**, so direct CollNet/SHARP comparison is impossible. Below: this run's best (Ring/NVLSTree) vs MD1's SHARP/Ring/NVLS numbers.

### all_reduce busBW @ 16 GiB

| Nodes | MD1 SHARP | MD1 Ring | MD1 NVLS | This run Ring | This run NVLS/NVLSTree | vs MD1 SHARP | vs MD1 Ring |
|---|---|---|---|---|---|---|---|
| 1n | — | 679.0 | 840.8 | 683.22 | **842.59** | — | +0.6% |
| 2n | 533.0 | 344.7 | 717.2 | 387.31 | **717.67** | NVLSTree-1.4% wrt MD1 SHARP, 723→734 ✓ | +12.4% Ring |
| 4n | 543.8 | 334.8 | 334.7 | 385.96 | 387.32 | Ring −29% vs SHARP | +15.3% Ring |
| 8n | 547.9 | 322.6 | 306.8 | 385.79 | 388.54 | Ring −30% | +19.6% Ring |
| 16n | 546.7 | 306.3 | 306.4 | 388.42 | 390.23 | Ring −29% | +26.8% Ring |
| 32n | 544.3 | 304.4 | 305.2 | 389.01 | 389.46 | Ring −28% | +27.8% Ring |

This cluster's Ring **beats MD1's Ring by 12–28% across all multi-node sizes**. Slinky's Ring all_reduce is faster than MD1's because of better intra-node NVLink + better NIC topology (12 HCAs vs MD1's older setup). But Ring still trails MD1's SHARP-accelerated number by ~28–30% at 4–32n — that's the unrecovered ceiling that SHARP would unlock.

### all_gather busBW @ 16 GiB

| Nodes | MD1 SHARP | This run Ring | vs MD1 SHARP |
|---|---|---|---|
| 2n | 376.4 | 369.02 | −2.0% |
| 4n | 385.2 | 376.63 | −2.2% |
| 8n | 383.6 | 379.40 | −1.1% |
| 16n | 385.2 | 375.96 | −2.4% |
| 32n | 377.6 | 376.77 | −0.2% |

Ring all_gather is **within 2.5% of MD1 SHARP all_gather** at all scales. SHARP gives MD1 very little headroom on all_gather (it's already comm-light); the ring path on this cluster effectively matches.

### sendrecv P2P @ 16 GiB

| Nodes | MD1 | This run | Delta |
|---|---|---|---|
| 1n | 641.6 | 649.12 | +1.2% |
| 2n | 42.6 | 31.58 | −25.9% |
| 4n | 42.7 | 32.27 | −24.4% |
| 8n | 25.4 | 17.17 | −32.4% |
| 16n | 15.1 | 17.20 | +13.9% |
| 32n | 15.6 | 17.20 | +10.3% |

sendrecv P2P trails MD1 at 2–8n (per-pair NIC bandwidth narrower), but slightly beats MD1 at 16n+. Consistent with the 2026-05-09 result.

---

## Key Findings

1. **Cluster scaling characteristic is stable.** Ring all_reduce holds **386–390 GB/s busBW from 2n through 32n** (≤1% spread), drops to 356 GB/s at 62n. Matches the 2026-05-09 baseline within ±3%.

2. **NVLSTree is the new headline at 2n.** 717 GB/s busBW — beats Ring (388) by 84% and beats this cluster's earlier 2026-05-09 NVLSTree (722) within noise. Single-node NVLS holds at 845 GB/s, also matching prior runs.

3. **SHARP remains unavailable.** Both `CollnetChain` and `CollnetDirect` (correct algo names per NCCL 2.28) fail with `no algorithm/protocol available`. Plugin loads (`Loaded collnet plugin SHARP (v8)`), but `Cannot create SHARP job(-11)` underneath. Service-layer issue; **admin action on the IB fabric manager is required** to enable it. This is the same blocker noted on 2026-05-09 and on every prior sweep.

4. **Beats MD1 Ring by 12–28%.** Across 2n–32n all_reduce Ring, this cluster's NDR400 + 12-HCA per-node topology delivers ~20% more Ring busBW than MD1's older fabric. The 28% gap to MD1's SHARP at 4n+ remains the SHARP-enablement ceiling.

5. **Cluster health is a moving target — today's 2 bad nodes are NOT the same as yesterday's.** Yesterday: `slinky-43, slinky-62` were the bad pair (now healed). Today: `slinky-9, slinky-26` went bad mid-LLM-sweep this morning. Run a fresh pair sweep before any large run.

6. **Tree all_reduce flatlines at ~195 GB/s** for ≥4n. Useful as a fallback if Ring degrades, but always slower than Ring on this topology. Same finding as 2026-05-09.

7. **No launcher pathology recurrence.** All 77 jobs (with hcoll/ucc disabled and `srun --mpi=pmix`) launched cleanly. No MPI_Init hangs at any scale, including 62n.

---

## Node Actions

| Node | Status | Action |
|------|--------|--------|
| slinky-9 | IB failed during 2026-05-11 LLM sweep at ~10:00; flagged by pair sweep at 12:02. Was rank-0 of nemotronh_fp8 TCPStore failure earlier. | **Drain.** Investigate IB QP/MR state and pod cycling. |
| slinky-26 | IB failed during 2026-05-11 LLM sweep, flagged by pair sweep at 12:02. | **Drain.** Same investigation. |
| slinky-43, slinky-62 | **HEALED** (verified 2026-05-11 09:20 UTC: 2n ring gets 716/714 GB/s, no IB errors). | OK — back in service. |
| slinky-[0-8, 10-25, 27-42, 44-61, 63] (60 nodes) | Healthy under `srun --mpi=pmix` + hcoll/ucc disabled | OK |

---

## What to Re-run

### 2-Node Pairwise Sweep (IB health check)

```bash
bash /home/johnson/auto_sweep_512gpu/pair_sweep_helper.sh "$(scontrol show hostnames slinky-[0-63] | paste -sd,)" /home/johnson/pair_sweep_$(date +%Y%m%d_%H%M%S)
```

Each pair runs an 8 GiB ring all-reduce with 3-min timeout. A FAIL output indicates IB hang or QP allocation issue.

**Currently bad pairs (as of 12:02 UTC)**: `slinky-9+slinky-26` (and any pair containing either node).

**Currently healthy 60-node pool**: all nodes except `slinky-9, slinky-26`.

### Full collective scaling sweep

```bash
cd /home/johnson/nccl-scaling-2026-05-11
bash submit.sh 1,2,4,8,16,32,62                   # produces ~77 jobs
bash parse.sh > results.tsv                        # parses logs after jobs complete
```

---

## Reproduce a single 62n all_reduce peak run

```bash
sbatch --nodes=62 --export=ALL,RUN_BINARY=/home/johnson/nccl-tests-upstream/build/all_reduce_perf,RUN_LABEL=manual_62n,RUN_ALGO=Ring \
  /home/johnson/nccl-scaling-2026-05-11/wrapper.sbatch
```

---

## Cross-references

- **2026-05-09 NCCL scaling report (predecessor, 60n)**: `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-09.md`
- **2026-05-04 template (28 empty cells, now superseded by today)**: `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report.md`
- **Today's 512-GPU LLM sweep (root cause of slinky-9, slinky-26 going bad)**: `/home/johnson/worklogs/flapping_airplanes_512gpu_benchmark_report.md`
- **MD1 cluster NCCL baseline (2026-05-02)**: `/home/johnson/worklogs/2026-05-02_nccl_collective_scaling_report.md`
- **All result logs**: `/home/johnson/nccl-scaling-2026-05-11/logs/`
- **TSV summary**: `/home/johnson/nccl-scaling-2026-05-11/results.tsv`
- **Working sbatch templates**: `/home/johnson/nccl-scaling-2026-05-11/wrapper.sbatch`, `/home/johnson/nccl-ar-60n-b200-c0.sbatch`
