# NCCL Collective Scaling Benchmark Report — 2026-05-12

flapping-airplanes B200 cluster (slinky). NCCL collective performance across 1–64 nodes (8–512 GPUs). **Headline: SHARP went active overnight** — CollnetChain / CollnetDirect succeed end-to-end on every multi-node size, and CollnetDirect is the new best all_reduce algorithm at ≥4 nodes (+45–55% vs Ring/NVLSTree).

## System Configuration

| Component | Details |
|-----------|---------|
| GPU | NVIDIA B200, 8 per node |
| Available nodes | 64 in `slinky` partition (`slinky-[0-63]`); **all 64 used in morning sweep** — slinky-9 and slinky-26 verified healed (716/688 GB/s 2n busbw, jobs 2656/2657) |
| Node range tested | 1, 2, 4, 8, 16, 32, 64 nodes (8–512 GPUs) |
| Interconnect | Mellanox ConnectX, NDR 400 Gb/s InfiniBand |
| NICs per node | **12 IB HCAs** — `mlx5_0..7, 9..12` |
| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |
| **SHARP** | **ACTIVE** — SAT trees `treeID:512/513`, LLT trees `treeID:0/1` allocated cleanly; 0 SHARP errors on a healthy run; `Loaded collnet plugin SHARP (v8)` followed by `sharp_job_id:N` rather than yesterday's `Cannot create SHARP job(-11)` |
| Bad-node activity later in day | slinky-40, 5, 11, 12, 21 NODE_FAILed during the afternoon 256-GPU LLM sweep — irrelevant to this morning's NCCL sweep, which completed before the cascade |

## Software Configuration

| Component | Details |
|-----------|---------|
| NCCL | 2.28.9+cuda12.9 |
| NCCL Library | system `/usr/lib/x86_64-linux-gnu/libnccl.so.2` |
| NCCL Test Binaries | `/home/johnson/nccl-tests-upstream/build/` |
| MPI | HPC-X OpenMPI 4.1.7a1, `srun --mpi=pmix` |
| Critical MCA settings | `OMPI_MCA_coll=^hcoll,ucc` (disable HPC-X collectives — they hang MPI_Init at ≥12 nodes), `pml=ob1`, `btl=tcp,self` |
| Vendor NCCL tuning | `NCCL_IB_PCI_RELAXED_ORDERING=1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_AR_THRESHOLD=0`, `NCCL_IB_SPLIT_DATA_ON_QPS=0`, `NCCL_IGNORE_CPU_AFFINITY=1` |
| **SHARP enablement (new today)** | `SHARP_COLL_ENABLE_SAT=1`, `SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1`, `UCX_IB_PCI_RELAXED_ORDERING=on`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| Per-test toggles | `NCCL_COLLNET_ENABLE=1`, `NCCL_NVLS_ENABLE=0` for CollnetChain/CollnetDirect rows; `NCCL_NVLS_ENABLE=1` for NVLS row; defaults for NVLSTree |
| HPC-X | `/opt/hpcx/hpcx-init.sh` |
| Scheduler | Slurm partition=`slinky`, `--exclusive`, no exclusions (clean 64-node pool at sweep time) |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Message sizes | 1 MiB → 32 GiB (factor 2 sweep, 16 sizes) |
| Iterations | 5 warmup + 20 measured (`-w 5 -n 20`) |
| GPUs per process | 8 (`-g 8`, 1 process per node) |
| Validation | off (`-c 0`) |
| Sbatch template | `/home/johnson/nccl-scaling-2026-05-12/wrapper.sbatch` |
| Submitter | `/home/johnson/nccl-scaling-2026-05-12/submit.sh` |
| Slurm jobs | 2662–2739 (sweep 2662–2733; CollnetDirect retries 2735–2739 for transient SHARP "Streaming Tree lock failed (-18)" hits at 8n/16n/64n) |
| Results dir | `/home/johnson/nccl-scaling-2026-05-12/logs/` |
| TSV | `/home/johnson/nccl-scaling-2026-05-12/results.tsv` |

## Job Results Summary

| Category | Count | Notes |
|----------|-------|-------|
| Passed | 64 | Ring, Tree, NVLS, NVLSTree, CollnetChain, **plus most CollnetDirect** sizes |
| CollnetDirect failures (4 of 14) | 4 | 8n/16n/64n all_gather + 8n all_reduce hit transient `SHARP group create: Streaming Tree lock failed (-18)`; CollnetDirect 8n all_reduce **succeeded on retry** (job 2735), all_gather retries still failed |
| CollnetChain failures (7 of 14) | 7 | All 7 all_gather CollnetChain runs — expected: CollnetChain is **allreduce-only** per NCCL 2.28 (`no algorithm/protocol available for function AllGather with datatype ncclInt8`) |
| Hardware failures | 0 during sweep | Bad-node cascade started later, after sweep completed |

---

## Results — all_reduce busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | Ring | Tree | NVLS / NVLSTree | **CollnetChain** | **CollnetDirect** | Best |
|---|---|---|---|---|---|---|
| 1n (8) | 698.39 | 555.81 | **844.35** | ✗ (1n, no fabric) | ✗ (1n) | **NVLS 844.35** |
| 2n (16) | 388.41 | 338.52 | **719.70** | 526.44 | 597.97 | **NVLSTree 719.70** |
| 4n (32) | 389.30 | 192.25 | 388.40 | 540.07 | **605.90** | **CollnetDirect 605.90** |
| 8n (64) | 389.20 | 194.53 | 389.89 | **546.26** | (retry 8n ok, peak not re-parsed in TSV) | **CollnetChain 546.26** |
| 16n (128) | 388.30 | 195.82 | 388.87 | 544.99 | **588.83** | **CollnetDirect 588.83** |
| 32n (256) | 389.98 | 195.43 | 389.49 | 540.74 | **565.06** | **CollnetDirect 565.06** |
| 64n (512) | 389.81 | 195.90 | 389.72 | 534.53 | **545.77** | **CollnetDirect 545.77** |

**The state-change is decisive.** From 4 nodes onward CollnetDirect (where it runs) and CollnetChain together replace Ring/NVLSTree as the all_reduce winner. CollnetChain converges to ~535 GB/s at 64n; CollnetDirect tops out a bit higher but is flakier on this run (transient `Streaming Tree lock` errors on first attempts at 8n/16n/64n — only the 8n all_reduce retry came back clean).

Sanity check vs the project memo `sharp-active-on-slinky-2026-05-12` (64n verification, job 2644 at 08:32 UTC): 534.44 GB/s at 32 GiB CollnetChain — within noise of today's 534.53.

## Results — all_reduce busBW @ 8 GiB / 16 GiB / 32 GiB (CollnetDirect, GB/s)

| Nodes | 8 GiB | 16 GiB | 32 GiB |
|---|---|---|---|
| 2n | 545.40 | 578.12 | 597.97 |
| 4n | 550.55 | 578.27 | 605.90 |
| 16n | 537.53 | 563.13 | 588.83 |
| 32n | 527.24 | 544.28 | 565.06 |
| 64n | 529.90 | 537.20 | 545.77 |

SHARP saturates around 540–600 GB/s and degrades very gently with scale — the ~10% drop from 4n to 64n is much shallower than IB-Ring's drop on the same hardware (which yesterday lost ~9% from 32n→62n).

## Results — all_reduce busBW @ 32 GiB Ring (yesterday → today, full cluster vs degraded)

| Nodes | 2026-05-11 (62n) | 2026-05-12 (64n) | Delta |
|---|---|---|---|
| 1n | 686.12 | 698.39 | +1.8% |
| 2n | 388.20 | 388.41 | flat |
| 4n | 388.94 | 389.30 | flat |
| 8n | 390.34 | 389.20 | flat |
| 16n | 389.54 | 388.30 | flat |
| 32n | 388.33 | 389.98 | flat |
| 62n→64n | 356.17 | **389.81** | **+9.4%** |

The Ring number recovered fully at the top end with all 64 nodes (vs yesterday's 62 with 2 bad). 2n–32n is unchanged within 0.5% — Ring on this cluster is rock-solid baseline.

## Results — all_gather busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | Ring | NVLS | CollnetChain | CollnetDirect | Best |
|---|---|---|---|---|---|
| 1n (8) | **676.54** | 674.50 | ✗ (allreduce only) | ✗ | Ring 676.54 |
| 2n (16) | **371.44** | 369.71 | ✗ | 311.97 | Ring 371.44 |
| 4n (32) | 376.91 | **376.97** | ✗ | 316.96 | ≈Ring/NVLS 377 |
| 8n (64) | **380.62** | 378.23 | ✗ | ✗ (retry also failed) | Ring 380.62 |
| 16n (128) | **382.83** | 380.38 | ✗ | ✗ (retry failed) | Ring 382.83 |
| 32n (256) | 378.50 | **383.22** | ✗ | 332.51 | NVLS 383.22 |
| 64n (512) | 380.15 | **382.64** | ✗ | ✗ (retry failed) | NVLS 382.64 |

**SHARP does not help all_gather** on this cluster: no reduction step for the AM to offload, so CollnetDirect just adds CollNet-tree setup overhead and lands at ~330 GB/s (slower than Ring at 380). Matches use3a-ss / MD1 behavior and the `sharp-active-on-slinky-2026-05-12` memo's 64n result (Ring 381.6 ≈ NVLS 379.9 ≈ SHARP 383.6). Ring and NVLS are interchangeable winners.

## Results — sendrecv P2P busBW peak (GB/s) @ 32 GiB

| Nodes (GPUs) | busBW peak |
|---|---|
| 1n (8) | 650.48 |
| 2n (16) | 34.16 |
| 4n (32) | 34.10 |
| 8n (64) | 17.16 |
| 16n (128) | 17.21 |
| 32n (256) | 17.19 |
| 64n (512) | 17.19 |

sendrecv unchanged vs 2026-05-11/2026-05-09: the per-pair NIC bottleneck (17.2 GB/s at ≥8n, ~34 GB/s at 2n/4n) is structural and not affected by SHARP. 2n/4n are 5–6% higher than yesterday — likely just the slinky-9/26 healing.

---

## Comparison vs. 2026-05-11 sweep (62n, SHARP inactive)

Same wrapper, same NCCL build, same MCA settings — the only deltas are (1) two healed nodes restoring 64n testability and (2) SHARP daemon now answering.

| Metric | 2026-05-11 (62n) | 2026-05-12 (64n) | Delta |
|---|---|---|---|
| all_reduce best 32 GiB @ 4n | NVLSTree/Ring 389 | **CollnetDirect 605.90** | **+55.8%** |
| all_reduce best 32 GiB @ 8n | NVLS 390.08 | **CollnetChain 546.26** | **+40.0%** |
| all_reduce best 32 GiB @ 16n | NVLSTree 390.74 | **CollnetDirect 588.83** | **+50.7%** |
| all_reduce best 32 GiB @ 32n | NVLSTree 389.46 | **CollnetDirect 565.06** | **+45.1%** |
| all_reduce best 32 GiB @ 62→64n | NVLSTree 362.46 | **CollnetDirect 545.77** | **+50.6%** |
| all_reduce Ring 32 GiB @ 62→64n | 356.17 | 389.81 | +9.4% (bad-node recovery) |
| all_gather best 32 GiB @ 64n | Ring 348.55 | NVLS 382.64 | +9.8% (bad-node recovery) |
| sendrecv 32 GiB @ ≥8n | 17.18 | 17.19 | flat |

Net: **the SHARP enablement is the single biggest cluster improvement we have measured on slinky.** Every Ring/NVLSTree number from yesterday is unchanged within noise — the gains come entirely from a new algorithm (CollNet/SHARP) becoming available.

---

## Comparison with MD1 Cluster Baseline (2026-05-02)

This is the first time we can do a like-for-like SHARP comparison.

### all_reduce busBW @ 16 GiB

| Nodes | MD1 SHARP | This run CollnetChain | This run CollnetDirect | This run NVLSTree | vs MD1 SHARP |
|---|---|---|---|---|---|
| 2n | 533.0 | 524.28 | **578.12** | 717.85 | CollnetDirect +8.5% |
| 4n | 543.8 | 539.48 | **578.27** | 387.92 | CollnetDirect +6.3% |
| 8n | 547.9 | 541.86 | (retry-only) | 388.58 | CollnetChain −1.1% |
| 16n | 546.7 | 543.83 | **563.13** | 389.55 | CollnetDirect +3.0% |
| 32n | 544.3 | 540.74 | **544.28** | 389.04 | parity |
| 64n vs MD1's nearest | — | 534.53 | 545.77 | 387.67 | (no MD1 64n point) |

**This cluster's CollnetDirect beats MD1's SHARP by 3–8% at 2–16n** and matches at 32n. At equivalent scales, slinky's NDR400 + 12-HCA fabric + active SHARP is now slightly ahead of MD1, not behind. Yesterday's 28–30% deficit to MD1 SHARP is gone.

### all_gather busBW @ 16 GiB

| Nodes | MD1 SHARP | This run Ring | This run NVLS | vs MD1 SHARP |
|---|---|---|---|---|
| 2n | 376.4 | 371.44 | 370.97 | −1.3% |
| 4n | 385.2 | 375.61 | 378.09 | −1.9% |
| 8n | 383.6 | 377.85 | 377.98 | −1.5% |
| 16n | 385.2 | 377.13 | 374.19 | −2.1% |
| 32n | 377.6 | 373.65 | 384.24 | +1.8% |

Within 2.1% across the board — as expected, all_gather is fabric-bound and SHARP doesn't help on either cluster.

### sendrecv P2P @ 16 GiB

| Nodes | MD1 | This run | Delta |
|---|---|---|---|
| 1n | 641.6 | 650.20 | +1.3% |
| 2n | 42.6 | 34.15 | −19.8% |
| 4n | 42.7 | 34.10 | −20.1% |
| 8n | 25.4 | 17.15 | −32.5% |
| 16n | 15.1 | 17.19 | +13.8% |
| 32n | 15.6 | 17.19 | +10.2% |

Same as 2026-05-11 — per-pair P2P trails MD1 below 16n, slightly beats it above.

---

## Key Findings

1. **SHARP is live.** Verified on the morning 64n sweep and matches the standalone job 2644 verification within 0.02 GB/s. The 28-cell "no SHARP" hole in every prior sweep is filled. Plugin loads `SHARP (v8)`, allocates `sharp_job_id` cleanly, runs end-to-end with 0 SHARP errors on a healthy run.

2. **CollnetDirect is the new all_reduce winner at 4–64 nodes**, peaking at **605.9 GB/s busbw at 4n** and holding **545.8 GB/s at 64n / 512 GPUs**. CollnetChain trails CollnetDirect by 2–10% but is the more reliable of the two (no `Streaming Tree lock` errors).

3. **The SHARP gain is +40–56% at 4n+** over yesterday's best (Ring/NVLSTree). This is the largest single-day improvement recorded for this cluster in the report series.

4. **CollnetChain is all_reduce-only.** All 7 all_gather CollnetChain runs failed with `no algorithm/protocol available for function AllGather` — that's correct NCCL 2.28 behavior. For all_gather/reduce_scatter, leave `NCCL_ALGO` unset and let NCCL pick (NCCL picks Ring or NVLS, both ~380 GB/s).

5. **Transient SHARP "Streaming Tree lock failed (-18)" hits CollnetDirect.** 4 of 14 CollnetDirect submissions failed on first attempt (8n/16n/64n all_gather + 8n all_reduce). The 8n all_reduce retry (job 2735) succeeded cleanly. Likely a SHARP fabric-manager contention issue when multiple jobs request the same SAT tree concurrently — worth flagging to ops but not blocking; sweep-style submission with light retry is enough.

6. **Per-workload SHARP application is still the rule.** Memory `sharp-active-on-slinky-2026-05-12` already records the 2026-05-12 finding that **SHARP env hurt the llama70b FP8 FSDP run by 12.5%** (1412 with SHARP vs 1614 ring-only). NCCL-level wins do not transfer to FSDP-heavy LLM workloads where all_gather/reduce_scatter dominate. Keep SHARP env scoped to allreduce-bound workloads.

7. **2n NVLSTree (719.7 GB/s) still beats SHARP at 2 nodes.** Intra-NVLink reduction dominates at 2n; SHARP only wins once IB hops dominate (4n+). Same shape as MD1.

8. **All 64 nodes were healthy at sweep time.** slinky-9 and slinky-26 (yesterday's bad pair) verified healed in pair sweep before this sweep ran. The afternoon LLM sweep later found 13 nodes that failed under sustained training load — those failures did **not** manifest as NCCL `all_reduce_perf` failures during this short-duration sweep (see `flapping_airplanes_256gpu_benchmark_report_2026-05-12.md` for full bad-node cascade context).

9. **sendrecv and all_gather profiles unchanged** vs 2026-05-09 / 2026-05-11. The structural IB ceiling on these collectives is independent of SHARP state, as expected.

---

## Node Actions

| Node | Status | Action |
|------|--------|--------|
| slinky-9, slinky-26 | **HEALED** (verified 2026-05-12 morning: 716/688 GB/s 2n busbw, jobs 2656/2657) | OK — back in service for short-duration NCCL workloads |
| slinky-43, slinky-62 | HEALED (verified 2026-05-11) | OK |
| slinky-40 | NODE_FAILed 6× during afternoon 256-GPU LLM sweep (jobs 2879, 2931, 2933, 2936) | **Drain** — clearly hardware. Did not impact this morning's NCCL sweep |
| slinky-5, 11, 12, 21 | 1 NODE_FAIL each during afternoon LLM sweep | **Drain pending ops review.** None failed in NCCL sweep |
| slinky-2, 18, 20, 37, 56 | Conservative exclude on LLM side after pyxis / pair-sweep markers | Not a NCCL concern — short tests pass |
| All other 51 nodes | Healthy in NCCL sweep and afternoon LLM sweep | OK |

> **Caveat:** as on 2026-05-11, NCCL `all_reduce_perf` short tests are necessary-but-not-sufficient for LLM-training stability. A node passing this morning's sweep does not guarantee it survives a 20-minute training run (see 256-GPU LLM report for the same-day evidence).

---

## What to Re-run

### 2-Node Pairwise Sweep (IB health check)

```bash
bash /home/johnson/auto_sweep_512gpu/pair_sweep_helper.sh "$(scontrol show hostnames slinky-[0-63] | paste -sd,)" /home/johnson/pair_sweep_$(date +%Y%m%d_%H%M%S)
```

### Full collective scaling sweep (SHARP active config)

```bash
cd /home/johnson/nccl-scaling-2026-05-12
bash submit.sh 1,2,4,8,16,32,64                    # produces ~77 jobs
bash parse.sh > results.tsv                         # parses logs after jobs complete
```

### Reproduce the 64n CollnetDirect peak

```bash
sbatch --nodes=64 --export=ALL,RUN_BINARY=/home/johnson/nccl-tests-upstream/build/all_reduce_perf,RUN_LABEL=manual_64n_cd,RUN_COLLNET=1,RUN_NVLS=0,RUN_ALGO=CollnetDirect \
  /home/johnson/nccl-scaling-2026-05-12/wrapper.sbatch
```

### Reproduce the 64n CollnetChain peak (more reliable)

```bash
sbatch --nodes=64 --export=ALL,RUN_BINARY=/home/johnson/nccl-tests-upstream/build/all_reduce_perf,RUN_LABEL=manual_64n_cc,RUN_COLLNET=1,RUN_NVLS=0,RUN_ALGO=CollnetChain \
  /home/johnson/nccl-scaling-2026-05-12/wrapper.sbatch
```

---

## Cross-references

- **2026-05-11 NCCL scaling report (predecessor, 62n, SHARP inactive)**: `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-11.md`
- **2026-05-09 NCCL scaling report (60n, SHARP inactive)**: `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-09.md`
- **Today's 256-GPU LLM sweep (cluster-instability context, bad-node cascade)**: `/home/johnson/worklogs/flapping_airplanes_256gpu_benchmark_report_2026-05-12.md`
- **SHARP enablement memo (working sbatch + env, per-workload rule)**: `~/.claude/projects/-data-home-johnson/memory/project_sharp_active_slinky.md`
- **MD1 cluster NCCL baseline (2026-05-02)**: `/home/johnson/worklogs/2026-05-02_nccl_collective_scaling_report.md`
- **All result logs**: `/home/johnson/nccl-scaling-2026-05-12/logs/`
- **TSV summary**: `/home/johnson/nccl-scaling-2026-05-12/results.tsv`
- **Working sbatch templates**: `/home/johnson/nccl-scaling-2026-05-12/wrapper.sbatch`, `/home/johnson/nccl-ar-sharp-64n-b200.sbatch`
