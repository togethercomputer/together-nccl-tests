# NCCL Status Update — 2026-05-11

## Morning verification (09:20 UTC, all 64 nodes)

After admin fix attempt on the previously-broken pair, ran cluster-wide health check:

| Test | Result | Verdict |
|---|---|---|
| 2n: slinky-0 + slinky-43 (was IB-broken) | 716 GB/s @ 8 GiB | **HEALED** |
| 2n: slinky-1 + slinky-62 (was IB-broken) | 714 GB/s @ 8 GiB | **HEALED** |
| 4n SHARP test (COLLNET_ENABLE=1) | 390 GB/s = ring fallback | Still **inactive** |
| **64-node ring all_reduce @ 32 GiB** | **388.5 GB/s** | Best result on this cluster — beats prior 60n=367 |

All 64 nodes online and contiguous (`slinky-[0-63]`, no gaps). Cluster grew from 51 nodes since 2026-05-09.

## Afternoon scaling sweep (13:25 UTC, 62 nodes)

Full collective scaling, 77 jobs, all algorithms × 7 node sizes. SHARP attempt repeated at every scale.

| Scale | Ring all_reduce 32 GiB | NVLS/NVLSTree |
|---|---:|---:|
| 1n | 686 | **845** |
| 2n | 388 | **723** |
| 4n | 389 | 389 |
| 8n | 390 | 390 |
| 16n | 390 | 391 |
| 32n | 389 | 389 |
| 62n | 356 | **362** |

**Ring holds 386–390 GB/s flat from 2n → 32n** (≤1% spread). Beats MD1's Ring by **+12 to +28%** at every multi-node size. NVLSTree dominates at 2n (1.86× Ring); converges with Ring at 4n+.

## Current cluster state (as of 13:30 UTC)

- **62 healthy nodes** confirmed (every Ring/Tree/NVLS job COMPLETED)
- **slinky-9, slinky-26**: degraded mid-day during the 512-GPU LLM sweep (between ~10:00 and 12:02). Drain candidates.
- **slinky-43, slinky-62**: healed (back in service)
- **SHARP**: still inactive at the service layer. Both `CollnetChain` and `CollnetDirect` rejected at every scale. Plugin loads but `Cannot create SHARP job(-11)` underneath. Admin action on IB fabric manager required.

## Bottom line

| Question | Answer |
|---|---|
| Can we hit MD1 Ring numbers? | Yes — exceed by 12–28% |
| Can we hit MD1 SHARP numbers? | No — capped at ~390 GB/s busbw vs MD1 SHARP ~545. **−28% ceiling** until SHARP is enabled. |
| Are nodes stable? | Within an hour, yes. Across half a day, no — bad nodes rotate. Run a fresh pair sweep before any large run. |
| Best workload-level number today | Llama 70B FP8 @ 512 GPU = **1614 TFLOPS/GPU** (+7.4% vs MD1 256-GPU baseline). |

## Files

- Full report (this sweep): `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-11.md`
- Prior sweep (60n): `/home/johnson/worklogs/flapping_airplanes_nccl_collective_scaling_report_2026-05-09.md`
- 512-GPU LLM sweep: `/home/johnson/worklogs/flapping_airplanes_512gpu_benchmark_report.md`
