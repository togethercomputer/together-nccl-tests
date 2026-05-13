---
name: 2026-05-12-512gpu-sweep-aborted
description: "SHARP-enabled 512-GPU LLM sweep on 2026-05-12 — aborted at 10:28 after 6 NODE_FAILs across 4 workload attempts, cluster severely degraded. 1 usable result (llama70b_fp8 = 1412.2 TFLOPS/GPU, -12.5% vs 2026-05-11). Key learning: SHARP env hurts FSDP-heavy workloads; apply per-workload not globally."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0b1f8edb-a830-4ce7-b9fa-49de3f8c9028
---

## What we tried

Mirror of 2026-05-11 sweep ([[2026-05-11-512-GPU-LLM-sweep]]) with one change: SHARP env enabled in submit_llmb via 6 exports (NCCL_COLLNET_ENABLE=1, NCCL_NVLS_ENABLE=1, SHARP_COLL_ENABLE_SAT=1, SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1, UCX_IB_PCI_RELAXED_ORDERING=on, CUDA_DEVICE_ORDER=PCI_BUS_ID). NCCL_ALGO unset.

Orchestrator: `/home/johnson/auto_sweep_512gpu/orchestrator_sharp.sh`
Results TSV: `/home/johnson/auto_sweep_512gpu/results_sharp.tsv`

## What actually happened

09:38 launch → first workload (llama70b_fp8) submitted → orchestrator crashed @ 09:38:52 on **unbound-variable bug** in my watchdog code (`local now=$(date +%s) elapsed=$((now - job_start))` — bash evaluates locals in unspecified order under `set -u`). Job 2740 ran orphaned to completion (1412.2 TFLOPS/GPU). Bug fixed by splitting to two `local` lines. Relaunched 09:50.

09:54 onward: **progressive cluster IB degradation**. 6 NODE_FAILs across 4 workload attempts:

| Time | Job | Workload | Scale | Failure | Bad node |
|---|---|---|---|---|---|
| 09:54 | 2742 | llama70b_nvfp4 | 512 | NODE_FAIL | slinky-9 |
| 09:59 | 2774 | llama70b_nvfp4 (fallback) | 256 | NODE_FAIL | slinky-26 |
| 10:07 | 2776 | nemotronh_fp8 | 512 | NODE_FAIL | slinky-9 |
| 10:21 | 2810 | llama70b_nvfp4 (retry) | 256 | NODE_FAIL | slinky-14 |
| 10:26 | 2827 | llama70b_nvfp4 (retry+pairs excluded) | 256 | NODE_FAIL | (unknown, 32-node) |
| 10:27 | 2829 | nemotronh_fp8 | 256 | NODE_FAIL **pyxis** | slinky-18 |

After 3rd consecutive failure I aborted (10:28). Killed orchestrator + pair-sweep jobs.

## Bad nodes flagged on 2026-05-12 (≥10 of 64 = 15.6%)

- Hard NODE_FAILs: slinky-9 (×2), slinky-14, slinky-18 (pyxis), slinky-26
- Pair-sweep flagged: slinky-2, 11, 13, 28, 37, 56

**Diagnostic:** slinky-9 was simultaneously running user `theoh`'s single-node MoE job 2843 successfully at 10:30. → Multi-node IB issue, not single-node GPU/compute.

## Only useful result obtained

| Workload | Scale | SHARP | TFLOPS/GPU | vs 2026-05-11 (no SHARP) |
|---|---|---|---|---|
| llama70b_fp8 | 512 | yes | **1412.2** | **-12.5%** vs 1614.1 |

llama70b is FSDP-heavy (DP=64, TP=2, PP=4). SHARP only offloads allreduce; FSDP collectives (all_gather, reduce_scatter) gain nothing and pay CollNet setup overhead. See [[sharp-active-on-slinky-2026-05-12]] "SHARP env HURTS FSDP" section.

## Lessons

**1. SHARP env is workload-specific, not globally beneficial.**
- ✅ Apply for allreduce-heavy: TP+PP+CP layouts with small DP (Llama 405B at TP=4 PP=8 CP=2 DP=2)
- ❌ Skip for FSDP-heavy: small TP×PP×CP, big DP (Llama 70B at any dtype, MoE)
- Default in `orchestrator_sharp.sh` was global SHARP — wrong call; should have been per-workload.

**2. Cluster IB stability on slinky is short-lived (<2 hours).** 09:00 fully healthy → 10:30 ≥10 bad nodes. The morning's verification (job 2644 SHARP, jobs 2656/2657 healed) doesn't mean the cluster will hold for a 3-hour sweep. Yesterday saw the same pattern at smaller scale.

**3. Bash `set -u` + multi-var `local` is a trap.** `local a=$(...) b=$((a + 1))` — `b`'s RHS evaluates `a` before `a` is bound, hits unbound-variable. Always split.

**4. Re-allocate-on-failure isn't sufficient when faults cascade.** Yesterday's orchestrator design assumed isolated bad nodes; today's cluster has wide IB degradation that grows during the sweep. A "global exclude" / "skip-to-256" mode is needed when faults are wider than 1-2 nodes. Orchestrator was patched mid-sweep to add this (commit log in `orchestrator_sharp.sh` comments).

## How to apply

Before next attempted SHARP sweep on slinky:
1. Wait for ops to investigate the IB fabric — slinky-9/14/26 in particular keep failing.
2. Switch to **per-workload SHARP env** — only enable for 405B FP8 and similar allreduce-heavy ones.
3. Add a sanity 4n SHARP verification right before each LLM submit (the morning 64n verification doesn't cover stability mid-day).
4. Consider running shorter individual workloads (MAX_STEPS=5) so each job finishes inside the cluster's short stable window.

Files preserved for forensics: `~/auto_sweep_512gpu/results_sharp.tsv`, `~/auto_sweep_512gpu/logs_sharp/` (orchestrator + 4 pair-sweep dirs), `~/auto_sweep_512gpu/orchestrator_sharp.sh` (now in 256-only emergency mode; restore from `orchestrator.sh` for next sweep).
