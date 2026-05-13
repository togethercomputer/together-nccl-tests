---
name: 2026-05-11 512-GPU LLM sweep on slinky
description: First full-cluster 512-GPU LLM benchmark sweep on slinky; orchestrator design, scaling efficiency results, fallback behavior, newly-bad nodes discovered mid-sweep
type: project
originSessionId: d35c32fd-26ac-4a1c-bb0d-188840f5274b
---
## Headline results (512 GPU / 64 nodes, MAX_STEPS=10, mean iter 3-9)

| Workload | 512 GPU | vs Yesterday @256 (slinky) | vs MD1 @256 |
|---|---:|---:|---:|
| Llama 70B FP8 | 1614.1 | 88.2% scaling | **+7.4%** |
| Llama 70B NVFP4 | 1946.5 | 94.3% scaling | −4.7% |
| Nemotron-H 56B FP8 | 1424.6 (fallback @256) | +27.5% same scale | −6.7% |
| Llama 405B FP8 | 931.4 | 91.1% scaling | −47% (SHARP-bound) |
| Llama 405B NVFP4 | 1785.0 (fallback @256, partial) | **first success ever** | −9.7% |
| Nemotron-4 340B FP8 | 1056.4 | 77% scaling vs same-day 256 (1371) | −15% |
| Nemotron-4 340B BF16 | 789.4 | 92.5% scaling | −9.1% |
| Qwen3 235B BF16 | 490.6 | **+2.4% superlinear** (EP=8 MoE) | −20% |

## Orchestrator location

- Script: `/home/johnson/auto_sweep_512gpu/orchestrator.sh`
- Pair-sweep helper: `/home/johnson/auto_sweep_512gpu/pair_sweep_helper.sh`
- Results: `/home/johnson/auto_sweep_512gpu/results.tsv`
- Logs: `/home/johnson/auto_sweep_512gpu/logs/`
- Report: `/home/johnson/worklogs/flapping_airplanes_512gpu_benchmark_report.md`

## Orchestrator design (key learnings)

1. **Preflight at workload scale, not fixed 32n.** 64-node NCCL all_reduce 8 GiB before each LLM submit (~30 s). Sufficient to gate against severe IB issues.
2. **Fallback fires only when mean iter 3-9 cannot be parsed.** Initial buggy version (`state in TIMEOUT,FAILED,NODE_FAIL,OOM OR mean empty`) caused a spurious n4340b_fp8 retry — Slurm TIMEOUTed after the model completed all 10 iters in post-training teardown stall.
3. **Pair sweep across the failed job's nodelist.** Submits 32 disjoint 2-node pairs in parallel; flags pairs without `Out of bounds values : 0 OK`. Caught slinky-9 + slinky-26 mid-sweep when they degraded.
4. **TCPStore bootstrap deadlocks are NOT IB-detectable.** Pair sweep correctly reported "no bad nodes" when nemotronh_fp8 hung at 512-rank UniqueNCCLID exchange — the failure was in the TCP control plane, not IB data plane. Fallback to 256 GPU still worked because 256-rank TCPStore doesn't deadlock.

## Failure modes seen at 512 GPU scale

- **PyTorch TCPStore UniqueNCCLID timeout**: nemotronh_fp8. 10-min wait, rank-0 host slinky-9. Symptom of control-plane flake, not necessarily IB.
- **NODE_FAIL with no usable iters**: llama405b_nvfp4. Real hardware failure mid-training. Pair sweep correctly identified the bad nodes (slinky-9, slinky-26).
- **Post-training Slurm TIMEOUT with valid iter data**: n4340b_fp8 (both 512 and 256-fallback). Training completes all 10 iters but post-training teardown stalls; Slurm hits wall time. Mean iter 3-9 still parseable from log. Treat as success.
- **Pyxis container child failure mid-job**: third node failure during llama405b_nvfp4 fallback. K8s pod cycling — not consistently reproducible.

## Cluster health is a moving target

- Pre-sweep verification 09:20 UTC: all 64 nodes pass (including previously-broken slinky-43, slinky-62). Cluster 388.5 GB/s on 64n ring.
- Mid-sweep 09:54 UTC: slinky-9 fails TCPStore as rank-0.
- End-of-sweep 12:02 UTC: pair sweep flags slinky-9, slinky-26 as IB-broken.
- 12:24 UTC: unidentified third node hits pyxis failures.

→ **Run a fresh pair sweep before any large run, not just yesterday's whitelist.**

## How to apply

When re-running at any large scale on slinky:

1. Verify cluster: 64n ring NCCL allreduce, expect 380-390 GB/s.
2. Use `auto_sweep_512gpu/orchestrator.sh` template; adjust workload list as needed.
3. Don't use any pre-baked whitelist — health drifts within hours.
4. Expect 1-2 fallbacks per 8-workload sweep on average.
5. Pyxis/pod-cycling failures still happen at 512 rank even with healthy IB — outside our control.

## SHARP still inactive

4n CollNet test (job 2412, 09:31 UTC) produced 390 GB/s = ring fallback. SHARP is the single biggest improvement available for Llama 405B FP8 (would close the −47% gap vs MD1). Admin action required on IB fabric manager.
