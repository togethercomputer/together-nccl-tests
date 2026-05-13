---
name: 2026-05-12-256gpu-sweep-recovery
description: 2026-05-12 256-GPU LLM sweep on slinky after the morning SHARP-attempt aborted. 7/8 valid workloads obtained on a cluster that exposed 13 bad nodes (20% of fleet). N4 340b workloads improved +24% / +8% unexplained. nemotronh_fp8 abandoned (TCPStore deadlock). Introduces auto-exclude-on-NODE_FAIL retry pattern.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0b1f8edb-a830-4ce7-b9fa-49de3f8c9028
---

## Headline

- 7 of 8 workloads completed with valid data despite a degraded cluster (13 of 64 nodes failed today during real LLM training)
- **N4 340B FP8 = 1363.4 (+24% vs 2026-05-10's 1096, +29% vs 2026-05-11's 1056) — unexplained jump worth verifying**
- **N4 340B BF16 = 916.7 (+8% / +16%)** — also up
- **Llama 405B NVFP4 = 1667.0** — **first clean 256-GPU result on this cluster** (2026-05-10 NODE_FAILed, 2026-05-11 only partial fallback)
- Llama 405B FP8 = 1043.6 (+2% vs 2026-05-10)
- Nemotronh_fp8 abandoned (5 attempts, 3 NODE_FAILs + 1 TCPStore deadlock)

## Full report

`/home/johnson/worklogs/flapping_airplanes_256gpu_benchmark_report_2026-05-12.md` (136 lines)

## Key lesson 1: NCCL pass ≠ training stability on slinky

This was the dominant pattern of the day:
- 10:55 — Full pair-sweep across 62 nodes: **31 of 31 pairs PASS at ~715 GB/s** (B200 baseline)
- 13:51 — 64-node NCCL all_reduce 8 GiB job 2925: **COMPLETED 0:0, 0 errors, 325 GB/s ring busbw**
- Yet 13 of those same nodes hard-NODE_FAILed within 2-5 min of LLM training start

**How to apply:** Short NCCL collective tests are necessary-but-not-sufficient. Trust them only for "is this node hardware-dead?" — not for "will this node hold up under 30 min of TP+PP+DP+sustained-allreduce training load?". Plan for active node attrition during multi-hour LLM sweeps regardless of pre-sweep verification.

## Key lesson 2: Auto-exclude-on-NODE_FAIL retry pattern

Designed and proved valuable today. Implemented in `auto_sweep_256gpu_retry_v3_2026-05-12.sh`:

```bash
# Base + runtime-discovered excludes
BASE_EXCLUDE="slinky-X,Y,Z..."
RUNTIME_EXCLUDE=""
for attempt in 1 2 3; do
    submit_llmb_with_exclude $(merged_exclude)
    wait_and_parse
    [ -n "$MEAN" ] && break  # success
    [ "$STATE" = "TIMEOUT" ] && break  # treat TIMEOUT-with-mean as success
    BAD=$(get_failing_node $jid $ws)   # parse "ON slinky-N CANCELLED" from log
    [ -z "$BAD" ] && break              # can't identify, give up
    RUNTIME_EXCLUDE="${RUNTIME_EXCLUDE:+$RUNTIME_EXCLUDE,}$BAD"
    sleep 45  # ride out Slurm reject window
done
```

**Why:** Whitelist-substitution alone is whack-a-mole (each substitution exposes a new flaky node). Auto-grow exclude lets the orchestrator learn during the run with no human in the loop.

**How to apply:** Reuse this pattern in any future sweep on a flaky cluster. Combine with `--exclude=` (not `--nodelist=`) so Slurm picks from the residual pool — gives flexibility as the bad-node set grows. Critical settings:
- 5×30s preflight retries (Slurm reject windows after a NODE_FAIL last 45-90s — 3×15s isn't enough)
- Both Megatron-Bridge `MODEL_TFLOP/s/GPU` AND NeMo `TFLOPS_per_GPU:` parser patterns (different workloads use different libraries)
- Treat NODE_FAIL-with-valid-mean as success (teardown failures shouldn't invalidate training data)

## Key lesson 3: Bash `set -u` + multi-var `local` is dangerous

```bash
# This crashes under set -u: "now: unbound variable"
local now=$(date +%s) elapsed=$((now - job_start))
```

Bash evaluates locals' RHS in unspecified order. `now` may not be bound when `$((now - ...))` evaluates. Split:

```bash
local now=$(date +%s)
local elapsed=$((now - job_start))
```

(Already noted in [[2026-05-12-512gpu-sweep-aborted]] but worth repeating.)

## Failure pattern — nemotronh_fp8

5 attempts across morning + retry v1 + retry v3 (3 attempts in v3 alone). All failed: 3 hard NODE_FAILs on 3 different nodes (slinky-12, 21, 40) + 1 TCPStore deadlock (manual scancel after 11 min log silence).

**Hypothesis:** intrinsic fragility, not pure cluster-luck. 2026-05-11 also had TCPStore deadlock on this workload. NeMo's distributed init for nemotron-h may have a single-rank-0-hostname dependency with no retry — when rank-0 host is even slightly flaky, the entire job hangs.

**How to apply:** Plan for nemotronh_fp8 to need multiple attempts on slinky. Consider larger N4-style retry budget OR investigate NeMo init configuration before relying on it for time-critical sweeps.

## Unexplained finding — N4 340b workloads

Both N4 340b workloads (FP8 and BF16) jumped significantly today:

| | 2026-05-10 (256 GPU) | 2026-05-11 (512 GPU) | 2026-05-12 (256 GPU, today) |
|---|---:|---:|---:|
| N4 340B FP8 | 1096.4 | 1056.4 | **1363.4** |
| N4 340B BF16 | 853.1 | 789.4 | **916.7** |

Today's same-day sweep was on a 32-node whitelist with 9 bad nodes excluded — different node mix than prior days. Could be cluster topology luck OR llmb-run installer drift (no obvious version changes in `/data/home/johnson/llmb/`). **Worth verifying with a clean-cluster repeat** to confirm the gain is reproducible vs node-set luck.

**How to apply:** When citing the N4 340b numbers in further reports, flag the +24% as "unverified — repeat needed to rule out node-set luck". MD1 baseline (1245 / 868) sits between today's and prior days' numbers — today's is significantly above MD1, suggesting genuine cluster improvement OR very lucky node selection.

## Today's bad-node tally (13 of 64 = 20%)

- **slinky-40**: 6 NODE_FAILs in one day. Almost certainly hardware-faulty. Worst single offender. Drain/reboot recommended.
- **slinky-9, slinky-26, slinky-14**: 1-2 NODE_FAILs each in morning. slinky-9 and slinky-14 also alternately hosted user `theoh`'s single-node MoE training (no failures there). Suggests **multi-node IB-only fault**, not GPU/compute.
- **slinky-5, slinky-11, slinky-12, slinky-21**: 1 NODE_FAIL each, all under LLM load. Newly-discovered today via auto-exclude.
- **slinky-18**: pyxis container failure (different mode from NODE_FAIL).
- **slinky-2, slinky-37, slinky-56**: pair-sweep-flagged in morning but never reproduced under LLM load — conservatively excluded.
- **slinky-20**: rank-0 of CANCELLED job (manual scancel) — added defensively.

**How to apply:** For next sweep, pre-exclude at least the 5 hard-failed nodes (slinky-5, 11, 12, 21, 40). Re-add only after ops confirms stability under sustained training.

## Cross-references

- Morning SHARP attempt: [[2026-05-12-512gpu-sweep-aborted]]
- Cluster environment: [[slinky-cluster-environment]]
- 2026-05-11 sweep: [[2026-05-11-512-GPU-LLM-sweep]]
- Full report: `~/worklogs/flapping_airplanes_256gpu_benchmark_report_2026-05-12.md`
- Orchestrator (final): `~/auto_sweep_256gpu_retry_v3_2026-05-12.sh`
- Results TSV: `~/auto_sweep_results_2026-05-12.tsv`
