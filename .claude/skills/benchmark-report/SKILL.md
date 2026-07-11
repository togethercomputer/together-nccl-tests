---
name: benchmark-report
description: Scaffold a new benchmark report (LLM training sweep, NCCL collective scaling, or straggler diagnostic) in the right repo with the right filename. Use when the user has just finished a run and wants to write up results, or asks where a report should go, or wants a fresh report template filled in for today's date and cluster.
---

# Benchmark report scaffolder

Creates a new report file in the correct location with a starter skeleton, following [[feedback-worklog-routing]].

## Step 1 — Classify the report

Ask (or infer from context):

| Topic | Destination |
|---|---|
| LLM training sweep (any dgxc workload) | `~/together-dgxc-benchmarking/worklog/` |
| NCCL collective scaling / NCCL test results | `~/together-nccl-tests/baselines/<cluster>/` |
| Straggler diagnostic | `~/together-nccl-tests/baselines/<cluster>/` |
| Profiler trace analysis (nsys / pytorch profiler) | `~/together-dgxc-benchmarking/worklog/profiler_reports/` |

## Step 2 — Pick the filename

Format: `YYYY-MM-DD_<short_descriptor>.md`. Examples:

- LLM training: `2026-05-12_flapping_airplanes_256gpu_benchmark_report.md`
- NCCL: `2026-05-12_nccl_collective_scaling_report.md`
- Straggler: `2026-05-13_straggler_report.md`
- Profiler: `profiler_reports/<model>_<dtype>_<scope>.md` (no date prefix in this subdir — matches existing convention)

Use today's date (`date +%Y-%m-%d`) unless the user specifies a different one.

## Step 3 — Reuse templates where they exist

- `~/together-dgxc-benchmarking/worklog/template_256gpu_benchmark_report.md` is the canonical template for LLM training sweeps. Copy and fill in `[N]`, `[DATE]`, `[CLUSTER_NAME]`, `[CLUSTER_ID]` placeholders.
- NCCL reports follow the pattern of `baselines/<cluster>/YYYY-MM-DD_nccl_collective_scaling_report.md` — header sections: System Configuration, Run Configuration, Per-Scale Results, Comparison to Baseline, Issues, Next Steps.
- Straggler reports follow `baselines/use3a-ss/straggler-report-20260429.md`.

## Step 4 — Write the skeleton

Fill these sections at minimum (leave `TODO` markers for what the user has to add):

```markdown
# <Report title> — YYYY-MM-DD

<cluster> B200 cluster (<cluster-id>). <One-line scope: workloads, scale, context>.

## Summary

TODO: 1-2 sentence outcome.

## System Configuration

- Cluster: <name> (<N> nodes × <M> GPUs)
- Software: NCCL <version>, HPCX <version>, NeMo container <tag>
- Job IDs: <list>

## Run Configuration

TODO

## Results

TODO scoreboard table

## Issues

TODO

## Next Steps

TODO
```

## Step 5 — After writing

- If filing in `together-dgxc-benchmarking/worklog/`, also update `worklog/README.md`'s Reports table.
- Don't auto-commit — let the user review first.

## What NOT to do

- Don't write reports to `~/worklogs/` or `~/worklog.md` — those are deprecated as destinations (see [[feedback-worklog-routing]]).
- Don't include raw .log / .out / .sbatch contents in the report — link to paths instead. Run artifacts stay in `~`.
- Don't auto-generate a "Conclusion" / "Recommendations" section without the user's actual conclusion.
