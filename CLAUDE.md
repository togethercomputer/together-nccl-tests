# Claude Code context for `together-nccl-tests`

This repo is the canonical home for Together's GPU cluster benchmarking workflow with Claude Code. It hosts skills, memory snapshots, and conventions used across:

- `together-nccl-tests` (this repo) — NCCL benchmarks, `find_stragglers.py`, sbatch templates, per-cluster baselines
- `together-dgxc-benchmarking` — LLM training workloads, sweep results, worklog

## What's in `.claude/`

| Path | Purpose |
|---|---|
| `.claude/skills/cluster-bringup/` | Procedure for onboarding a new GPU cluster (discover → install → validate → sweep) |
| `.claude/skills/benchmark-report/` | Scaffold a benchmark report in the right location with date prefix |
| `.claude/memory/` | Reference memory: per-cluster state, debug ladders, conventions |
| `.claude/install.sh` | Symlinks the above into your personal Claude Code paths |

## First-time setup (per machine)

```bash
cd ~/together-nccl-tests
./.claude/install.sh /data/home/$USER   # use your primary workdir
```

Then restart Claude Code. Skills become invokable as `/cluster-bringup` and `/benchmark-report`; memory entries auto-load across future sessions.

## Filing reports (conventions)

| Topic | Destination | Filename |
|---|---|---|
| LLM training benchmark | `together-dgxc-benchmarking/worklog/` | `YYYY-MM-DD_<descriptor>.md` |
| NCCL collective scaling | `together-nccl-tests/baselines/<cluster>/` | `YYYY-MM-DD_nccl_collective_scaling_report.md` |
| Straggler diagnostic | `together-nccl-tests/baselines/<cluster>/` | `YYYY-MM-DD_straggler_report.md` |
| Profiler / nsys analysis | `together-dgxc-benchmarking/worklog/profiler_reports/` | `<model>_<dtype>_<scope>.md` |

Run artifacts (`.log`, `.out`, `.sbatch`, `.tsv`) stay in `~/` and are not committed.

## Branch convention

Personal development on `johnson-dev` / `johnson_dev`. Open PRs to `main` for shared work.
