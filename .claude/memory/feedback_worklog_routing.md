---
name: feedback-worklog-routing
description: "Where to file new benchmark worklogs/reports — by topic and cluster, with YYYY-MM-DD_ filename prefix"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 40ae4eee-9e1c-40ba-92a1-c2fca3af82b8
---

When writing or filing a new benchmark report / worklog, route by **topic**, not by which session generated it. Use `YYYY-MM-DD_` filename prefix to match existing convention.

| Topic | Destination |
|---|---|
| LLM training benchmarks (llama, deepseek, nemotron, qwen, grok, gpt-oss, dgxc-benchmarking sweeps) | `~/together-dgxc-benchmarking/worklog/YYYY-MM-DD_<name>.md` |
| NCCL collective scaling / NCCL test reports | `~/together-nccl-tests/baselines/<cluster>/YYYY-MM-DD_<name>.md` |
| Straggler diagnostic reports | `~/together-nccl-tests/baselines/<cluster>/` (alongside `healthy_*.json`) |
| Profiler / nsys / PyTorch trace analyses | `~/together-dgxc-benchmarking/worklog/profiler_reports/<model>_<name>.md` |

Cluster directory names: `use3a-ss`, `flapping-airplanes`.

When adding to `together-dgxc-benchmarking/worklog/`, also update the Reports table in `worklog/README.md` (it is hand-maintained, not auto-generated).

**Why:** Confirmed during 2026-05-13 cleanup. User had ~7 mixed-topic worklogs in `~/worklogs/` and explicitly chose: dgxc training reports → dgxc repo's `worklog/`, NCCL reports → nccl-tests' `baselines/<cluster>/` (matching the precedent set by `baselines/use3a-ss/straggler-report-20260429.md`). User also chose `YYYY-MM-DD_` rename for un-prefixed files.

**How to apply:** Skip `~/worklogs/` and `~/worklog.md` as a destination. Write reports directly into the right repo. Run/result artifacts (.log, .out, .sbatch, .tsv) and sweep scripts stay in `~` — do not check those in. See also [[project_dgxc_benchmarking]] and [[project_nccl_benchmark_scripts]].
