# find_stragglers.py

A single-script cluster-health diagnostic for NCCL-based GPU clusters. Finds bad nodes, classifies failure modes by layer, and produces an actionable bad-node list for Slurm exclude.

```bash
~/together-nccl-tests/stragglers/find_stragglers.py
```

That's it. No args needed. ~12-15 min wall on a 64-node B200 cluster. Outputs a structured JSON plus a Round-by-Round console report.

## What it detects

| Failure mode | Caught by | Detection layer |
|---|---|---|
| Missing shared library on a node (libmpi.so, libcudart.so) | Phase 1 bootstrap-fail classifier | software / runtime |
| Wedged Slurm/pmix (idle-but-broken slurmstepd) | Phase 1 bootstrap-fail classifier | Slurm/MPI |
| Group running below cluster median (4n-slow nodes) | Phase 1 z-score + Phase 3 localization | NCCL protocol/topology |
| Marginal candidates (run-to-run variability) | Phase 5 confirmation retest | NCCL — disambiguates topology vs node-bound |
| Inter-link / spine degradation | Phase 4 cross-pair (healthy-pair 8n test) | fabric topology |
| HCA port physically/logically down | Phase 6 IB sweep (`--with-ib-sweep`) | IB hardware |
| HCA running at degraded bandwidth | Phase 6 with K=2 — node-bound vs link-bound | IB hardware |

## Quick start

```bash
# Default — Round 1-4 (Phase 1+2+3+4+5), no IB sweep, ~12 min
~/together-nccl-tests/stragglers/find_stragglers.py

# Full diagnostic including IB layer (~15-17 min)
~/together-nccl-tests/stragglers/find_stragglers.py --with-ib-sweep

# IB-only fast hardware probe (~5 min)
~/together-nccl-tests/stragglers/find_stragglers.py \
    --skip-localize --skip-cross-pair --skip-confirm --with-ib-sweep

# Dry-run plan (no jobs submitted)
~/together-nccl-tests/stragglers/find_stragglers.py --dry-run

# Smoke test on a tiny subset (debug)
~/together-nccl-tests/stragglers/find_stragglers.py --max-nodes 8

# Force baseline refresh (after a known cluster change)
~/together-nccl-tests/stragglers/find_stragglers.py --rebaseline
```

## Output

Console: a sequence of "Round N results" blocks. JSON: full structured results at
`/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers/<timestamp>/results.json`.

The end of every run prints:

- **Stragglers identified — Round 1**: bootstrap-failed nodes (missing lib, pmix wedge) with classified evidence
- **Performance stragglers — Round 1**: groups labeled `suspect` or `severe` against in-run baseline
- **Round 2 results — Localization**: per-candidate test verdicts (`candidate_bad` / `candidate_healthy`)
- **Localized bad nodes — Round 2**: nodes confirmed bad by parallel-encoding
- **Round 3 results — Cross-pair (8n)**: inter-link suspects (degraded spine/uplink between healthy groups)
- **Round 4 results — Confirmation retest**: marginal candidates re-tested with new H-triple
- **Final node verdicts**: `CONFIRMED_BAD` / `TOPOLOGY_SENSITIVE` / mixed
- **Round 5 results — IB sweep** (with `--with-ib-sweep`): per-HCA cluster stats, bad HCAs with classification

## How it works

The tool runs sequential **Rounds**, each composed of one or more **Phases**. A Round defines a synchronization point — all jobs in a Round complete before the next Round starts.

### Round 1: Phase 1+2 — sweep & classify

1. Discover idle nodes via `sinfo -N -p batch -h -o "%N %t"`, filter on exact `STATE==idle` (`sinfo -t idle` lumps drained nodes in; we don't want that).
2. Disjoint 4-node partition: `[(h0,h1,h2,h3), (h4,h5,h6,h7), ...]`.
3. For each (group, algo) where `algo ∈ {ring, collnet_sharp}`, submit one sbatch running `all_reduce_perf` on the 4-node nodelist. Every group × algo = parallel job.
4. **Watchdog** polls each output file every 60s for the `Avg bus bandwidth` line. As soon as it appears, `scancel`s the job (NCCL has a known post-srun-hang where the job sits idle holding the allocation after writing data — scancel-on-data avoids it).
5. Per-job timeout (default 180s without "Avg bus bandwidth" line) → flag as **bootstrap-fail** and classify the failure (`missing_lib` / `pmix_wedge` / `sharp_no_data` / `unknown_hang`) by regex-matching the log.
6. Compute trimmed-quartile cluster median + stddev per algo from all completed groups; persist to `~/together-nccl-tests/baselines/<cluster>/healthy_4n.json` if absent (or `--rebaseline`).
7. z-score classify each group: `healthy` (|z|<2), `suspect` (-5<z<-2), `severe` (z<-5). Stddev floor of 0.5% of median to keep sub-percent jitter from hitting suspect threshold.

### Round 2: Phase 3 — localization

For each `severe` group, identify which of its 4 candidates is the actual bad node:

1. Build healthy H-pool from groups labelled `healthy` in Round 1.
2. For each severe group × each algo (where the group's algo score is suspect/severe):
   - For each of the 4 candidates: pair `[candidate + 3 disjoint healthy H-nodes]` and run that algo.
   - Within an algo wave, all candidates' H-triples are mutually disjoint so the 4-8 tests run truly in parallel.
3. Verdict per (candidate, algo): if the test bw is below z=-2, the candidate is `candidate_bad` for that algo.
4. A node is "localized bad" if any algo flags it as `candidate_bad`.

### Round 3: Phase 4 — inter-link cross-pair

Detect degraded spine/uplink between healthy 4n groups:

1. From healthy groups in Round 1, generate disjoint 4n-pair matchings (each healthy group appears at most once per wave).
2. Submit one sbatch per pair, each running 8-node ring `all_reduce_perf`.
3. Compute in-run 8n median + stddev. Flag pairs whose 8n bw is below threshold despite both 4n halves being healthy → likely a degraded path between A's spine and B's spine.

### Round 4: Phase 5 — confirmation retest

Marginal Phase 3 `candidate_bad` results (z above `--retest-z-threshold`, default -10) are re-tested with a *different* H-triple drawn from a disjoint shuffle of the H-pool. Strong signals (z ≤ -10) skip retest.

Verdict per (candidate, algo):
- `confirmed_bad`: bad in original AND retest → node-intrinsic
- `topology_sensitive`: bad in original but healthy in retest → bandwidth depends on which peers it's paired with; the issue is between certain spines, not the node itself
- `confirmed_bad_no_retest`: signal already strong enough (z≤-10) that retest would waste cycles

### Round 5: Phase 6 — IB-layer per-HCA sweep (opt-in)

`--with-ib-sweep` runs 8-HCA ib_write_bw across all idle nodes:

1. Build K=2 disjoint matchings (each node tested in 2 different pair contexts):
   - Wave 1: adjacent pairs
   - Wave 2: cross-half pairs
2. Each pair sequentially probes its 8 IB HCAs (`mlx5_0, _1, _4, _5, _6, _11, _14, _15`):
   - Pre-probe `ibv_devinfo` on both sides; if either is not `PORT_ACTIVE`, record `SKIP` and move on.
   - Otherwise: server-side starts `ib_write_bw -d <hca> -F -s 8M -n 2000`, client-side connects.
3. Aggregate per (node, hca) across K=2 trials. Classify:
   - `port_down`: self port not PORT_ACTIVE in any trial
   - `node_intrinsic_severe`: all trials < 95% of cluster median
   - `node_intrinsic_suspect`: all trials < 98% of cluster median
   - `link_specific`: bad with one peer, healthy with another (degraded path between specific nodes)
   - (no flag): all trials within threshold

## Configuration

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--partition` | `batch` | Slurm partition to query |
| `--exclude` | `""` | Slurm hostlist to subtract from idle pool |
| `--max-nodes` | (none) | Cap idle-node count (debugging) |
| `--results-dir` | timestamped | Override output directory |
| `--dry-run` | off | Print sbatch plan; submit nothing |
| `--job-timeout-s` | `180` | Per-job watchdog timeout for "no data line yet" |
| `--rebaseline` | off | Overwrite saved cluster baseline from this run's stats |
| `--skip-localize` | off | Skip Round 2 (Phase 3) |
| `--skip-cross-pair` | off | Skip Round 3 (Phase 4) |
| `--skip-confirm` | off | Skip Round 4 (Phase 5) |
| `--retest-z-threshold` | `-10.0` | Above this z, retest the candidate (Phase 5); below = strong signal, no retest |
| `--with-ib-sweep` | off | Enable Round 5 (Phase 6) |
| `--n-pairs` | `12` | Phase 4 cross-pair sample size cap |

### Cluster-specific constants (in source)

Adapt these for a different cluster:

```python
PARTITION = "batch"
GPUS_PER_NODE = 8
GROUP_SIZE = 4
BINARY_PATH = "/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB = "/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT = "/var/tmp/hpcx-2.18/hpcx-init.sh"  # local-disk fallback
RESULTS_ROOT = Path("/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers")
BASELINE_DIR = Path.home() / "together-nccl-tests" / "baselines"
CLUSTER_NAME = "use3a-ss"
IB_HCAS = ["mlx5_0", "mlx5_1", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_11", "mlx5_14", "mlx5_15"]
```

The HPC-X cascade in the generated sbatch prefers `/mnt/vast` (shared FS, robust to per-node `/var/tmp` drift). If your cluster has a different shared HPC-X path, update `build_sbatch()`.

### Baseline file

`~/together-nccl-tests/baselines/<cluster>/healthy_4n.json` — auto-seeded on first run from the trimmed quartile of healthy groups. Contains per-algo `median_busbw_gbs`, `stddev_busbw_gbs`, `n_samples`. Loaded silently on subsequent runs; pass `--rebaseline` to overwrite after a known cluster change.

## Output: the only structures you need to know

```jsonc
{
  "iteration": "1.5",
  "timestamp_utc": "...",
  "n_idle": 66,
  "groups": [
    {"gid": "01", "nodes": [...],
     "algos": {"ring": {...}, "collnet_sharp": {...}},
     "perf_label": "severe", "perf_scores": {...}}
  ],
  "bootstrap_failed_groups": [
    {"gid": "01", "nodes": [...], "failed_algos": [...],
     "failure_classes": ["missing_lib"], "suspect_nodes": [...]}
  ],
  "node_state_failed": [
    {"node": "...", "class": "missing_lib", "missing_lib": "libcudart.so.12", ...}
  ],
  "perf_stragglers": [...],         // Round 1 z-score severe/suspect groups
  "localization": {                  // Round 2 Phase 3 results
    "waves": [{"algo": "ring", "tests": [...]}],
    "localized_bad_nodes": ["use3a-ss-b200-gpu-184", ...]
  },
  "confirm": {                       // Round 4 Phase 5 results
    "skipped_strong": [...],
    "retest_waves": [...]
  },
  "cross_pair": {                    // Round 3 Phase 4 results
    "median_8n_ring_gbs": 306.8,
    "inter_link_suspects": [...]
  },
  "ib_sweep": {                      // Round 5 Phase 6 (with --with-ib-sweep)
    "k": 2,
    "per_hca_stats": {"mlx5_0": {"median_mibs": ..., ...}, ...},
    "bad_hcas": [
      {"node": "...", "hca": "mlx5_0", "status": "port_down", ...}
    ]
  }
}
```

## Known limitations

- **Single-snapshot** — flapping ports (e.g. mlx5_0 oscillating UP/DOWN every few hours) can pass any single sweep. Mitigation: run 2-3× per day; persistent state log (planned Iteration 8) would aggregate.
- **K=2 is the maximum** — distinguishes node-bound from link-bound HCA degradation, but does not isolate single specific switch ports. K=3+ would, at higher cost; deferred until needed.
- **NCCL Phase 3 localization assumes H-pool is truly healthy** — if a "healthy" group is borderline, it can introduce noise into localization tests. Phase 5 retest with rotated H-triples mitigates.
- **No latency / packet-loss / counter readings** — Phase 6 reads only bandwidth; doesn't read `ibstat` symbol-error counters or perfquery latencies. Some HCA failure modes (e.g. high error-rate but full bandwidth) won't trigger.
- **Healthy baseline tied to one cluster** — if cluster topology / hardware genuinely changes (firmware, fabric upgrade), the saved baseline becomes wrong. Use `--rebaseline` to refresh.

## Architecture / extension points

The script is a single `find_stragglers.py` (~1100 lines). Functions of interest:

- `discover_idle()`, `select_h_pool()` — node selection
- `build_sbatch()` — generates the per-NCCL-job sbatch text (4n group, ring or collnet_sharp)
- `build_ib_pair_sbatch()` — generates the per-IB-pair sbatch text (8 HCAs sequentially)
- `watchdog()` — scancel-on-data-line + per-job-timeout polling
- `classify_failure()` — bootstrap-fail signature classifier
- `classify_perf()` — z-score → healthy/suspect/severe
- `compute_in_run_baseline()` — trimmed-quartile median+stddev
- `phase3_localize()`, `phase4_cross_pair()`, `phase5_confirm()`, `phase6_ib_sweep()` — round implementations
- `parse_busbw()`, `parse_ib_pair_output()` — output parsers

To add a new phase: add a `phaseN_*()` function that takes `(out, results_dir, sizes, args, ...)` and returns a dict with structured findings; wire it into `main()` after the existing phases; add the CLI flag and output section.

## Files written

- `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers/<ts>/`
  - `results.json` — structured all-rounds output
  - `phase1_grp<NN>_<algo>.out` — per-group sbatch logs
  - `phase3_<algo>_g<NN>_c<short>.out` — per-localization-test logs
  - `phase4_pair<NN>_ring.out` — per-cross-pair logs
  - `phase5_<algo>_g<NN>_c<short>_retest.out` — per-retest logs
  - `phase6_w<W>_pair<NN>.out` — per-IB-pair logs
- `~/together-nccl-tests/baselines/<cluster>/healthy_4n.json` — saved baseline

## See also

- `~/johnson/worklog/2026-04-28_straggler_finder_report.md` — initial findings + tool overview
- `~/johnson/worklog/2026-04-28_ib_layer_investigation.md` — IB layer + Phase 6 + K=2 + flapping discovery
- `submit_per4_groups.sh` (sibling project) — manual per-4-node sweep, predecessor to this tool
- 04-26 worklog for the manual diagnosis this tool was designed to automate
