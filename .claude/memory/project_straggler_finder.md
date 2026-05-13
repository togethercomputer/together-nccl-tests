---
name: find-stragglers-py-straggler-detection-tool
description: "2-round disjoint-pair sweep for slinky/flapping-airplanes — Round 1 watchdog, Round 2 localization"
metadata: 
  node_type: memory
  type: project
  originSessionId: 40ae4eee-9e1c-40ba-92a1-c2fca3af82b8
---

## Location
`/home/johnson/together-nccl-tests/stragglers/find_stragglers.py` (~622 lines, rewritten 2026-05-13 commit c50a13c).
README: `/home/johnson/together-nccl-tests/stragglers/README.md`

## What it does
2-round automated diagnostic that flags bad nodes in the slinky/flapping-airplanes B200 cluster via ring all_reduce, then localizes the culprit within a bad group.

## 2-Round methodology

| Round | What it does |
|---|---|
| 1 | All discovered idle nodes split into disjoint 2-node groups. Each group runs ring all_reduce at 8G, 60s per-job watchdog timeout (no busbw line → bad group). z-score classify against saved baseline. |
| 2 | For each bad group's 2 nodes, pair each candidate with 1 healthy H-node (distinct H per candidate, all run in parallel). `timed_out` or no busbw → `candidate_bad`; suspect/severe z → `candidate_bad_perf`; otherwise → `candidate_healthy`. |

## Failure classes (in `classify_failure`)
- `missing_lib` — `error while loading shared libraries` regex + suspect node from `srun ... Exited with exit code 127`
- `pmix_wedge` — `collective timeout seq=` + suspect node from `wait contrib: slinky-NN` counter
- `unknown_hang` — no recognized signature
- `no_output` — outfile missing entirely

## Performance classification (z-score against baseline)
- `healthy` — within ±2σ
- `suspect` — z < -2 (below)
- `severe` — z < -5
- `outlier_fast` — z > +2 (faster than baseline)
- Stddev floor: 0.5% of median (prevents tight clusters from flagging everything)

## Cluster constants (hardcoded for flapping-airplanes/slinky)
```python
PARTITION    = "slinky"
GROUP_SIZE   = 2
BINARY_PATH  = os.path.expanduser("~/together-nccl-tests/build")
NCCL_LIB     = "/usr/lib/x86_64-linux-gnu"
HPCX_INIT    = "/opt/hpcx/hpcx-init.sh"
RESULTS_ROOT = Path("/data/home/johnson/nccl-results/flapping-airplanes/find-stragglers")
CLUSTER_NAME = "flapping-airplanes"
DEFAULT_SIZES = ("8G","8G",2)   # min=max=8G
WARMUP, ITERS = 2, 5
JOB_TIMEOUT_S    = 60
POLL_INTERVAL_S  = 30
WATCHDOG_BUDGET_S = 1200
```

sbatch sets `UCX_TLS=self,sm,cuda_ipc,cuda_copy,tcp`, `UCX_NET_DEVICES=eth0`, `NCCL_SOCKET_IFNAME=eth0`, `OMPI_MCA_btl_tcp_if_include=eth0`, `ulimit -l unlimited`, `NCCL_ALGO=Ring`, `NCCL_NVLS_ENABLE=0`, `NCCL_COLLNET_ENABLE=0`. Does NOT set `NCCL_IB_HCA` (different from `run_slurm.sh` — see [[project_nccl_working_config]]).

## CLI flags
```bash
find_stragglers.py                          # full 2-round
find_stragglers.py --dry-run                # print sbatch plan only
find_stragglers.py --skip-localize          # Round 1 only
find_stragglers.py --rebaseline             # overwrite baseline from this run
find_stragglers.py --max-nodes 8            # debug with 4 groups
find_stragglers.py --exclude slinky-[4-7]   # exclude hostlist
find_stragglers.py --job-timeout-s 120      # raise per-job watchdog
find_stragglers.py --group-size 4           # 4-node groups instead of 2
```

## Baseline
`~/together-nccl-tests/baselines/flapping-airplanes/healthy_2n_ring.json` (committed 2026-05-13).
Auto-seeded on first run from trimmed-quartile of the in-run sample. `--rebaseline` overwrites.

## Output
- `<RESULTS_ROOT>/<timestamp>/results.json` — full structured JSON (groups, bad_groups, localization, baseline)
- `<RESULTS_ROOT>/<timestamp>/phase1_grp<NN>_ring.out` and `phase2_g<NN>_c<short>_ring.out` — raw NCCL outputs
- Console: per-round table with busbw, Δ%, z, status, nodes

## What's no longer here (vs the pre-2026-05-13 1604-line version)
Removed: 4-node disjoint sweep, parallel-encoding K=3 localization, inter-link cross-pair, confirmation retest, IB-layer HCA sweep via `ib_write_bw`, multi-algo support (`collnet_sharp` + `nvls`), SHARP failure class, `--with-ib-sweep` / `--skip-cross-pair` / `--skip-confirm` flags. The new version is intentionally focused on the fast 2-node ring path.

## Why: rewrite motivation
The 5-round version was tuned for use3a-ss. On flapping-airplanes, the 2-node ring at 8G with 60s watchdog turned out to catch the same bad nodes as the full 5-round sweep in a fraction of the time. The narrowed scope also made the code maintainable.

See also: [[project_nccl_working_config]] for the parallel run_slurm.sh HCA list, [[project_slinky_cluster]] for current bad-node inventory, [[feedback-worklog-routing]] for where to file straggler reports.
