#!/usr/bin/env python3
"""find_stragglers.py — Iteration 1.

Phase 0: discover truly-idle nodes in a Slurm partition.
Phase 1: disjoint 4n partition; submit ring + collnet_sharp all_reduce; watchdog;
         parse 'Avg bus bandwidth'; dump results.json.

Future iterations will add classification (z-score), localization (parallel
encoding), inter-link cross-pair check, and a JSON action report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Constants (match submit_per4_groups.sh) ─────────────────────────────────
GPUS_PER_NODE = 8
PARTITION = "batch"
GROUP_SIZE = 4
TIME_LIMIT = "00:20:00"
BINARY_PATH = "/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build"
NCCL_LIB = "/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu"
HPCX_INIT = "/var/tmp/hpcx-2.18/hpcx-init.sh"
RESULTS_ROOT = Path("/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers")
BASELINE_DIR = Path.home() / "together-nccl-tests" / "baselines"
CLUSTER_NAME = "use3a-ss"

# Performance classification thresholds (z-score against in-run trimmed stats).
# stddev gets a floor of 0.5% of the median so an ultra-tight distribution
# doesn't make sub-percent jitter look like a giant outlier.
Z_SUSPECT = 2.0   # |z| in (Z_SUSPECT, Z_SEVERE]  -> suspect
Z_SEVERE  = 5.0   # z < -Z_SEVERE                  -> severe
STDDEV_FLOOR_FRAC = 0.005

ALGOS = ["ring", "collnet_sharp"]
ALGO_ENV = {
    "ring":          {"algo": "Ring", "nvls": "0", "collnet": "0"},
    "collnet_sharp": {"algo": None,   "nvls": "0", "collnet": "1"},
}

DEFAULT_SIZES = ("2G", "16G", 2)
WARMUP, ITERS = 5, 20

POLL_INTERVAL_S = 60      # check outfiles every 1 min
WATCHDOG_BUDGET_S = 1200  # global ceiling; should never hit on healthy cluster
JOB_TIMEOUT_S = 180       # per-job; longer => bootstrap-fail (Class C dead node)

AVG_BUSBW_RE = re.compile(r"Avg bus bandwidth\s*:\s*([\d.]+)")
LIB_MISSING_RE = re.compile(
    r"error while loading shared libraries: (\S+?): cannot open shared object file"
)
SRUN_EXIT_127_RE = re.compile(r"srun: error: (\S+): tasks .*Exited with exit code 127")
PMIX_TIMEOUT_RE = re.compile(r"collective timeout seq=")
PMIX_WAIT_CONTRIB_RE = re.compile(r"wait contrib: (use3a-ss-b200-gpu-\d+)")
SHARP_INIT_RE = re.compile(r"sharp_job_id:\d+")


def discover_idle(partition: str, exclude: set[str]) -> list[str]:
    """Return sorted list of nodes with STATE=='idle' in partition.

    `sinfo -t idle` lumps drained nodes in; filter on the exact STATE column.
    """
    out = subprocess.check_output(
        ["sinfo", "-N", "-p", partition, "-h", "-o", "%N %t"], text=True
    )
    hosts = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "idle" and parts[0] not in exclude:
            hosts.append(parts[0])
    return sorted(set(hosts))


def build_sbatch(job_name: str, nodes: list[str], algo: str,
                 outfile: str, sizes: tuple) -> str:
    env = ALGO_ENV[algo]
    nodelist = ",".join(nodes)
    nccl_algo = (f"export NCCL_ALGO={env['algo']}"
                 if env["algo"] else "unset NCCL_ALGO 2>/dev/null || true")
    min_b, max_b, factor = sizes
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={len(nodes)}
#SBATCH --ntasks-per-node={GPUS_PER_NODE}
#SBATCH --gpus-per-node={GPUS_PER_NODE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition={PARTITION}
#SBATCH --output={outfile}
#SBATCH --time={TIME_LIMIT}
#SBATCH --exclusive
#SBATCH --chdir=/tmp
#SBATCH --nodelist={nodelist}

# Prefer the shared-FS path so every node in the allocation sees identical
# libmpi at the same LD_LIBRARY_PATH entry. The /var/tmp local-disk copy is
# a node-state landmine — it exists on the batch host but not all peers.
if [ -f /mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/hpcx-init.sh ]; then
    source /mnt/vast/dgxc-benchmarking-auto/hpcx-2.18/hpcx-init.sh && hpcx_load
elif [ -f {HPCX_INIT} ]; then
    source {HPCX_INIT} && hpcx_load
elif [ -f /opt/hpcx/hpcx-init.sh ]; then
    source /opt/hpcx/hpcx-init.sh && hpcx_load
else
    echo "ERROR: no hpcx-init.sh on $(hostname)" >&2; exit 127
fi
export LD_LIBRARY_PATH={NCCL_LIB}:${{LD_LIBRARY_PATH:-}}
{nccl_algo}
export NCCL_NVLS_ENABLE={env["nvls"]}
export NCCL_COLLNET_ENABLE={env["collnet"]}
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA="=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_11:1,mlx5_14:1,mlx5_15:1"
export UCX_NET_DEVICES=bond0
export OMPI_MCA_btl_tcp_if_include=bond0
export CUDA_DEVICE_MAX_CONNECTIONS=32

echo "=== {job_name} | nodes={nodelist} | algo={algo} ==="
echo "Job ID: $SLURM_JOB_ID"
date

srun --mpi=pmix {BINARY_PATH}/all_reduce_perf \\
    -b {min_b} -e {max_b} -f {factor} \\
    -g 1 -n {ITERS} -w {WARMUP}

echo "=== Done: {job_name} ==="
"""


def submit_one(sbatch_text: str) -> int:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="strag-"
    ) as f:
        f.write(sbatch_text)
        path = f.name
    try:
        out = subprocess.check_output(["sbatch", "--parsable", path], text=True)
        return int(out.strip())
    finally:
        os.unlink(path)


def watchdog(jobs: dict[int, str], poll_s: int = POLL_INTERVAL_S,
             budget_s: int = WATCHDOG_BUDGET_S,
             job_timeout_s: int = JOB_TIMEOUT_S) -> dict[int, str]:
    """Poll outfiles; scancel as soon as 'Avg bus bandwidth' appears.

    Two failure modes handled:
      1. Post-srun hang: data line written, but slurmstepd doesn't exit.
         -> scancel on data line.
      2. Bootstrap-fail (Class C): node passes sinfo idle but pmix wedges,
         so no data line is ever written. -> scancel after job_timeout_s.

    Returns {jobid: "completed" | "timed_out" | "vanished"}.
    """
    submitted_at = {jid: time.time() for jid in jobs}
    statuses: dict[int, str] = {}
    pending = dict(jobs)
    start = time.time()
    print(f"[watchdog] polling {len(pending)} jobs every {poll_s}s "
          f"(per-job timeout {job_timeout_s}s, global ceiling {budget_s}s)",
          flush=True)
    while pending:
        elapsed = time.time() - start
        if elapsed > budget_s:
            print(f"[watchdog] hit {budget_s}s global ceiling; cancelling "
                  f"{len(pending)} remaining jobs", flush=True)
            for jid in pending:
                subprocess.run(["scancel", str(jid)], capture_output=True)
                statuses[jid] = "timed_out"
            return statuses
        time.sleep(poll_s)
        for jid, outfile in list(pending.items()):
            status = None
            has_data = False
            if os.path.exists(outfile):
                try:
                    with open(outfile) as f:
                        content = f.read()
                    if AVG_BUSBW_RE.search(content):
                        has_data = True
                except OSError:
                    pass
            if has_data:
                subprocess.run(["scancel", str(jid)], capture_output=True)
                status = "completed"
            else:
                age = time.time() - submitted_at[jid]
                if age > job_timeout_s:
                    print(f"[watchdog] job {jid} exceeded {job_timeout_s}s "
                          f"without data line -> bootstrap-fail suspect",
                          flush=True)
                    subprocess.run(["scancel", str(jid)], capture_output=True)
                    status = "timed_out"
                else:
                    # Reap if Slurm finished it naturally (e.g. failure exit).
                    r = subprocess.run(
                        ["squeue", "-j", str(jid), "-h"],
                        capture_output=True, text=True
                    )
                    if not r.stdout.strip():
                        status = "vanished"
            if status:
                statuses[jid] = status
                pending.pop(jid, None)
        n_done = len(jobs) - len(pending)
        print(f"[watchdog] t={int(time.time()-start)}s done={n_done}/{len(jobs)} "
              f"pending={len(pending)}", flush=True)
    return statuses


def parse_busbw(outfile: str) -> float | None:
    if not os.path.exists(outfile):
        return None
    with open(outfile) as f:
        m = AVG_BUSBW_RE.search(f.read())
    return float(m.group(1)) if m else None


PERF_JUSTIFICATIONS = {
    "severe":
        "Group bandwidth is far below the in-run healthy median (z<-{Z_SEVERE}). "
        "Either a single bad node drags the whole group, or all nodes share a "
        "degraded fabric path. Iteration 3 will localize via parallel encoding.",
    "suspect":
        "Group is moderately below the healthy median (-{Z_SEVERE}<z<-{Z_SUSPECT}). "
        "Could be node-level drift or a borderline-bad fabric. Worth localizing.",
    "outlier_fast":
        "Group is unusually fast vs the in-run median; not a straggler, but worth "
        "noting (could indicate skewed baseline or unusually clean nodes).",
}
FAILURE_JUSTIFICATIONS = {
    "missing_lib":
        "Suspect node is missing a shared library — ranks on it exit 127 before "
        "MPI_Init; pmix then times out waiting for them. For libmpi.so* check "
        "that /var/tmp/hpcx-2.18 was staged (or fall through to /mnt/vast). For "
        "libcudart.so* the CUDA runtime is broken on the node — drain + investigate.",
    "pmix_wedge":
        "Node passed sinfo idle but its slurmstepd never contributed to the pmix "
        "ring; peers logged 'wait contrib:<suspect>' until timeout. Underlying "
        "Slurm/pmix state is broken on this node. Fix: drain + reboot.",
    "sharp_no_data":
        "SHARP init logs reached sharp_job_id allocation but the data line never "
        "appeared within the watchdog timeout. Ring on the same nodes was fine, "
        "so this is not a fabric-wide issue. Either CollNet/SHARP wedged on this "
        "specific node set, or the timeout is too tight for SHARP startup here.",
    "unknown_hang":
        "Job exceeded watchdog timeout without producing 'Avg bus bandwidth' and "
        "no recognized failure signature was found in the log.",
    "no_output":
        "Job timed out before any output was written. No log to classify.",
}


def classify_failure(outfile: str, algo: str) -> dict:
    """Inspect a bootstrap-failed log and return class + suspect_node + evidence.

    Recognized classes:
      - "missing_lib":    any libX.so missing on a remote node (exit 127 + linker err).
                          suspect_node = node srun reports failing.
      - "pmix_wedge":     pmix collective timeout with no NCCL output.
                          suspect_node = node most often listed in 'wait contrib:'.
      - "sharp_no_data":  SHARP init started but no busbw within the timeout.
                          suspect_node = None (no localization possible from log).
      - "unknown_hang":   no recognizable signature; suspect_node = None.
    """
    if not os.path.exists(outfile):
        return {"class": "no_output", "suspect_node": None, "evidence": None,
                "missing_lib": None}
    with open(outfile) as f:
        content = f.read()

    libs = LIB_MISSING_RE.findall(content)
    if libs:
        m = SRUN_EXIT_127_RE.search(content)
        evidence = None
        for line in content.splitlines():
            if "Exited with exit code 127" in line:
                evidence = line.strip()
                break
        return {
            "class": "missing_lib",
            "suspect_node": m.group(1) if m else None,
            "evidence": evidence,
            "missing_lib": libs[0],
        }

    if PMIX_TIMEOUT_RE.search(content):
        from collections import Counter
        counts = Counter(PMIX_WAIT_CONTRIB_RE.findall(content))
        suspect = counts.most_common(1)[0][0] if counts else None
        evidence = None
        if suspect:
            for line in content.splitlines():
                if f"wait contrib: {suspect}" in line:
                    evidence = line.strip()
                    break
        return {"class": "pmix_wedge", "suspect_node": suspect,
                "evidence": evidence, "missing_lib": None}

    # SHARP-specific: init started but no data — distinct from generic hang.
    if algo == "collnet_sharp" and SHARP_INIT_RE.search(content):
        evidence = "SHARP init reached sharp_job_id allocation but no "\
                   "'Avg bus bandwidth' within timeout"
        return {"class": "sharp_no_data", "suspect_node": None,
                "evidence": evidence, "missing_lib": None}

    return {"class": "unknown_hang", "suspect_node": None, "evidence": None,
            "missing_lib": None}


# ── Performance classification ──────────────────────────────────────────────


def compute_in_run_baseline(busbws: list[float]) -> dict | None:
    """Trimmed-quartile median + stddev. Drops top/bottom 25% before stats.

    Returns None if fewer than 3 valid samples (not enough to form a baseline).
    """
    import statistics
    valid = sorted(b for b in busbws if b is not None)
    if len(valid) < 3:
        return None
    q = len(valid) // 4
    trimmed = valid[q:len(valid) - q] if q else valid
    return {
        "median_busbw_gbs": statistics.median(trimmed),
        "stddev_busbw_gbs": (statistics.stdev(trimmed) if len(trimmed) > 1 else 0.5),
        "n_samples": len(trimmed),
        "trimmed_from": len(valid),
        "min": trimmed[0],
        "max": trimmed[-1],
    }


def classify_perf(busbw: float | None, baseline_for_algo: dict | None) -> dict:
    """Score one group/algo bandwidth against the algo's healthy baseline."""
    if busbw is None or baseline_for_algo is None:
        return {"label": "no_data", "z": None, "delta_pct": None}
    median = baseline_for_algo["median_busbw_gbs"]
    stddev = max(baseline_for_algo["stddev_busbw_gbs"],
                 median * STDDEV_FLOOR_FRAC)
    z = (busbw - median) / stddev
    delta_pct = (busbw - median) / median * 100
    if z < -Z_SEVERE:
        label = "severe"
    elif z < -Z_SUSPECT:
        label = "suspect"
    elif z > Z_SUSPECT:
        label = "outlier_fast"
    else:
        label = "healthy"
    return {"label": label, "z": z, "delta_pct": delta_pct}


def select_h_pool(groups: list[dict]) -> list[str]:
    """Return nodes from groups labelled 'healthy' on every algo."""
    pool = []
    for g in groups:
        if g.get("perf_label") == "healthy":
            pool.extend(g["nodes"])
    return pool


def phase3_localize(out: dict, results_dir: Path, sizes: tuple,
                    in_run_baseline: dict, args) -> dict:
    """Round 2 — parallel-encoding localization on each severe group.

    For each severe group, for each algo where the group's score is suspect/severe,
    submit one job per candidate paired with 3 disjoint healthy H-nodes. Within an
    algo wave, every test gets a distinct H-triple so all tests run in parallel.

    Returns dict keyed by 'waves' and 'localized_bad_nodes'.
    """
    severe_groups = [g for g in out["groups"] if g.get("perf_label") == "severe"]
    if not severe_groups:
        return {"round": 2, "waves": [], "localized_bad_nodes": [],
                "note": "no severe groups in Round 1"}

    h_pool = select_h_pool(out["groups"])
    if len(h_pool) < 3:
        return {"round": 2, "waves": [], "localized_bad_nodes": [],
                "note": f"insufficient healthy pool ({len(h_pool)}<3)"}

    # Build the test plan: one (gid, candidate, algo) tuple per test needed.
    plan_per_algo: dict[str, list[tuple[str, str]]] = {a: [] for a in ALGOS}
    for g in severe_groups:
        for algo in ALGOS:
            score = g["perf_scores"][algo]
            if score["label"] in ("suspect", "severe"):
                for c in g["nodes"]:
                    plan_per_algo[algo].append((g["gid"], c))

    waves: list[dict] = []
    bad_nodes_by_node: dict[str, list[dict]] = {}

    for algo in ALGOS:
        items = plan_per_algo[algo]
        if not items:
            continue
        n_tests = len(items)
        # Need disjoint H-triples for parallel execution.
        if len(h_pool) < n_tests * 3:
            print(f"  [warn ] {algo}: only {len(h_pool)} healthy nodes for "
                  f"{n_tests} tests — would need overlap; truncating.")
            usable = len(h_pool) // 3
            items = items[:usable]
            n_tests = usable

        print()
        print(f"--- Wave {len(waves)+1}: {algo} localization "
              f"({n_tests} tests, {n_tests*3} healthy slots) ---")
        submitted = []
        for i, (gid, candidate) in enumerate(items):
            h_triple = h_pool[i*3:(i+1)*3]
            test_nodes = [candidate] + list(h_triple)
            short = candidate.split("-")[-1]
            outfile = str(results_dir / f"phase3_{algo}_g{gid}_c{short}.out")
            sb = build_sbatch(
                f"strag-p3-g{gid}-c{short}-{algo}",
                test_nodes, algo, outfile, sizes,
            )
            jid = submit_one(sb)
            print(f"  [submit] grp{gid} cand={candidate} "
                  f"H={','.join(h_triple)}  -> job {jid}")
            submitted.append({
                "gid": gid, "candidate": candidate, "h_nodes": list(h_triple),
                "algo": algo, "jobid": jid, "outfile": outfile,
            })

        job_map = {s["jobid"]: s["outfile"] for s in submitted}
        print()
        statuses = watchdog(job_map, job_timeout_s=args.job_timeout_s)

        for s in submitted:
            s["status"] = statuses.get(s["jobid"], "unknown")
            bw = parse_busbw(s["outfile"])
            s["busbw_gbs"] = bw
            score = classify_perf(bw, in_run_baseline.get(algo))
            s["z"] = score["z"]
            s["delta_pct"] = score["delta_pct"]
            s["perf_label"] = score["label"]
            if s["status"] == "timed_out":
                s["verdict"] = "test_bootstrap_failed"
                s["fail_classification"] = classify_failure(s["outfile"], algo)
            elif score["label"] == "healthy":
                s["verdict"] = "candidate_healthy"
            else:
                s["verdict"] = "candidate_bad"
                bad_nodes_by_node.setdefault(s["candidate"], []).append({
                    "algo": algo, "delta_pct": s["delta_pct"], "z": s["z"],
                    "label": score["label"],
                })

        waves.append({"algo": algo, "tests": submitted})

    localized = sorted(bad_nodes_by_node.keys())
    return {
        "round": 2, "waves": waves,
        "localized_bad_nodes": localized,
        "evidence_per_node": bad_nodes_by_node,
    }


IB_HCAS = ["mlx5_0", "mlx5_1", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_11", "mlx5_14", "mlx5_15"]
IB_SIZE_BYTES = 8388608
IB_ITERS = 2000
IB_LINE_SPEED_GBS = 50.0  # NDR400


def build_ib_pair_sbatch(host_a: str, host_b: str, outfile: str) -> str:
    """One sbatch that benchmarks all 8 HCAs sequentially between host_a and host_b
    using ib_write_bw. Skips HCAs where either side reports the port not active.
    Output format (parseable):
        PAIR <server> <client>
        HCA <hca> SKIP server=<state> client=<state>
        HCA <hca> BW <bw_mibs_avg>
    """
    short_a = host_a.split("-")[-1]
    short_b = host_b.split("-")[-1]
    hca_list = " ".join(IB_HCAS)
    return f"""#!/bin/bash
#SBATCH --job-name=ib-pair-{short_a}-{short_b}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output={outfile}
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --chdir=/tmp
#SBATCH --nodelist={host_a},{host_b}

set -u
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER=${{HOSTS[0]}}
CLIENT=${{HOSTS[1]}}
HCAS=({hca_list})

echo "PAIR $SERVER $CLIENT"

for hca in "${{HCAS[@]}}"; do
    SS=$(srun -N1 --nodelist=$SERVER --ntasks=1 --quiet --exclusive \\
        ibv_devinfo -d $hca 2>/dev/null | awk '/state:/{{print $2; exit}}')
    CS=$(srun -N1 --nodelist=$CLIENT --ntasks=1 --quiet --exclusive \\
        ibv_devinfo -d $hca 2>/dev/null | awk '/state:/{{print $2; exit}}')
    if [[ "$SS" != "PORT_ACTIVE" || "$CS" != "PORT_ACTIVE" ]]; then
        echo "HCA $hca SKIP server=${{SS:-MISSING}} client=${{CS:-MISSING}}"
        continue
    fi
    PORT=$((18515 + RANDOM % 1000))
    timeout 30 srun -N1 --nodelist=$SERVER --ntasks=1 --quiet --exclusive \\
        ib_write_bw -d $hca -F -s {IB_SIZE_BYTES} -n {IB_ITERS} -p $PORT >/dev/null 2>&1 &
    SPID=$!
    sleep 3
    BW=$(timeout 30 srun -N1 --nodelist=$CLIENT --ntasks=1 --quiet --exclusive \\
        ib_write_bw $SERVER -d $hca -F -s {IB_SIZE_BYTES} -n {IB_ITERS} -p $PORT 2>/dev/null \\
        | awk '/^ *{IB_SIZE_BYTES} /{{print $4}}')
    wait $SPID 2>/dev/null
    if [[ -z "$BW" ]]; then
        echo "HCA $hca FAIL"
    else
        echo "HCA $hca BW $BW"
    fi
done
"""


def parse_ib_pair_output(outfile: str) -> dict:
    """Parse the structured PAIR/HCA lines emitted by build_ib_pair_sbatch."""
    result = {"server": None, "client": None, "per_hca": {}}
    if not os.path.exists(outfile):
        return result
    with open(outfile) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "PAIR" and len(parts) >= 3:
                result["server"], result["client"] = parts[1], parts[2]
            elif parts[0] == "HCA" and len(parts) >= 3:
                hca = parts[1]
                if parts[2] == "BW" and len(parts) >= 4:
                    try:
                        result["per_hca"][hca] = {
                            "status": "ok",
                            "busbw_mibs": float(parts[3]),
                        }
                    except ValueError:
                        result["per_hca"][hca] = {"status": "parse_error",
                                                   "raw": " ".join(parts[3:])}
                elif parts[2] == "SKIP":
                    states = {kv.split("=")[0]: kv.split("=")[1]
                              for kv in parts[3:] if "=" in kv}
                    result["per_hca"][hca] = {
                        "status": "skip",
                        "server_state": states.get("server", "?"),
                        "client_state": states.get("client", "?"),
                    }
                elif parts[2] == "FAIL":
                    result["per_hca"][hca] = {"status": "fail"}
    return result


def build_disjoint_matchings(hosts: list[str], k: int) -> list[list[tuple[str, str]]]:
    """Return up to k disjoint perfect matchings of hosts.

    Wave 1: adjacent pairs (h_0,h_1), (h_2,h_3), ...
    Wave 2: cross-half pairs (h_0,h_{N/2}), (h_1,h_{N/2+1}), ...

    Each node appears at most once per wave; partners differ across waves so a
    node is tested against different peers, enabling node-vs-link disambiguation.
    """
    n = (len(hosts) // 2) * 2
    if n < 2:
        return []
    matchings = [[(hosts[2 * i], hosts[2 * i + 1]) for i in range(n // 2)]]
    if k >= 2 and n >= 4:
        half = n // 2
        matchings.append([(hosts[i], hosts[i + half]) for i in range(half)])
    return matchings[:k]


def _wait_for_jobs(job_ids: list[int], label: str = "", ceiling_s: int = 600):
    """Poll until none of the jobs are in squeue. ib_write_bw exits cleanly."""
    if not job_ids:
        return
    poll_start = time.time()
    while True:
        if time.time() - poll_start > ceiling_s:
            print(f"  [wait{label}] {ceiling_s}s ceiling hit; cancelling")
            subprocess.run(["scancel"] + [str(j) for j in job_ids],
                           capture_output=True)
            return
        r = subprocess.run(
            ["squeue", "-h", "-j", ",".join(str(j) for j in job_ids), "-o", "%i"],
            capture_output=True, text=True,
        )
        running = [x for x in r.stdout.split() if x.strip()]
        if not running:
            print(f"  [wait{label}] drained after {int(time.time()-poll_start)}s")
            return
        time.sleep(20)


def phase6_ib_sweep(idle_hosts: list[str], results_dir: Path, args, k: int = 2) -> dict:
    """Round 5 — IB layer per-HCA sweep with K=2 disjoint pair matchings.

    Each node is tested in K different pair contexts. Disambiguates:
      - node_intrinsic_bad:  HCA is slow/down regardless of peer (all K trials bad)
      - link_specific:       HCA is slow only with one peer (mixed trials)
      - port_down:           self port not PORT_ACTIVE in any trial
      - healthy:             all K trials within threshold
    """
    n = (len(idle_hosts) // 2) * 2
    if n < 4:
        return {"phase": 6, "skipped": f"need >=4 idle nodes, have {n}"}
    hosts = idle_hosts[:n]

    matchings = build_disjoint_matchings(hosts, k=k)
    if not matchings:
        return {"phase": 6, "skipped": "no matchings produced"}
    actual_k = len(matchings)

    print()
    print(f"--- Round 5: IB sweep — K={actual_k} waves × {n//2} pairs "
          f"across {n} nodes, {len(IB_HCAS)} HCAs each ---")

    waves_data = []
    for wave_idx, pairs in enumerate(matchings, 1):
        print()
        print(f"--- Wave {wave_idx}/{actual_k} ({len(pairs)} pairs) ---")
        submitted = []
        for i, (host_a, host_b) in enumerate(pairs, 1):
            outfile = str(results_dir / f"phase6_w{wave_idx}_pair{i:02d}.out")
            sb = build_ib_pair_sbatch(host_a, host_b, outfile)
            jid = submit_one(sb)
            print(f"  [submit] W{wave_idx} pair{i:02d}: {host_a} <-> {host_b}  -> job {jid}")
            submitted.append({
                "wave": wave_idx, "pair_idx": i,
                "host_a": host_a, "host_b": host_b,
                "jobid": jid, "outfile": outfile,
            })
        print(f"  [wait W{wave_idx}] for {len(submitted)} jobs to drain...")
        _wait_for_jobs([s["jobid"] for s in submitted], label=f" W{wave_idx}")
        waves_data.append({"wave": wave_idx, "submitted": submitted})

    # ── Parse all outputs into per-(node, hca) trial lists ─────────────────
    per_node_hca_trials: dict[str, dict[str, list[dict]]] = {}
    for wave in waves_data:
        for s in wave["submitted"]:
            parsed = parse_ib_pair_output(s["outfile"])
            s["parsed"] = parsed
            server = parsed.get("server")
            client = parsed.get("client")
            for hca, entry in parsed["per_hca"].items():
                for (node, side) in [(server, "server"), (client, "client")]:
                    if node is None:
                        continue
                    peer = client if side == "server" else server
                    trial = {
                        "wave": s["wave"],
                        "side": side,
                        "peer": peer,
                        "status": entry["status"],
                    }
                    if entry["status"] == "ok":
                        trial["busbw_mibs"] = entry["busbw_mibs"]
                    elif entry["status"] == "skip":
                        self_state = (entry.get("server_state") if side == "server"
                                      else entry.get("client_state"))
                        peer_state = (entry.get("client_state") if side == "server"
                                      else entry.get("server_state"))
                        trial["self_port_state"] = self_state
                        trial["peer_port_state"] = peer_state
                        trial["self_is_down"] = (self_state != "PORT_ACTIVE")
                    per_node_hca_trials.setdefault(node, {}).setdefault(hca, []).append(trial)

    # ── Cluster-wide stats per HCA (from all 'ok' trials across waves) ─────
    import statistics
    per_hca_stats = {}
    for hca in IB_HCAS:
        bws = []
        for node, hcamap in per_node_hca_trials.items():
            for t in hcamap.get(hca, []):
                if t["status"] == "ok":
                    bws.append(t["busbw_mibs"])
        if len(bws) < 2:
            per_hca_stats[hca] = None
            continue
        med = statistics.median(bws)
        sd = statistics.stdev(bws) if len(bws) > 1 else 0.0
        per_hca_stats[hca] = {
            "median_mibs": med, "stddev_mibs": sd,
            "min_mibs": min(bws), "max_mibs": max(bws),
            "n_samples": len(bws),
        }

    # ── Classify per (node, hca) using K trials ────────────────────────────
    bad_hcas = []
    for node, hcamap in per_node_hca_trials.items():
        for hca, trials in hcamap.items():
            # 1) Self port_down (definitive — port-state probe)
            if any(t.get("self_is_down") for t in trials):
                bad_hcas.append({
                    "node": node, "hca": hca, "status": "port_down",
                    "n_trials": len(trials),
                    "reason": "self port reported not PORT_ACTIVE",
                })
                continue

            ok_trials = [t for t in trials if t["status"] == "ok"]
            if not ok_trials:
                # All skipped because peer was down — can't say anything about this node
                continue

            stats = per_hca_stats.get(hca)
            if stats is None:
                continue
            cluster_med = stats["median_mibs"]

            # Per-trial classification
            slow_trials = [t for t in ok_trials
                           if t["busbw_mibs"] < cluster_med * 0.98]
            severe_trials = [t for t in ok_trials
                             if t["busbw_mibs"] < cluster_med * 0.95]

            if not slow_trials:
                continue  # healthy

            n_ok = len(ok_trials)
            n_slow = len(slow_trials)

            if n_slow == n_ok:
                # All trials slow -> consistent, the node's HCA is bad
                lvl = "node_intrinsic_severe" if severe_trials == ok_trials \
                      else "node_intrinsic_suspect"
                med_for_node = statistics.median(t["busbw_mibs"] for t in ok_trials)
                bad_hcas.append({
                    "node": node, "hca": hca, "status": lvl,
                    "n_trials": n_ok,
                    "median_mibs_for_node": med_for_node,
                    "pct_of_cluster_median": med_for_node / cluster_med * 100,
                    "peers": sorted({t["peer"] for t in ok_trials}),
                    "reason": f"slow in all {n_ok} trials with different peers",
                })
            else:
                # Mixed: slow with some peers, fine with others -> link/topology
                slowest = min(ok_trials, key=lambda t: t["busbw_mibs"])
                fastest = max(ok_trials, key=lambda t: t["busbw_mibs"])
                bad_hcas.append({
                    "node": node, "hca": hca, "status": "link_specific",
                    "n_trials": n_ok,
                    "slow_peer": slowest["peer"],
                    "slow_pct": slowest["busbw_mibs"] / cluster_med * 100,
                    "fast_peer": fastest["peer"],
                    "fast_pct": fastest["busbw_mibs"] / cluster_med * 100,
                    "reason": (f"slow with peer {slowest['peer']} "
                               f"({slowest['busbw_mibs']/cluster_med*100:.1f}%) "
                               f"but healthy with {fastest['peer']} "
                               f"({fastest['busbw_mibs']/cluster_med*100:.1f}%)"),
                })

    return {
        "phase": 6,
        "k": actual_k,
        "n_pairs_per_wave": n // 2,
        "n_nodes_tested": n,
        "per_hca_stats": per_hca_stats,
        "per_node_hca_trials": per_node_hca_trials,
        "bad_hcas": bad_hcas,
        "waves": waves_data,
    }


def phase5_confirm(loc_result: dict, h_pool: list[str], results_dir: Path,
                   sizes: tuple, in_run_baseline: dict, args,
                   z_threshold: float = -10.0) -> dict:
    """Round 4 — confirmation retest.

    For each Phase 3 candidate_bad with marginal z (z > z_threshold), re-test
    with a different H-triple drawn from a disjoint shuffle of the H-pool.
    Strong signals (z <= z_threshold, e.g. -29 for the gpu-184 SHARP case)
    are skipped — they're already unambiguous.

    Verdict per (candidate, algo):
      - confirmed_bad:        bad in original AND retest
      - topology_sensitive:   bad in original but healthy in retest (or vice versa)
                              -> the candidate's bandwidth depends on which
                                 healthy peers it's paired with; the issue is
                                 between certain spines, not the node itself
    Strong signals retain "candidate_bad" without retest.
    """
    if not loc_result or not loc_result.get("waves"):
        return {"phase": 5, "retest_waves": [], "note": "no localization to confirm"}

    to_retest: list[dict] = []
    strong_passes: list[dict] = []
    for wave in loc_result["waves"]:
        for t in wave["tests"]:
            if t.get("verdict") != "candidate_bad":
                continue
            z = t.get("z")
            if z is not None and z <= z_threshold:
                strong_passes.append({
                    "gid": t["gid"], "candidate": t["candidate"], "algo": t["algo"],
                    "z": z, "busbw": t["busbw_gbs"],
                    "verdict": "confirmed_bad_no_retest",
                })
                continue
            to_retest.append({
                "gid": t["gid"], "candidate": t["candidate"], "algo": t["algo"],
                "original_h": t["h_nodes"],
                "original_z": z, "original_busbw": t["busbw_gbs"],
            })

    if not to_retest:
        return {"phase": 5, "retest_waves": [],
                "skipped_strong": strong_passes,
                "note": f"no marginal candidates (z>{z_threshold}) — nothing to retest"}

    print()
    print(f"--- Round 4: confirmation retest "
          f"({len(to_retest)} marginal candidates, "
          f"{len(strong_passes)} strong signals skipped) ---")

    by_algo: dict[str, list[dict]] = {}
    for e in to_retest:
        by_algo.setdefault(e["algo"], []).append(e)

    import random
    retest_waves = []
    for algo, entries in by_algo.items():
        random.seed(hash(algo) & 0xffff)  # deterministic but distinct shuffle
        pool = list(h_pool)
        random.shuffle(pool)

        used_h: set[str] = set()
        submitted = []
        for e in entries:
            forbidden = set(e["original_h"]) | used_h
            available = [n for n in pool if n not in forbidden]
            if len(available) < 3:
                print(f"  [skip] grp{e['gid']} {e['candidate']} {algo}: "
                      f"H-pool exhausted ({len(available)} usable)")
                continue
            h_triple = available[:3]
            used_h.update(h_triple)

            test_nodes = [e["candidate"]] + list(h_triple)
            short = e["candidate"].split("-")[-1]
            outfile = str(results_dir
                          / f"phase5_{algo}_g{e['gid']}_c{short}_retest.out")
            sb = build_sbatch(
                f"strag-p5-g{e['gid']}-c{short}-{algo}",
                test_nodes, algo, outfile, sizes,
            )
            jid = submit_one(sb)
            print(f"  [submit] grp{e['gid']} {e['candidate']} {algo}  "
                  f"H_new={','.join(h_triple)}  (was {','.join(e['original_h'])})  "
                  f"-> job {jid}")
            submitted.append({
                "gid": e["gid"], "candidate": e["candidate"],
                "algo": algo, "h_nodes": list(h_triple),
                "original_h": e["original_h"], "original_z": e["original_z"],
                "original_busbw": e["original_busbw"],
                "jobid": jid, "outfile": outfile,
            })

        print()
        job_map = {s["jobid"]: s["outfile"] for s in submitted}
        statuses = watchdog(job_map, job_timeout_s=args.job_timeout_s)

        for s in submitted:
            s["status"] = statuses.get(s["jobid"], "unknown")
            s["busbw_gbs"] = parse_busbw(s["outfile"])
            score = classify_perf(s["busbw_gbs"], in_run_baseline.get(algo))
            s["z"] = score["z"]
            s["delta_pct"] = score["delta_pct"]
            s["perf_label"] = score["label"]
            if s["status"] == "timed_out" or s["busbw_gbs"] is None:
                s["final_verdict"] = "retest_failed"
            elif score["label"] in ("suspect", "severe"):
                s["final_verdict"] = "confirmed_bad"
            else:
                s["final_verdict"] = "topology_sensitive"

        retest_waves.append({"algo": algo, "tests": submitted})

    return {
        "phase": 5,
        "retest_z_threshold": z_threshold,
        "retest_waves": retest_waves,
        "skipped_strong": strong_passes,
    }


def phase4_cross_pair(out: dict, results_dir: Path, sizes: tuple,
                      args, n_pairs: int = 12) -> dict:
    """Round 3 — inter-link cross-pair check.

    Sample pairs of confirmed-healthy 4n groups, run ring on the combined 8n
    allocation, and flag pairs whose 8n bandwidth is significantly below the
    in-run 8n median. A pair flagged here is a likely inter-link/spine issue:
    both halves were individually fine at 4n, yet joining them is slow.
    """
    healthy = [g for g in out["groups"] if g.get("perf_label") == "healthy"]
    if len(healthy) < 2:
        return {"round": 3, "n_pairs": 0, "pairs": [],
                "note": f"only {len(healthy)} healthy 4n groups available"}

    # Build *disjoint* pairs: no group appears in more than one pair this wave,
    # otherwise Slurm serializes the conflicting jobs and they miss the timeout.
    import random, statistics
    random.seed(42)
    shuffled = list(healthy)
    random.shuffle(shuffled)
    pairs = []
    while len(shuffled) >= 2 and len(pairs) < n_pairs:
        a = shuffled.pop()
        b = shuffled.pop()
        pairs.append((a, b))

    print()
    print(f"--- Round 3: {len(pairs)} cross-pairs sampled from "
          f"{len(healthy)} healthy 4n groups (8n ring) ---")

    submitted = []
    for i, (gA, gB) in enumerate(pairs, 1):
        nodes = list(gA["nodes"]) + list(gB["nodes"])
        outfile = str(results_dir / f"phase4_pair{i:02d}_ring.out")
        sb = build_sbatch(
            f"strag-p4-pair{i:02d}-ring", nodes, "ring", outfile, sizes,
        )
        jid = submit_one(sb)
        print(f"  [submit] pair{i:02d}: grp{gA['gid']}+grp{gB['gid']}  -> job {jid}")
        submitted.append({
            "pair_idx": i,
            "gA_gid": gA["gid"], "gB_gid": gB["gid"],
            "nodes_A": list(gA["nodes"]), "nodes_B": list(gB["nodes"]),
            "jobid": jid, "outfile": outfile,
        })

    print()
    job_map = {s["jobid"]: s["outfile"] for s in submitted}
    statuses = watchdog(job_map, job_timeout_s=args.job_timeout_s)

    bws = []
    for s in submitted:
        s["status"] = statuses.get(s["jobid"], "unknown")
        s["busbw_gbs"] = parse_busbw(s["outfile"])
        if s["busbw_gbs"] is not None:
            bws.append(s["busbw_gbs"])

    if len(bws) < 3:
        return {"round": 3, "n_pairs": len(submitted), "pairs": submitted,
                "note": f"only {len(bws)} valid 8n samples; not classifying",
                "inter_link_suspects": []}

    median_8n = statistics.median(bws)
    stddev_8n = statistics.stdev(bws) if len(bws) > 1 else 1.0
    stddev_floor = max(stddev_8n, median_8n * STDDEV_FLOOR_FRAC)

    suspects = []
    for s in submitted:
        bw = s["busbw_gbs"]
        if bw is None:
            s["z"] = None
            s["delta_pct"] = None
            s["label"] = "no_data"
            continue
        z = (bw - median_8n) / stddev_floor
        s["z"] = z
        s["delta_pct"] = (bw - median_8n) / median_8n * 100
        if z < -Z_SEVERE:
            s["label"] = "severe"
        elif z < -Z_SUSPECT:
            s["label"] = "suspect"
        elif z > Z_SUSPECT:
            s["label"] = "outlier_fast"
        else:
            s["label"] = "healthy"
        if s["label"] in ("severe", "suspect"):
            suspects.append(s)

    return {
        "round": 3,
        "n_pairs": len(submitted),
        "median_8n_ring_gbs": median_8n,
        "stddev_8n_ring_gbs": stddev_8n,
        "pairs": submitted,
        "inter_link_suspects": suspects,
    }


def load_or_seed_baseline(in_run_baseline: dict[str, dict],
                          force_rebaseline: bool,
                          source_label: str) -> tuple[dict, str]:
    """Load saved cluster baseline, or seed it from this run's healthy quartile.

    Returns (baseline_doc, origin) where origin in {"loaded", "seeded", "rebaselined"}.
    """
    path = BASELINE_DIR / CLUSTER_NAME / "healthy_4n.json"
    if path.exists() and not force_rebaseline:
        with open(path) as f:
            return json.load(f), "loaded"

    # Seed or rebaseline.
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "cluster": CLUSTER_NAME,
        "group_size": GROUP_SIZE,
        "sizes": f"{DEFAULT_SIZES[0]}-{DEFAULT_SIZES[1]} factor={DEFAULT_SIZES[2]}",
        "iters": f"{WARMUP}w+{ITERS}m",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "z_thresholds": {"suspect": Z_SUSPECT, "severe": Z_SEVERE},
        "stddev_floor_frac": STDDEV_FLOOR_FRAC,
        "algos": in_run_baseline,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc, ("rebaselined" if path.exists() and force_rebaseline else "seeded")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition", default=PARTITION)
    ap.add_argument("--exclude", default="",
                    help="Slurm hostlist to exclude from idle pool")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="cap nodes used (debug; e.g. 8 -> 2 groups)")
    ap.add_argument("--results-dir", default=None,
                    help="override results dir (default: timestamped under RESULTS_ROOT)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print sbatch plan; do not submit")
    ap.add_argument("--job-timeout-s", type=int, default=JOB_TIMEOUT_S,
                    help="per-job watchdog timeout; >this without "
                         "'Avg bus bandwidth' = bootstrap-fail (default %(default)s)")
    ap.add_argument("--rebaseline", action="store_true",
                    help="overwrite the cluster baseline file from this run's "
                         "trimmed-quartile stats")
    ap.add_argument("--skip-localize", action="store_true",
                    help="skip Round 2 (Phase 3 parallel-encoding localization)")
    ap.add_argument("--skip-cross-pair", action="store_true",
                    help="skip Round 3 (Phase 4 inter-link cross-pair)")
    ap.add_argument("--n-pairs", type=int, default=12,
                    help="number of cross-pairs to sample in Round 3 (default 12)")
    ap.add_argument("--skip-confirm", action="store_true",
                    help="skip Round 4 (Phase 5 marginal-candidate confirmation retest)")
    ap.add_argument("--retest-z-threshold", type=float, default=-10.0,
                    help="z-score above which a Phase 3 candidate_bad gets retested "
                         "(default -10; signals stronger than that are kept as-is)")
    ap.add_argument("--with-ib-sweep", action="store_true",
                    help="run Round 5 (Phase 6) IB-layer per-HCA sweep "
                         "(adds ~3 min wall, default off)")
    args = ap.parse_args()

    excl = set()
    if args.exclude:
        excl = set(subprocess.check_output(
            ["scontrol", "show", "hostnames", args.exclude], text=True
        ).split())

    hosts_all = discover_idle(args.partition, excl)
    if args.max_nodes:
        hosts_all = hosts_all[:args.max_nodes]
    n_groups = len(hosts_all) // GROUP_SIZE
    if n_groups == 0:
        print(f"ERROR: only {len(hosts_all)} idle nodes (need >= {GROUP_SIZE})",
              file=sys.stderr)
        sys.exit(1)
    used = n_groups * GROUP_SIZE
    leftover_hosts = hosts_all[used:]
    hosts = hosts_all[:used]
    groups = [hosts[i*GROUP_SIZE:(i+1)*GROUP_SIZE] for i in range(n_groups)]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================")
    print(f"  find_stragglers.py — Round 1: Phase 1 (4n disjoint sweep)")
    print(f"================================================================")
    print(f"  partition:    {args.partition}")
    print(f"  idle nodes:   {len(hosts_all)} ({used} used, {len(leftover_hosts)} leftover)")
    print(f"  groups:       {n_groups} of {GROUP_SIZE}")
    print(f"  algos:        {ALGOS}")
    print(f"  sizes:        {DEFAULT_SIZES[0]}->{DEFAULT_SIZES[1]} factor={DEFAULT_SIZES[2]}")
    print(f"  iters:        {WARMUP}w + {ITERS}m")
    print(f"  job timeout:  {args.job_timeout_s}s (>this without data line = bootstrap-fail)")
    print(f"  results dir:  {results_dir}")
    print(f"  dry-run:      {args.dry_run}")
    if leftover_hosts:
        print(f"  leftover:     {' '.join(leftover_hosts)}")
    print()

    # ── Submit Phase 1 ──────────────────────────────────────────────────────
    submitted: dict[tuple[int, str], tuple[int | None, str]] = {}
    for gidx, gnodes in enumerate(groups):
        gid = f"{gidx+1:02d}"
        for algo in ALGOS:
            outfile = str(results_dir / f"phase1_grp{gid}_{algo}.out")
            sb = build_sbatch(f"strag-p1-g{gid}-{algo}", gnodes, algo,
                              outfile, DEFAULT_SIZES)
            if args.dry_run:
                print(f"[dry-run] grp{gid} {algo:14s}  nodes={','.join(gnodes)}")
                submitted[(gidx, algo)] = (None, outfile)
            else:
                jid = submit_one(sb)
                print(f"[submit ] grp{gid} {algo:14s}  nodes={','.join(gnodes)}  -> job {jid}")
                submitted[(gidx, algo)] = (jid, outfile)

    if args.dry_run:
        print("\nDry run complete. No jobs submitted.")
        return

    # ── Watchdog ────────────────────────────────────────────────────────────
    print()
    job_map = {jid: of for (jid, of) in submitted.values() if jid is not None}
    statuses = watchdog(job_map, job_timeout_s=args.job_timeout_s)

    # ── Parse + emit JSON ───────────────────────────────────────────────────
    out = {
        "iteration": "1.5",
        "timestamp_utc": ts,
        "partition": args.partition,
        "group_size": GROUP_SIZE,
        "n_idle": len(hosts_all),
        "leftover_hosts": leftover_hosts,
        "results_dir": str(results_dir),
        "groups": [],
        "bootstrap_failed_groups": [],
        "node_state_failed": [],   # {node, class, evidence_outfile}
    }
    seen_bad_nodes: dict[str, dict] = {}
    for gidx, gnodes in enumerate(groups):
        gid = f"{gidx+1:02d}"
        per_algo = {}
        bootstrap_failed_algos = []
        for algo in ALGOS:
            jid, of = submitted[(gidx, algo)]
            status = statuses.get(jid, "unknown") if jid is not None else "dry-run"
            entry = {
                "jobid": jid,
                "status": status,
                "busbw_gbs": parse_busbw(of),
                "outfile": of,
            }
            if status == "timed_out":
                cls = classify_failure(of, algo)
                entry["failure_class"] = cls["class"]
                entry["suspect_node"] = cls["suspect_node"]
                entry["missing_lib"] = cls.get("missing_lib")
                bootstrap_failed_algos.append(algo)
                if cls["suspect_node"]:
                    seen_bad_nodes.setdefault(
                        cls["suspect_node"],
                        {"node": cls["suspect_node"], "class": cls["class"],
                         "missing_lib": cls.get("missing_lib"),
                         "evidence_outfile": of}
                    )
            per_algo[algo] = entry
        out["groups"].append({"gid": gid, "nodes": gnodes, "algos": per_algo})
        if bootstrap_failed_algos:
            classes = {per_algo[a].get("failure_class") for a in bootstrap_failed_algos}
            suspects = sorted({per_algo[a].get("suspect_node")
                               for a in bootstrap_failed_algos
                               if per_algo[a].get("suspect_node")})
            out["bootstrap_failed_groups"].append({
                "gid": gid,
                "nodes": gnodes,
                "failed_algos": bootstrap_failed_algos,
                "failure_classes": sorted(c for c in classes if c),
                "suspect_nodes": suspects,
            })
    out["node_state_failed"] = list(seen_bad_nodes.values())

    # ── Compute baseline + classify performance ─────────────────────────────
    in_run = {}
    for algo in ALGOS:
        bws = [g["algos"][algo]["busbw_gbs"] for g in out["groups"]]
        b = compute_in_run_baseline(bws)
        if b is not None:
            in_run[algo] = b

    baseline_doc, baseline_origin = (None, "skipped")
    if in_run:
        baseline_doc, baseline_origin = load_or_seed_baseline(
            in_run, args.rebaseline,
            source_label=f"trimmed quartile from {results_dir.name}",
        )
    out["baseline_origin"] = baseline_origin
    out["baseline_in_run"] = in_run
    out["baseline_used"] = (baseline_doc.get("algos") if baseline_doc else None)

    # Score each group against the *in-run* trimmed baseline (robust to drift).
    perf_stragglers = []  # (label, gid, nodes, scores_per_algo)
    for g in out["groups"]:
        scores = {}
        labels = []
        for algo in ALGOS:
            bw = g["algos"][algo]["busbw_gbs"]
            scores[algo] = classify_perf(bw, in_run.get(algo))
            labels.append(scores[algo]["label"])
        # Any algo missing data (e.g. bootstrap-failed group) → not healthy;
        # mark as "no_data" so it's excluded from the healthy H-pool downstream.
        if "no_data" in labels:
            worst_label = "no_data"
        elif "severe" in labels:
            worst_label = "severe"
        elif "suspect" in labels:
            worst_label = "suspect"
        else:
            worst_label = "healthy"
        g["perf_scores"] = scores
        g["perf_label"] = worst_label
        if worst_label in ("severe", "suspect"):
            perf_stragglers.append((worst_label, g["gid"], g["nodes"], scores))
    out["perf_stragglers"] = [
        {"label": lbl, "gid": gid, "nodes": ns,
         "scores": {algo: scores[algo] for algo in ALGOS}}
        for (lbl, gid, ns, scores) in perf_stragglers
    ]

    # ── Round 2: Phase 3 localization ───────────────────────────────────────
    if not args.skip_localize and out.get("perf_stragglers"):
        print()
        print("================================================================")
        print("  Round 2: Phase 3 — parallel-encoding localization")
        print("================================================================")
        out["localization"] = phase3_localize(
            out, results_dir, DEFAULT_SIZES, in_run, args
        )
    elif args.skip_localize:
        out["localization"] = {"skipped": "by --skip-localize"}

    # ── Round 3: Phase 4 inter-link cross-pair ──────────────────────────────
    if not args.skip_cross_pair:
        healthy_count = sum(1 for g in out["groups"]
                            if g.get("perf_label") == "healthy")
        if healthy_count >= 2:
            # Each pair must use 2 disjoint healthy groups (no group reused this wave).
            # Cap = floor(healthy_groups / 2). Also bounded by floor(idle / 8).
            cap = min(healthy_count // 2, max(1, len(hosts) // 8))
            effective_pairs = min(args.n_pairs, cap)
            if effective_pairs < args.n_pairs:
                print(f"\n  [info] capping cross-pairs to {effective_pairs} "
                      f"(healthy_groups={healthy_count}, idle={len(hosts)})")
            print()
            print("================================================================")
            print("  Round 3: Phase 4 — inter-link cross-pair check (8n ring)")
            print("================================================================")
            out["cross_pair"] = phase4_cross_pair(
                out, results_dir, DEFAULT_SIZES, args, n_pairs=effective_pairs,
            )
        else:
            out["cross_pair"] = {"skipped": f"only {healthy_count} healthy groups"}
    else:
        out["cross_pair"] = {"skipped": "by --skip-cross-pair"}

    # ── Round 4: Phase 5 confirmation retest ────────────────────────────────
    if (not args.skip_confirm and out.get("localization")
            and out["localization"].get("waves")):
        print()
        print("================================================================")
        print("  Round 4: Phase 5 — confirmation retest for marginal candidates")
        print("================================================================")
        h_pool_for_confirm = select_h_pool(out["groups"])
        out["confirm"] = phase5_confirm(
            out["localization"], h_pool_for_confirm,
            results_dir, DEFAULT_SIZES, in_run, args,
            z_threshold=args.retest_z_threshold,
        )
    elif args.skip_confirm:
        out["confirm"] = {"skipped": "by --skip-confirm"}

    # ── Round 5: Phase 6 IB-layer per-HCA sweep ─────────────────────────────
    if args.with_ib_sweep:
        print()
        print("================================================================")
        print("  Round 5: Phase 6 — IB-layer per-HCA sweep (ib_write_bw)")
        print("================================================================")
        out["ib_sweep"] = phase6_ib_sweep(hosts_all, results_dir, args)
    else:
        out["ib_sweep"] = {"skipped": "by default (use --with-ib-sweep to enable)"}

    summary = results_dir / "results.json"
    with open(summary, "w") as f:
        json.dump(out, f, indent=2)

    # ── Round 1 results table ───────────────────────────────────────────────
    print()
    if baseline_origin != "skipped":
        rb = in_run.get("ring", {})
        sb = in_run.get("collnet_sharp", {})
        rmed = rb.get("median_busbw_gbs", float("nan"))
        smed = sb.get("median_busbw_gbs", float("nan"))
        print(f"───────────────── Round 1 results "
              f"(baseline {baseline_origin}: ring~{rmed:.1f}, sharp~{smed:.1f}) ─────────────────")
    else:
        print("───────────────── Round 1 results ─────────────────")
    print(f"{'grp':<4} {'ring':>9} {'sharp':>9}  status              nodes")
    print(f"{'-'*4} {'-'*9} {'-'*9}  {'-'*18}  {'-'*40}")
    for g in out["groups"]:
        ring = g["algos"]["ring"]["busbw_gbs"]
        sharp = g["algos"]["collnet_sharp"]["busbw_gbs"]
        rs = f"{ring:9.1f}" if ring is not None else f"{'--':>9}"
        ss = f"{sharp:9.1f}" if sharp is not None else f"{'--':>9}"
        ring_status = g["algos"]["ring"]["status"]
        sharp_status = g["algos"]["collnet_sharp"]["status"]
        if ring_status == "completed" and sharp_status == "completed":
            perf = g.get("perf_label", "healthy")
            status_label = "ok" if perf == "healthy" else perf.upper()
        elif ring_status == "timed_out" or sharp_status == "timed_out":
            cls = (g["algos"]["ring"].get("failure_class")
                   or g["algos"]["collnet_sharp"].get("failure_class")
                   or "unknown")
            status_label = f"FAIL:{cls}"
        else:
            status_label = f"{ring_status}/{sharp_status}"
        print(f"  {g['gid']:<2} {rs} {ss}  {status_label:<18}  {','.join(g['nodes'])}")

    # ── Stragglers identified this round ────────────────────────────────────
    print()
    print("───────────────── Stragglers identified — Round 1 ─────────────────")
    if not out["node_state_failed"]:
        print("  (none localized to a specific node)")
    else:
        for i, nf in enumerate(out["node_state_failed"], 1):
            cls = nf["class"]
            tag = f"class={cls}"
            if nf.get("missing_lib"):
                tag += f"  lib={nf['missing_lib']}"
            print(f"  [{i}] {nf['node']}  {tag}")
            evidence = None
            for g in out["groups"]:
                if nf["node"] not in g["nodes"]:
                    continue
                for algo in ALGOS:
                    a = g["algos"][algo]
                    if a.get("suspect_node") == nf["node"] and a.get("evidence"):
                        evidence = a["evidence"]
                        break
                if evidence:
                    break
            if evidence:
                ev = evidence[:140] + ("…" if len(evidence) > 140 else "")
                print(f"      evidence: {ev}")
            print(f"      why:      {FAILURE_JUSTIFICATIONS.get(cls, '(no justification)')}")
            print(f"      log:      {nf['evidence_outfile']}")

    # Group-level findings without a localized suspect (e.g. sharp_no_data).
    unlocalized = [bf for bf in out["bootstrap_failed_groups"]
                   if not bf.get("suspect_nodes")]
    if unlocalized:
        print()
        print("───────────────── Groups needing investigation — Round 1 ─────────────────")
        for bf in unlocalized:
            classes = "/".join(bf["failure_classes"]) or "unknown"
            algos_str = ",".join(bf["failed_algos"])
            print(f"  grp{bf['gid']}  nodes={','.join(bf['nodes'])}")
            print(f"      class={classes}  failed_algos={algos_str}")
            print(f"      why:  {FAILURE_JUSTIFICATIONS.get(bf['failure_classes'][0] if bf['failure_classes'] else 'unknown_hang', '(no justification)')}")

    if out["bootstrap_failed_groups"]:
        groups_with_localized = [bf for bf in out["bootstrap_failed_groups"]
                                 if bf.get("suspect_nodes")]
        if groups_with_localized:
            print()
            print(f"  Note: in {len(groups_with_localized)} bootstrap-failed group(s) "
                  "the localized suspect was identified; the other 3 nodes are "
                  "inconclusive (may be healthy but couldn't be tested).")

    # ── Performance stragglers ──────────────────────────────────────────────
    print()
    print("───────────────── Performance stragglers — Round 1 ─────────────────")
    if not out["perf_stragglers"]:
        print("  (none)")
    else:
        # Sort severe first, then by worst z across algos.
        def _worst_z(entry):
            zs = [s["z"] for s in entry["scores"].values() if s["z"] is not None]
            return min(zs) if zs else 0
        ordered = sorted(out["perf_stragglers"],
                         key=lambda e: (e["label"] != "severe", _worst_z(e)))
        for i, entry in enumerate(ordered, 1):
            label = entry["label"]
            gid = entry["gid"]
            scores = entry["scores"]
            ring_s = scores["ring"]
            sharp_s = scores["collnet_sharp"]

            def _fmt(s):
                if s["z"] is None:
                    return "no_data"
                return f"{s['delta_pct']:+.1f}% (z={s['z']:+.1f}, {s['label']})"
            print(f"  [{i}] [{label.upper():<7}] grp{gid}  "
                  f"ring={_fmt(ring_s)}   sharp={_fmt(sharp_s)}")
            print(f"        nodes: {','.join(entry['nodes'])}")
            tag = "ring+sharp" if (ring_s["label"] != "healthy"
                                   and sharp_s["label"] != "healthy") else \
                  "ring-only"  if ring_s["label"] != "healthy" else \
                  "sharp-only"
            print(f"        signal: {tag}  ({PERF_JUSTIFICATIONS.get(label, '').format(Z_SUSPECT=int(Z_SUSPECT), Z_SEVERE=int(Z_SEVERE))})")

    # ── Round 2 results ─────────────────────────────────────────────────────
    loc = out.get("localization")
    if loc and loc.get("waves"):
        print()
        print("───────────────── Round 2 results — Localization ─────────────────")
        for w_idx, wave in enumerate(loc["waves"], 1):
            print(f"  Wave {w_idx} ({wave['algo']}):")
            print(f"    {'grp':<4} {'candidate':<32} {'busbw':>8}  {'Δ%':>7}  {'z':>6}  verdict")
            for t in wave["tests"]:
                bw = t["busbw_gbs"]
                bw_s = f"{bw:8.1f}" if bw is not None else f"{'--':>8}"
                d = t.get("delta_pct")
                ds = f"{d:+6.1f}%" if d is not None else f"{'--':>7}"
                z = t.get("z")
                zs = f"{z:+6.1f}" if z is not None else f"{'--':>6}"
                print(f"    {t['gid']:<4} {t['candidate']:<32} {bw_s}  {ds}  {zs}  {t['verdict']}")

        print()
        print("───────────────── Localized bad nodes — Round 2 ─────────────────")
        if not loc.get("localized_bad_nodes"):
            print("  (none — all candidates tested healthy when paired with healthy peers)")
            # Edge case: severe groups with no localized node = inter-link or transient.
            severe_unlocalized = [g["gid"] for g in out["groups"]
                                  if g.get("perf_label") == "severe"]
            if severe_unlocalized:
                print(f"  Severe groups remained unlocalized: {severe_unlocalized}")
                print("  → likely inter-link weakness or transient. Iteration 4 cross-pair check.")
        else:
            for node in loc["localized_bad_nodes"]:
                ev = loc.get("evidence_per_node", {}).get(node, [])
                tags = ", ".join(f"{e['algo']} {e['delta_pct']:+.1f}% z={e['z']:+.1f}"
                                 for e in ev)
                print(f"  ★ {node}  ({tags})")

    # ── Round 3 results — Cross-pair ────────────────────────────────────────
    cp = out.get("cross_pair")
    if cp and cp.get("pairs"):
        print()
        med = cp.get("median_8n_ring_gbs")
        sdv = cp.get("stddev_8n_ring_gbs")
        if med is not None:
            print(f"───────────────── Round 3 results — Cross-pair "
                  f"(8n ring median={med:.1f}, stddev={sdv:.2f}) ─────────────────")
        else:
            print("───────────────── Round 3 results — Cross-pair ─────────────────")
        print(f"  {'pair':<5} {'A':<4} {'B':<4} {'busbw':>8}  {'Δ%':>7}  {'z':>6}  label")
        for p in cp["pairs"]:
            bw = p.get("busbw_gbs")
            bw_s = f"{bw:8.1f}" if bw is not None else f"{'--':>8}"
            d = p.get("delta_pct")
            ds = f"{d:+6.1f}%" if d is not None else f"{'--':>7}"
            z = p.get("z")
            zs = f"{z:+6.1f}" if z is not None else f"{'--':>6}"
            label = p.get("label", p.get("status", "?"))
            print(f"  {p['pair_idx']:<5} {p['gA_gid']:<4} {p['gB_gid']:<4} "
                  f"{bw_s}  {ds}  {zs}  {label}")

        print()
        print("───────────────── Inter-link suspects — Round 3 ─────────────────")
        sus = cp.get("inter_link_suspects", [])
        if not sus:
            print("  (none — all 8n cross-pairs match in-run median)")
        else:
            for s in sus:
                print(f"  ★ pair{s['pair_idx']}: grp{s['gA_gid']}+grp{s['gB_gid']}  "
                      f"8n={s['busbw_gbs']:.1f}  ({s['delta_pct']:+.1f}%, z={s['z']:+.1f}, {s['label']})")
                print(f"      A nodes: {','.join(s['nodes_A'])}")
                print(f"      B nodes: {','.join(s['nodes_B'])}")
                print(f"      why:     8n is far below the in-run median despite "
                      "both 4n halves being healthy at Phase 1. Likely a degraded "
                      "spine/uplink between A and B. Avoid collectives that span "
                      "this pair until fabric is investigated.")

    # ── Round 4 results — Confirmation retest ───────────────────────────────
    cf = out.get("confirm")
    if cf and (cf.get("retest_waves") or cf.get("skipped_strong")):
        print()
        print("───────────────── Round 4 results — Confirmation retest ─────────────────")
        if cf.get("skipped_strong"):
            print(f"  Strong signals kept as-is (z <= {cf['retest_z_threshold']}, no retest):")
            for s in cf["skipped_strong"]:
                print(f"    • {s['candidate']} {s['algo']:14s}  z={s['z']:+.1f}  "
                      f"({s['busbw']:.1f} GB/s)")
        for w_idx, wave in enumerate(cf.get("retest_waves", []), 1):
            print()
            print(f"  Wave {w_idx} ({wave['algo']}) — retest with new H-triple:")
            print(f"    {'cand':<32} {'orig z':>8} {'retest z':>9}  verdict")
            for t in wave["tests"]:
                orig_z = t.get("original_z")
                ozs = f"{orig_z:+8.1f}" if orig_z is not None else f"{'--':>8}"
                z = t.get("z")
                zs = f"{z:+9.1f}" if z is not None else f"{'--':>9}"
                v = t.get("final_verdict", "?")
                print(f"    {t['candidate']:<32} {ozs} {zs}  {v}")

        # Summarize per-node confirmed verdicts.
        confirmed: dict[str, dict[str, str]] = {}  # node -> {algo: verdict}
        # Strong signals are confirmed without retest.
        for s in cf.get("skipped_strong", []):
            confirmed.setdefault(s["candidate"], {})[s["algo"]] = "confirmed_bad"
        for wave in cf.get("retest_waves", []):
            for t in wave["tests"]:
                confirmed.setdefault(t["candidate"], {})[t["algo"]] = t["final_verdict"]

        if confirmed:
            print()
            print("───────────────── Final node verdicts ─────────────────")
            for node in sorted(confirmed):
                per_algo = confirmed[node]
                # Aggregate: any "confirmed_bad" wins; "topology_sensitive" if mixed; else healthy.
                vals = list(per_algo.values())
                if "confirmed_bad" in vals and "topology_sensitive" not in vals:
                    overall = "CONFIRMED_BAD"
                elif "topology_sensitive" in vals and "confirmed_bad" not in vals:
                    overall = "TOPOLOGY_SENSITIVE"
                elif "confirmed_bad" in vals and "topology_sensitive" in vals:
                    overall = "CONFIRMED_BAD (one algo); TOPOLOGY_SENSITIVE (other)"
                elif "retest_failed" in vals:
                    overall = "RETEST_FAILED"
                else:
                    overall = ", ".join(f"{a}={v}" for a, v in per_algo.items())
                tags = ", ".join(f"{a}:{v}" for a, v in per_algo.items())
                print(f"  ★ {node}  → {overall}")
                print(f"      per-algo: {tags}")

    # ── Round 5 results — IB sweep ─────────────────────────────────────────
    ib = out.get("ib_sweep")
    if ib and ib.get("per_hca_stats"):
        print()
        print(f"───────────────── Round 5 results — IB sweep "
              f"(K={ib.get('k', 1)} × {ib['n_pairs_per_wave']} pairs / "
              f"{ib['n_nodes_tested']} nodes) ─────────────────")
        print(f"  {'HCA':<10} {'median':>10} {'stddev':>9} {'min':>10} {'max':>10}  n")
        for hca in IB_HCAS:
            s = ib["per_hca_stats"].get(hca)
            if s is None:
                print(f"  {hca:<10}  (insufficient data)")
                continue
            print(f"  {hca:<10} {s['median_mibs']:>10.0f} {s['stddev_mibs']:>9.1f} "
                  f"{s['min_mibs']:>10.0f} {s['max_mibs']:>10.0f}  {s['n_samples']}")

        print()
        print("───────────────── Bad HCAs — Round 5 ─────────────────")
        bads = ib.get("bad_hcas", [])
        if not bads:
            print("  (none — all HCAs healthy on every tested node)")
        else:
            order = {
                "port_down": 0,
                "node_intrinsic_severe": 1,
                "node_intrinsic_suspect": 2,
                "link_specific": 3,
                "test_fail": 4,
            }
            for b in sorted(bads, key=lambda x: (order.get(x["status"], 9),
                                                  x["node"], x["hca"])):
                st = b["status"]
                node, hca = b["node"], b["hca"]
                if st == "port_down":
                    print(f"  ★ {node:32s} {hca:8s} PORT_DOWN")
                    print(f"      → drain + check IB cable / SFP / switch port for {hca}")
                elif st == "node_intrinsic_severe":
                    print(f"  ★ {node:32s} {hca:8s} NODE_BAD (severe)  "
                          f"{b['median_mibs_for_node']:.0f} MiB/s "
                          f"({b['pct_of_cluster_median']:.1f}% of cluster median)")
                    print(f"      → slow in all {b['n_trials']} trials with peers "
                          f"{b['peers']}; the HCA itself is degraded")
                elif st == "node_intrinsic_suspect":
                    print(f"  ⚠ {node:32s} {hca:8s} NODE_BAD (suspect)  "
                          f"{b['median_mibs_for_node']:.0f} MiB/s "
                          f"({b['pct_of_cluster_median']:.1f}% of cluster median)")
                    print(f"      → slow in all {b['n_trials']} trials with peers "
                          f"{b['peers']}; mild but consistent")
                elif st == "link_specific":
                    print(f"  ⚠ {node:32s} {hca:8s} LINK_SPECIFIC")
                    print(f"      → slow with peer {b['slow_peer']} "
                          f"({b['slow_pct']:.1f}%) but healthy with {b['fast_peer']} "
                          f"({b['fast_pct']:.1f}%)")
                    print(f"      → likely degraded path between {node} and "
                          f"{b['slow_peer']} (switch port / cable), not the HCA itself")
                elif st == "test_fail":
                    print(f"  ⚠ {node:32s} {hca:8s} TEST_FAIL")

    print(f"\nresults: {summary}")


if __name__ == "__main__":
    main()
