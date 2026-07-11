#!/usr/bin/env python3
"""find_stragglers.py — flapping-airplanes cluster (slinky partition).

Round 1: disjoint 2-node sweep, ring all_reduce at 8G only.
         Watchdog: 60s per-job timeout — no "Avg bus bandwidth" = bad node in group.
Round 2: localize — each candidate paired with 1 healthy H-node (parallel).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Cluster constants (flapping-airplanes / slinky) ─────────────────────────
GPUS_PER_NODE   = 8
PARTITION       = "slinky"
GROUP_SIZE      = 2
TIME_LIMIT      = "00:05:00"
BINARY_PATH     = os.path.expanduser("~/together-nccl-tests/build")
NCCL_LIB        = "/usr/lib/x86_64-linux-gnu"
HPCX_INIT       = "/opt/hpcx/hpcx-init.sh"
RESULTS_ROOT    = Path("/data/home/johnson/nccl-results/flapping-airplanes/find-stragglers")
BASELINE_DIR    = Path.home() / "together-nccl-tests" / "baselines"
CLUSTER_NAME    = "flapping-airplanes"

# Performance z-score thresholds
Z_SUSPECT          = 2.0
Z_SEVERE           = 5.0
STDDEV_FLOOR_FRAC  = 0.005

ALGO_ENV = {
    "ring": {"algo": "Ring", "nvls": "0", "collnet": "0"},
}

# Single 8G message — healthy test finishes in ~25-35s total wall time.
DEFAULT_SIZES = ("8G", "8G", 2)
WARMUP, ITERS = 2, 5

POLL_INTERVAL_S  = 30    # watchdog poll cadence
WATCHDOG_BUDGET_S = 1200  # global ceiling
JOB_TIMEOUT_S    = 60    # per-job: no output after 60s → bad node

AVG_BUSBW_RE        = re.compile(r"Avg bus bandwidth\s*:\s*([\d.]+)")
LIB_MISSING_RE      = re.compile(
    r"error while loading shared libraries: (\S+?): cannot open shared object file"
)
SRUN_EXIT_127_RE    = re.compile(r"srun: error: (\S+): tasks .*Exited with exit code 127")
PMIX_TIMEOUT_RE     = re.compile(r"collective timeout seq=")
PMIX_WAIT_CONTRIB_RE = re.compile(r"wait contrib: (slinky-\d+)")


# ── Node discovery ────────────────────────────────────────────────────────────

def discover_idle(partition: str, exclude: set[str]) -> list[str]:
    out = subprocess.check_output(
        ["sinfo", "-N", "-p", partition, "-h", "-o", "%N %t"], text=True
    )
    hosts = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "idle" and parts[0] not in exclude:
            hosts.append(parts[0])
    return sorted(set(hosts))


# ── sbatch builder ────────────────────────────────────────────────────────────

def build_sbatch(job_name: str, nodes: list[str], algo: str, outfile: str,
                 sizes: tuple) -> str:
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

source {HPCX_INIT} && hpcx_load
export LD_LIBRARY_PATH={NCCL_LIB}:${{LD_LIBRARY_PATH:-}}
ulimit -l unlimited 2>/dev/null || true
export UCX_TLS=self,sm,cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=eth0
export OMPI_MCA_btl_tcp_if_include=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=300
export CUDA_DEVICE_MAX_CONNECTIONS=32
{nccl_algo}
export NCCL_NVLS_ENABLE={env["nvls"]}
export NCCL_COLLNET_ENABLE={env["collnet"]}
export NCCL_DEBUG=WARN

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


# ── Watchdog ──────────────────────────────────────────────────────────────────

def watchdog(jobs: dict[int, str], poll_s: int = POLL_INTERVAL_S,
             budget_s: int = WATCHDOG_BUDGET_S,
             job_timeout_s: int = JOB_TIMEOUT_S) -> dict[int, str]:
    """Poll outfiles every poll_s seconds.

    - "Avg bus bandwidth" line seen  → scancel (post-srun hang avoidance) → "completed"
    - Age exceeds job_timeout_s with no data → scancel → "timed_out"
    - Job vanished from squeue with no data  → "vanished"
    """
    submitted_at  = {jid: time.time() for jid in jobs}
    statuses: dict[int, str] = {}
    pending       = dict(jobs)
    start         = time.time()
    print(f"[watchdog] {len(pending)} jobs, poll={poll_s}s, "
          f"per-job timeout={job_timeout_s}s, ceiling={budget_s}s", flush=True)
    while pending:
        if time.time() - start > budget_s:
            print(f"[watchdog] {budget_s}s ceiling — cancelling {len(pending)} remaining",
                  flush=True)
            for jid in pending:
                subprocess.run(["scancel", str(jid)], capture_output=True)
                statuses[jid] = "timed_out"
            return statuses
        time.sleep(poll_s)
        for jid, outfile in list(pending.items()):
            status = None
            if os.path.exists(outfile):
                try:
                    with open(outfile) as fh:
                        content = fh.read()
                    if AVG_BUSBW_RE.search(content):
                        subprocess.run(["scancel", str(jid)], capture_output=True)
                        status = "completed"
                except OSError:
                    pass
            if status is None:
                age = time.time() - submitted_at[jid]
                if age > job_timeout_s:
                    print(f"[watchdog] job {jid} — {job_timeout_s}s without output → timed_out",
                          flush=True)
                    subprocess.run(["scancel", str(jid)], capture_output=True)
                    status = "timed_out"
                else:
                    r = subprocess.run(["squeue", "-j", str(jid), "-h"],
                                       capture_output=True, text=True)
                    if not r.stdout.strip():
                        status = "vanished"
            if status:
                statuses[jid] = status
                pending.pop(jid, None)
        n_done = len(jobs) - len(pending)
        print(f"[watchdog] t={int(time.time()-start)}s  "
              f"done={n_done}/{len(jobs)}  pending={len(pending)}", flush=True)
    return statuses


def wait_for_cg_drain(partition: str, ceiling_s: int = 120) -> None:
    """Wait until no jobs on partition are in CG state.

    Hung NCCL/IB processes block in kernel space and can take 30-90s to die
    after scancel (Slurm must escalate SIGTERM→SIGKILL then wait for cleanup).
    Nodes stay unavailable for new jobs during this window; skipping the wait
    causes the next discover_idle() to see fewer nodes than expected.
    """
    start = time.time()
    while True:
        r = subprocess.run(
            ["squeue", "-p", partition, "-h", "-t", "CG", "-o", "%i"],
            capture_output=True, text=True,
        )
        cg_jobs = r.stdout.split()
        if not cg_jobs:
            return
        elapsed = int(time.time() - start)
        if elapsed >= ceiling_s:
            print(f"[cg-drain] {ceiling_s}s ceiling hit with {len(cg_jobs)} CG jobs still lingering — proceeding anyway",
                  flush=True)
            return
        print(f"[cg-drain] {len(cg_jobs)} CG job(s) still completing (t={elapsed}s) — waiting...",
              flush=True)
        time.sleep(10)


# ── Output parsers ────────────────────────────────────────────────────────────

def parse_busbw(outfile: str) -> float | None:
    if not os.path.exists(outfile):
        return None
    with open(outfile) as fh:
        m = AVG_BUSBW_RE.search(fh.read())
    return float(m.group(1)) if m else None


def classify_failure(outfile: str) -> dict:
    """Classify why a job produced no bandwidth line."""
    if not os.path.exists(outfile):
        return {"class": "no_output", "suspect_node": None, "evidence": None}
    with open(outfile) as fh:
        content = fh.read()

    libs = LIB_MISSING_RE.findall(content)
    if libs:
        m = SRUN_EXIT_127_RE.search(content)
        evidence = next(
            (l.strip() for l in content.splitlines() if "Exited with exit code 127" in l),
            None,
        )
        return {"class": "missing_lib", "suspect_node": m.group(1) if m else None,
                "evidence": evidence, "missing_lib": libs[0]}

    if PMIX_TIMEOUT_RE.search(content):
        counts = Counter(PMIX_WAIT_CONTRIB_RE.findall(content))
        suspect = counts.most_common(1)[0][0] if counts else None
        evidence = next(
            (l.strip() for l in content.splitlines() if suspect and f"wait contrib: {suspect}" in l),
            None,
        )
        return {"class": "pmix_wedge", "suspect_node": suspect, "evidence": evidence}

    return {"class": "unknown_hang", "suspect_node": None, "evidence": None}


# ── Performance classification ────────────────────────────────────────────────

def compute_baseline(busbws: list[float]) -> dict | None:
    valid = sorted(b for b in busbws if b is not None)
    if len(valid) < 3:
        return None
    q = len(valid) // 4
    trimmed = valid[q: len(valid) - q] if q else valid
    return {
        "median_busbw_gbs": statistics.median(trimmed),
        "stddev_busbw_gbs": statistics.stdev(trimmed) if len(trimmed) > 1 else 0.5,
        "n_samples": len(trimmed),
    }


def classify_perf(busbw: float | None, baseline: dict | None) -> dict:
    if busbw is None or baseline is None:
        return {"label": "no_data", "z": None, "delta_pct": None}
    median = baseline["median_busbw_gbs"]
    stddev = max(baseline["stddev_busbw_gbs"], median * STDDEV_FLOOR_FRAC)
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
    """Nodes from groups that completed with healthy bandwidth."""
    pool = []
    for g in groups:
        if g.get("perf_label") == "healthy":
            pool.extend(g["nodes"])
    return pool


# ── Round 2: 2-node localization ──────────────────────────────────────────────

def phase2_localize(bad_groups: list[dict], h_pool: list[str],
                    results_dir: Path, sizes: tuple,
                    in_run_baseline: dict | None, args) -> dict:
    """For each bad 2-node group (A, B), test each candidate with a healthy H-node.

    Uses distinct H-nodes per candidate so all tests run in parallel.
    Verdict: timed_out or no busbw → candidate_bad; otherwise → candidate_healthy.
    """
    if not bad_groups:
        return {"round": 2, "tests": [], "bad_nodes": [],
                "note": "no bad groups from Round 1"}
    if not h_pool:
        return {"round": 2, "tests": [], "bad_nodes": [],
                "note": "no healthy H-pool available for localization"}

    # Assign H-nodes round-robin; different candidates in the same Slurm
    # wave must use distinct H-nodes to run in parallel.
    submitted = []
    h_idx = 0
    for g in bad_groups:
        for candidate in g["nodes"]:
            h_node = h_pool[h_idx % len(h_pool)]
            h_idx += 1
            short = candidate.split("-")[-1]
            outfile = str(results_dir / f"phase2_g{g['gid']}_c{short}_ring.out")
            sb = build_sbatch(
                f"strag-p2-g{g['gid']}-c{short}",
                [candidate, h_node], "ring", outfile, sizes,
            )
            jid = submit_one(sb)
            print(f"  [submit] grp{g['gid']} cand={candidate} H={h_node} -> job {jid}",
                  flush=True)
            submitted.append({
                "gid": g["gid"], "candidate": candidate, "h_node": h_node,
                "jobid": jid, "outfile": outfile,
            })

    print(flush=True)
    statuses = watchdog(
        {s["jobid"]: s["outfile"] for s in submitted},
        job_timeout_s=args.job_timeout_s,
    )

    bad_nodes = []
    for s in submitted:
        s["status"] = statuses.get(s["jobid"], "unknown")
        bw = parse_busbw(s["outfile"])
        s["busbw_gbs"] = bw
        score = classify_perf(bw, in_run_baseline)
        s["z"] = score["z"]
        s["delta_pct"] = score["delta_pct"]
        s["perf_label"] = score["label"]
        if s["status"] == "timed_out" or bw is None:
            s["verdict"] = "candidate_bad"
            s["fail_classification"] = classify_failure(s["outfile"])
            bad_nodes.append(s["candidate"])
        elif score["label"] in ("suspect", "severe"):
            s["verdict"] = "candidate_bad_perf"
            bad_nodes.append(s["candidate"])
        else:
            s["verdict"] = "candidate_healthy"

    return {
        "round": 2,
        "tests": submitted,
        "bad_nodes": sorted(set(bad_nodes)),
    }


# ── Baseline persistence ──────────────────────────────────────────────────────

def load_or_seed_baseline(in_run: dict, force_rebaseline: bool,
                          source_label: str) -> tuple[dict, str]:
    path = BASELINE_DIR / CLUSTER_NAME / "healthy_2n_ring.json"
    if path.exists() and not force_rebaseline:
        with open(path) as f:
            return json.load(f), "loaded"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "cluster": CLUSTER_NAME,
        "group_size": GROUP_SIZE,  # baseline always uses default group size
        "sizes": f"{DEFAULT_SIZES[0]}-{DEFAULT_SIZES[1]} factor={DEFAULT_SIZES[2]}",
        "iters": f"{WARMUP}w+{ITERS}m",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "z_thresholds": {"suspect": Z_SUSPECT, "severe": Z_SEVERE},
        "stddev_floor_frac": STDDEV_FLOOR_FRAC,
        "ring": in_run,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    origin = "rebaselined" if force_rebaseline else "seeded"
    return doc, origin


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--partition",     default=PARTITION)
    ap.add_argument("--exclude",       default="",
                    help="Slurm hostlist to exclude from idle pool")
    ap.add_argument("--max-nodes",     type=int, default=None,
                    help="cap idle nodes (debug)")
    ap.add_argument("--results-dir",   default=None)
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--job-timeout-s", type=int, default=JOB_TIMEOUT_S,
                    help="per-job watchdog timeout in seconds (default %(default)s)")
    ap.add_argument("--group-size",   type=int, default=GROUP_SIZE,
                    help="nodes per group (default %(default)s)")
    ap.add_argument("--rebaseline",    action="store_true",
                    help="overwrite saved baseline from this run")
    ap.add_argument("--skip-localize", action="store_true",
                    help="skip Round 2 localization")
    args = ap.parse_args()

    excl: set[str] = set()
    if args.exclude:
        excl = set(subprocess.check_output(
            ["scontrol", "show", "hostnames", args.exclude], text=True
        ).split())

    if not args.dry_run:
        wait_for_cg_drain(args.partition)

    hosts_all = discover_idle(args.partition, excl)
    if args.max_nodes:
        hosts_all = hosts_all[: args.max_nodes]

    n_groups = len(hosts_all) // args.group_size
    if n_groups == 0:
        print(f"ERROR: only {len(hosts_all)} idle nodes (need >= {args.group_size})",
              file=sys.stderr)
        sys.exit(1)

    used           = n_groups * args.group_size
    leftover_hosts = hosts_all[used:]
    hosts          = hosts_all[:used]
    groups         = [hosts[i * args.group_size:(i + 1) * args.group_size]
                      for i in range(n_groups)]

    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT / ts
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  find_stragglers.py — flapping-airplanes / slinky")
    print("  Round 1: 2-node ring @ 8G  |  60s timeout per job")
    print("=" * 66)
    print(f"  partition:   {args.partition}")
    print(f"  idle nodes:  {len(hosts_all)}  ({used} used, {len(leftover_hosts)} leftover)")
    print(f"  groups:      {n_groups} groups of {args.group_size}")
    print(f"  size:        {DEFAULT_SIZES[0]}  iters: {WARMUP}w + {ITERS}m")
    print(f"  timeout:     {args.job_timeout_s}s per job")
    print(f"  results:     {results_dir}")
    print(f"  dry-run:     {args.dry_run}")
    if leftover_hosts:
        print(f"  leftover:    {' '.join(leftover_hosts)}")
    print()

    # ── Round 1: submit all 2-node ring jobs ──────────────────────────────────
    submitted: dict[int, tuple[int | None, str]] = {}  # gidx -> (jobid, outfile)
    for gidx, gnodes in enumerate(groups):
        gid     = f"{gidx + 1:02d}"
        outfile = str(results_dir / f"phase1_grp{gid}_ring.out")
        sb      = build_sbatch(f"strag-p1-g{gid}", gnodes, "ring", outfile,
                               DEFAULT_SIZES)
        if args.dry_run:
            print(f"[dry-run] grp{gid}  nodes={','.join(gnodes)}")
            submitted[gidx] = (None, outfile)
        else:
            jid = submit_one(sb)
            print(f"[submit ] grp{gid}  nodes={','.join(gnodes)}  -> job {jid}")
            submitted[gidx] = (jid, outfile)

    if args.dry_run:
        print("\nDry run complete. No jobs submitted.")
        return

    print()
    job_map  = {jid: of for jid, of in submitted.values() if jid is not None}
    statuses = watchdog(job_map, job_timeout_s=args.job_timeout_s)

    # ── Parse Round 1 results ─────────────────────────────────────────────────
    out = {
        "timestamp_utc": ts,
        "partition": args.partition,
        "group_size": args.group_size,
        "n_idle": len(hosts_all),
        "leftover_hosts": leftover_hosts,
        "results_dir": str(results_dir),
        "groups": [],
        "bad_groups": [],
    }

    all_bws: list[float] = []
    for gidx, gnodes in enumerate(groups):
        gid         = f"{gidx + 1:02d}"
        jid, outfile = submitted[gidx]
        status      = statuses.get(jid, "unknown") if jid is not None else "dry-run"
        bw          = parse_busbw(outfile)
        entry = {
            "gid": gid, "nodes": gnodes,
            "jobid": jid, "status": status,
            "busbw_gbs": bw, "outfile": outfile,
        }
        if status == "timed_out":
            entry["fail_classification"] = classify_failure(outfile)
        if bw is not None:
            all_bws.append(bw)
        out["groups"].append(entry)

    # Baseline + z-score classify
    in_run_baseline = compute_baseline(all_bws)
    baseline_doc, baseline_origin = (None, "skipped")
    if in_run_baseline:
        baseline_doc, baseline_origin = load_or_seed_baseline(
            in_run_baseline, args.rebaseline,
            source_label=f"trimmed quartile from {results_dir.name}",
        )
    out["baseline_origin"] = baseline_origin
    out["baseline_in_run"] = in_run_baseline

    for g in out["groups"]:
        score = classify_perf(g["busbw_gbs"], in_run_baseline)
        g["perf_label"] = score["label"]
        g["perf_z"]     = score["z"]
        g["perf_delta_pct"] = score["delta_pct"]
        is_bad = (g["status"] == "timed_out"
                  or g["busbw_gbs"] is None
                  or score["label"] in ("suspect", "severe"))
        if is_bad:
            out["bad_groups"].append(g)

    # ── Print Round 1 table ───────────────────────────────────────────────────
    print()
    med_s = (f"  median={in_run_baseline['median_busbw_gbs']:.1f} GB/s"
             if in_run_baseline else "")
    print(f"─── Round 1 results (ring @ 8G{med_s}) {'─' * 20}")
    print(f"{'grp':<4}  {'busbw':>9}  {'Δ%':>7}  {'z':>6}  status        nodes")
    print(f"{'─'*4}  {'─'*9}  {'─'*7}  {'─'*6}  {'─'*13}  {'─'*30}")
    for g in out["groups"]:
        bw   = g["busbw_gbs"]
        bw_s = f"{bw:9.1f}" if bw is not None else f"{'--':>9}"
        d    = g.get("perf_delta_pct")
        ds   = f"{d:+6.1f}%" if d is not None else f"{'--':>7}"
        z    = g.get("perf_z")
        zs   = f"{z:+6.1f}" if z is not None else f"{'--':>6}"
        if g["status"] == "timed_out":
            cls   = g.get("fail_classification", {}).get("class", "hang")
            slabel = f"TIMEOUT:{cls}"
        elif bw is None:
            slabel = "no_output"
        else:
            pl = g.get("perf_label", "healthy")
            slabel = "ok" if pl == "healthy" else pl.upper()
        print(f"  {g['gid']:<2}  {bw_s}  {ds}  {zs}  {slabel:<13}  {','.join(g['nodes'])}")

    # ── Round 1 summary ───────────────────────────────────────────────────────
    n_ok  = sum(1 for g in out["groups"] if g not in out["bad_groups"])
    n_bad = len(out["bad_groups"])
    print()
    print(f"Round 1:  {n_ok} healthy  |  {n_bad} bad groups")
    if out["bad_groups"]:
        print("  Bad groups:")
        for g in out["bad_groups"]:
            cls = (g.get("fail_classification", {}).get("class", "")
                   if g["status"] == "timed_out" else g.get("perf_label", ""))
            suspect = (g.get("fail_classification", {}).get("suspect_node")
                       if g["status"] == "timed_out" else None)
            tag = f"  suspect={suspect}" if suspect else ""
            print(f"    grp{g['gid']}  nodes={','.join(g['nodes'])}  class={cls}{tag}")

    # ── Round 2: localization ─────────────────────────────────────────────────
    if not args.skip_localize and out["bad_groups"]:
        h_pool = select_h_pool(out["groups"])
        print()
        print("=" * 66)
        print("  Round 2: localize — candidate + healthy H-node (parallel)")
        print("=" * 66)
        print(f"  H-pool: {len(h_pool)} healthy nodes: {', '.join(h_pool)}")
        print()
        loc = phase2_localize(
            out["bad_groups"], h_pool, results_dir,
            DEFAULT_SIZES, in_run_baseline, args,
        )
        out["localization"] = loc

        print()
        print(f"─── Round 2 results {'─' * 40}")
        print(f"  {'grp':<4} {'candidate':<20} {'busbw':>9}  {'Δ%':>7}  {'z':>6}  verdict")
        for t in loc["tests"]:
            bw   = t["busbw_gbs"]
            bw_s = f"{bw:9.1f}" if bw is not None else f"{'--':>9}"
            d    = t.get("delta_pct")
            ds   = f"{d:+6.1f}%" if d is not None else f"{'--':>7}"
            z    = t.get("z")
            zs   = f"{z:+6.1f}" if z is not None else f"{'--':>6}"
            print(f"  {t['gid']:<4} {t['candidate']:<20} {bw_s}  {ds}  {zs}  {t['verdict']}")

        print()
        if loc["bad_nodes"]:
            print(f"  ★ Confirmed bad nodes ({len(loc['bad_nodes'])}):")
            for node in loc["bad_nodes"]:
                print(f"      {node}")
        else:
            print("  (no nodes confirmed bad — may be inter-link or transient)")
    elif args.skip_localize:
        out["localization"] = {"skipped": "by --skip-localize"}

    # ── Save results ──────────────────────────────────────────────────────────
    summary = results_dir / "results.json"
    with open(summary, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults: {summary}")


if __name__ == "__main__":
    main()
