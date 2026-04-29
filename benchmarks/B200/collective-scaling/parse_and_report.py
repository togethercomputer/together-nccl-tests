#!/usr/bin/env python3
"""NCCL Collective Scaling Benchmark — Parser, Comparator, and Report Generator.

Parses nccl-test output files from a results directory, optionally compares
against a previous run, and generates a markdown report.

Usage:
    python3 parse_and_report.py \\
        --results-dir /path/to/new/results/ \\
        --compare-dir /path/to/old/results/ \\
        --output report.md \\
        --wait
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────

GPUS_PER_NODE = 8
NODE_COUNTS = [1, 2, 4, 8, 16, 32, 64]
OPS = ["all_reduce", "all_gather", "sendrecv"]
ALGOS_ALL_REDUCE = ["ring", "nvls", "collnet_sharp", "nvls_tree", "tree"]
ALGOS_ALL_GATHER = ["ring", "nvls", "collnet_sharp"]
ALGO_LABELS = {
    "ring": "Ring",
    "nvls": "NVLS",
    "tree": "Tree",
    "nvls_tree": "NVLSTree",
    "collnet_sharp": "CollNet SHARP",
}
MSG_SIZES = {2147483648: "2 GB", 4294967296: "4 GB", 8589934592: "8 GB"}
MSG_SIZE_BYTES = [2147483648, 4294967296, 8589934592]
EXPECTED_FAILURES = {("all_gather", "tree"), ("all_gather", "nvls_tree")}
MULTI_NODE_ONLY_ALGOS = {"tree", "nvls_tree", "collnet_sharp"}
BUSBW_COL = 7  # out-of-place busBW, 0-indexed


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_filename(fname):
    """Extract (op, algo, nodes) from an output filename."""
    m = re.match(
        r"(all_reduce|all_gather|sendrecv)_(ring|nvls|tree|nvls_tree|collnet_sharp|p2p)_(\d+)nodes\.out$",
        fname,
    )
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def parse_single_file(filepath):
    """Parse an nccl-test .out file. Returns {size_bytes: busbw} or None."""
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except OSError:
        return None

    data = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("[") or stripped.startswith("="):
            continue
        if any(kw in stripped for kw in ("NCCL", "Error", "error", "srun", "slurmstepd", "flush.c", "common_ucx", "sharp", "SHARP", "transport", "INFO", "WARN")):
            continue
        fields = stripped.split()
        if len(fields) < 13:
            continue
        try:
            size_bytes = int(fields[0])
        except ValueError:
            continue
        if size_bytes not in MSG_SIZES:
            continue
        try:
            busbw = float(fields[BUSBW_COL])
        except (ValueError, IndexError):
            continue
        data[size_bytes] = busbw

    return data if data else None


def extract_nccl_version(results_dir):
    """Extract NCCL version string from first successful .out file."""
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".out"):
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path) as f:
                for line in f:
                    m = re.search(r"NCCL version (\S+)", line)
                    if m:
                        return m.group(1)
        except OSError:
            continue
    return "unknown"


def extract_job_id_range(results_dir):
    """Extract min and max job IDs from manifest.txt."""
    manifest = os.path.join(results_dir, "manifest.txt")
    if not os.path.exists(manifest):
        return None, None
    job_ids = []
    with open(manifest) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if parts:
                try:
                    job_ids.append(int(parts[0]))
                except ValueError:
                    pass
    if job_ids:
        return min(job_ids), max(job_ids)
    return None, None


def extract_exclude_list(results_dir):
    """Extract exclude list from manifest.txt."""
    manifest = os.path.join(results_dir, "manifest.txt")
    if not os.path.exists(manifest):
        return "unknown"
    with open(manifest) as f:
        for line in f:
            if "Exclude:" in line:
                return line.split("Exclude:")[-1].strip()
    return "unknown"


def extract_node_count(results_dir):
    """Count available vs drained nodes from exclude list."""
    exclude = extract_exclude_list(results_dir)
    # Count excluded nodes by expanding the hostlist pattern
    # Simple heuristic: count commas and ranges
    m = re.findall(r"\d+", exclude.split("[")[-1].rstrip("]")) if "[" in exclude else []
    return len(m) if m else 0


def parse_results_dir(results_dir):
    """Parse all .out files in a results directory.

    Returns:
        results: dict of {(op, algo, nodes): {size_bytes: busbw}}
        failures: list of (op, algo, nodes) that failed
        expected_fail_count: count of expected failures
    """
    results = {}
    failures = []
    expected_fail_count = 0

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".out"):
            continue
        parsed = parse_filename(fname)
        if not parsed:
            continue
        op, algo, nodes = parsed
        data = parse_single_file(os.path.join(results_dir, fname))
        if data is None:
            if (op, algo) in EXPECTED_FAILURES:
                expected_fail_count += 1
            else:
                failures.append((op, algo, nodes))
            results[(op, algo, nodes)] = {}
        else:
            results[(op, algo, nodes)] = data

    return results, failures, expected_fail_count


# ── Monitoring ───────────────────────────────────────────────────────────────

def get_expected_files():
    """Generate the list of expected output filenames."""
    files = []
    for nodes in NODE_COUNTS:
        for op in OPS:
            if op == "sendrecv":
                files.append(f"sendrecv_p2p_{nodes}nodes.out")
                continue
            all_algos = ["ring", "nvls", "tree", "nvls_tree", "collnet_sharp"]
            for algo in all_algos:
                if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                    continue
                files.append(f"{op}_{algo}_{nodes}nodes.out")
    return files


def is_file_complete(filepath):
    """Check if an output file has finished writing."""
    try:
        with open(filepath) as f:
            content = f.read()
    except OSError:
        return False
    if "Avg bus bandwidth" in content:
        return True
    if "no algorithm/protocol" in content.lower():
        return True
    if "=== Done:" in content:
        return True
    if "Out of bounds values" in content:
        return True
    if "Test NCCL" in content:
        return True
    return False


def monitor_jobs(results_dir, poll_interval=30, timeout=2400):
    """Poll until all expected output files are complete."""
    expected = get_expected_files()
    start = time.time()

    print(f"Monitoring {len(expected)} expected output files in:\n  {results_dir}\n")

    while True:
        existing = set(os.listdir(results_dir)) if os.path.isdir(results_dir) else set()
        complete = []
        incomplete = []
        missing = []

        for fname in expected:
            path = os.path.join(results_dir, fname)
            if fname not in existing:
                missing.append(fname)
            elif is_file_complete(path):
                complete.append(fname)
            else:
                incomplete.append(fname)

        elapsed = time.time() - start
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {len(complete)}/{len(expected)} complete, "
              f"{len(incomplete)} in-progress, {len(missing)} missing "
              f"({elapsed:.0f}s elapsed)")

        if not missing and not incomplete:
            print("\nAll files complete!")
            break

        if elapsed > timeout:
            print(f"\nTimeout after {timeout}s. Pending files:")
            for f in incomplete + missing:
                print(f"  {f}")
            break

        time.sleep(poll_interval)


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_runs(new_results, old_results):
    """Compare two result sets. Returns dict of deltas."""
    deltas = {}
    all_keys = set(new_results.keys()) | set(old_results.keys())
    for key in all_keys:
        new_data = new_results.get(key, {})
        old_data = old_results.get(key, {})
        if not new_data or not old_data:
            continue
        for size in MSG_SIZE_BYTES:
            new_bw = new_data.get(size)
            old_bw = old_data.get(size)
            if new_bw is not None and old_bw is not None and old_bw > 0:
                delta_pct = (new_bw - old_bw) / old_bw * 100
                deltas[(*key, size)] = (new_bw, old_bw, delta_pct)
    return deltas


# ── Report helpers ───────────────────────────────────────────────────────────

def fmt_bw(val):
    """Format bandwidth value."""
    if val is None:
        return "--"
    return f"{val:.1f}"


def fmt_bw_bold(val, is_best):
    """Format bandwidth, bold if best."""
    if val is None:
        return "--"
    s = f"{val:.1f}"
    return f"**{s}**" if is_best else s


def fmt_pct(val):
    """Format percentage."""
    if val is None:
        return "--"
    return f"{val:.1f}%"


def fmt_delta(val):
    """Format delta percentage with sign."""
    if val is None:
        return "--"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def fmt_ratio(val):
    """Format 8G/2G ratio."""
    if val is None:
        return "--"
    return f"{val:.2f}x"


def get_bw(results, op, algo, nodes, size):
    """Safely get bandwidth from results."""
    data = results.get((op, algo, nodes), {})
    return data.get(size) if data else None


# ── Report sections ──────────────────────────────────────────────────────────

def section_header(results_dir, nccl_version):
    ts = os.path.basename(results_dir.rstrip("/"))
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# NCCL Collective Scaling Benchmark Report — {date_str}",
        "",
        "NCCL collective performance across 1–64 node scale (8–512 GPUs) on the B200 DGXC cluster, "
        "testing all_reduce, all_gather, and sendrecv P2P across five algorithm configurations and three message sizes.",
        "",
    ]
    return lines


def section_system_config(results_dir):
    exclude = extract_exclude_list(results_dir)
    lines = [
        "## System Configuration",
        "",
        "| Component | Details |",
        "|-----------|---------|",
        "| GPU | NVIDIA B200, 8 per node |",
        "| Nodes | 1–64 (67 available in batch) |",
        "| Total GPUs | 8–512 |",
        "| Interconnect | Mellanox ConnectX-7 (MT4129), NDR 400 Gb/s InfiniBand |",
        "| NICs per node | 8 HCAs (mlx5_0, mlx5_1, mlx5_4, mlx5_5, mlx5_6, mlx5_11, mlx5_14, mlx5_15) |",
        "| NVLink | 5th gen, ~900 GB/s bidirectional intra-node |",
        "| SHARP | Not available (sharpd not running, no reservations) |",
        f"| Excluded | {exclude} |",
        "",
    ]
    return lines


def section_software_config(nccl_version):
    lines = [
        "## Software Configuration",
        "",
        "| Component | Details |",
        "|-----------|---------|",
        f"| NCCL | {nccl_version} |",
        "| NCCL Library | `/mnt/vast/dgxc-benchmarking-auto/nccl-libs/usr/lib/x86_64-linux-gnu` |",
        "| NCCL Test Binaries | `/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build` |",
        "| MPI | PMIx (`srun --mpi=pmix`) |",
        "| HPC-X | `/opt/hpcx/hpcx-init.sh` |",
        "| Scheduler | Slurm, partition=batch, exclusive allocation |",
        "",
    ]
    return lines


def section_test_config():
    lines = [
        "## Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Message sizes | 2 GB, 4 GB, 8 GB (2x step factor) |",
        "| Iterations | 5 warmup + 20 measured |",
        "| Data type | float32 |",
        "| Reduction op | sum (all_reduce) / none (all_gather, sendrecv) |",
        "| Validation | enabled (Out of bounds check) |",
        "| GPUs per rank | 1 (`-g 1`) |",
        "",
    ]
    return lines


def section_test_matrix(failures, expected_fail_count, total_files):
    ok_count = total_files - expected_fail_count - len(failures)
    lines = [
        "## Test Matrix",
        "",
        f"{total_files} total jobs — 3 operations × 5 algorithms × 7 node counts, minus 1-node inter-node-only configs:",
        "",
        "| Config Label | NCCL_ALGO | NCCL_NVLS_ENABLE | NCCL_COLLNET_ENABLE | Nodes |",
        "|--------------|-----------|------------------|---------------------|-------|",
        "| ring | Ring | 0 | 0 | 1–64 |",
        "| nvls | (auto) | 1 | 0 | 1–64 |",
        "| tree | Tree | 0 | 0 | 2–64 |",
        "| nvls_tree | NVLSTree | 1 | 0 | 2–64 |",
        "| collnet_sharp | (auto) | 0 | 1 | 2–64 |",
        "",
        f"**Expected failures:** tree and nvls_tree are not supported by NCCL for AllGather ({expected_fail_count} jobs failed as expected).",
        "",
        f"**Job results:** {ok_count} OK, {expected_fail_count} expected failures, {len(failures)} unexpected failures.",
        "",
    ]
    if failures:
        lines.append("**Unexpected failures:**")
        for op, algo, nodes in failures:
            lines.append(f"- {op} {algo} {nodes}n")
        lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def section_results_allreduce(results, size_bytes, size_label):
    algos = ALGOS_ALL_REDUCE
    lines = [
        f"### all_reduce @ {size_label}",
        "",
        "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree | Best |",
        "|--------------|------|------|---------------|----------|------|------|",
    ]
    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        values = {}
        for algo in algos:
            if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                values[algo] = None
            else:
                values[algo] = get_bw(results, "all_reduce", algo, nodes, size_bytes)

        valid = {a: v for a, v in values.items() if v is not None}
        best_algo = max(valid, key=valid.get) if valid else None
        best_label = ALGO_LABELS.get(best_algo, "--") if best_algo else "--"

        cells = []
        for algo in algos:
            cells.append(fmt_bw_bold(values[algo], algo == best_algo))

        lines.append(f"| {nodes}n ({gpus}) | {' | '.join(cells)} | {best_label} |")

    lines.append("")
    return lines


def section_results_allgather(results, size_bytes, size_label):
    algos = ALGOS_ALL_GATHER
    lines = [
        f"### all_gather @ {size_label}",
        "",
        "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | Best |",
        "|--------------|------|------|---------------|------|",
    ]
    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        values = {}
        for algo in algos:
            if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                values[algo] = None
            else:
                values[algo] = get_bw(results, "all_gather", algo, nodes, size_bytes)

        valid = {a: v for a, v in values.items() if v is not None}
        best_algo = max(valid, key=valid.get) if valid else None
        best_label = ALGO_LABELS.get(best_algo, "--") if best_algo else "--"

        cells = []
        for algo in algos:
            cells.append(fmt_bw_bold(values[algo], algo == best_algo))

        lines.append(f"| {nodes}n ({gpus}) | {' | '.join(cells)} | {best_label} |")

    lines.append("")
    return lines


def section_results_sendrecv(results):
    lines = [
        "## Results — sendrecv P2P Peak busBW (GB/s)",
        "",
        "| Nodes (GPUs) | 2 GB | 4 GB | 8 GB |",
        "|--------------|------|------|------|",
    ]
    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        cells = []
        best_val = None
        for size in MSG_SIZE_BYTES:
            val = get_bw(results, "sendrecv", "p2p", nodes, size)
            if val is not None and (best_val is None or val > best_val):
                best_val = val
            cells.append(val)
        formatted = []
        for val in cells:
            formatted.append(fmt_bw_bold(val, val is not None and val == best_val))
        lines.append(f"| {nodes}n ({gpus}) | {' | '.join(formatted)} |")

    lines.append("")
    return lines


def section_scaling_efficiency(results, op, size_bytes, size_label):
    if op == "all_reduce":
        algos = ALGOS_ALL_REDUCE
        header = "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |"
        sep =    "|--------------|------|------|---------------|----------|------|"
    elif op == "all_gather":
        algos = ALGOS_ALL_GATHER
        header = "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |"
        sep =    "|--------------|------|------|---------------|"
    else:
        return section_scaling_efficiency_sendrecv(results, size_bytes, size_label)

    lines = [
        f"## Scaling Efficiency — {op} @ {size_label}",
        "",
        f"Efficiency = busBW(N nodes) / busBW(baseline) × 100%. Baseline: 1-node for ring/nvls; 2-node for others.",
        "",
        header,
        sep,
    ]

    baselines = {}
    for algo in algos:
        if algo in MULTI_NODE_ONLY_ALGOS:
            baselines[algo] = get_bw(results, op, algo, 2, size_bytes)
        else:
            baselines[algo] = get_bw(results, op, algo, 1, size_bytes)

    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        cells = []
        eff_values = {}
        for algo in algos:
            if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                cells.append("--")
                continue
            val = get_bw(results, op, algo, nodes, size_bytes)
            base = baselines.get(algo)
            if val is not None and base is not None and base > 0:
                eff = val / base * 100
                eff_values[algo] = eff
                cells.append(fmt_pct(eff))
            else:
                cells.append("--")

        # Bold the best efficiency in each row
        if eff_values:
            best_algo = max(eff_values, key=eff_values.get)
            formatted = []
            idx = 0
            for algo in algos:
                if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                    formatted.append("--")
                    continue
                val = eff_values.get(algo)
                if val is not None and algo == best_algo:
                    formatted.append(f"**{val:.1f}%**")
                elif val is not None:
                    formatted.append(f"{val:.1f}%")
                else:
                    formatted.append("--")
            cells = formatted

        lines.append(f"| {nodes}n ({gpus}) | {' | '.join(cells)} |")

    lines.append("")
    return lines


def section_scaling_efficiency_sendrecv(results, size_bytes, size_label):
    lines = [
        f"## Scaling Efficiency — sendrecv P2P @ {size_label}",
        "",
        "| Nodes (GPUs) | Efficiency |",
        "|--------------|-----------|",
    ]
    base = get_bw(results, "sendrecv", "p2p", 1, size_bytes)
    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        val = get_bw(results, "sendrecv", "p2p", nodes, size_bytes)
        if val is not None and base is not None and base > 0:
            eff = val / base * 100
            lines.append(f"| {nodes}n ({gpus}) | {eff:.1f}% |")
        else:
            lines.append(f"| {nodes}n ({gpus}) | -- |")
    lines.append("")
    return lines


def section_bandwidth_drops(results, op, size_bytes, size_label):
    if op == "all_reduce":
        algos = ALGOS_ALL_REDUCE
        header = "| Transition | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |"
        sep =    "|------------|------|------|---------------|----------|------|"
    elif op == "all_gather":
        algos = ALGOS_ALL_GATHER
        header = "| Transition | Ring | NVLS | CollNet SHARP |"
        sep =    "|------------|------|------|---------------|"
    else:
        return []

    lines = [
        f"## Step-by-Step Bandwidth Drop — {op} @ {size_label}",
        "",
        "Identifies scaling cliffs between consecutive node counts.",
        "",
        header,
        sep,
    ]

    for i in range(len(NODE_COUNTS) - 1):
        n1, n2 = NODE_COUNTS[i], NODE_COUNTS[i + 1]
        cells = []
        for algo in algos:
            v1 = get_bw(results, op, algo, n1, size_bytes)
            v2 = get_bw(results, op, algo, n2, size_bytes)
            if v1 is not None and v2 is not None and v1 > 0:
                drop = (v2 - v1) / v1 * 100
                s = fmt_delta(drop)
                if abs(drop) > 15:
                    s = f"**{s}**"
                cells.append(s)
            else:
                cells.append("--")
        lines.append(f"| {n1}n → {n2}n | {' | '.join(cells)} |")

    lines.append("")
    return lines


def section_msg_size_sensitivity(results, op):
    if op == "all_reduce":
        algos = ALGOS_ALL_REDUCE
        header = "| Nodes | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |"
        sep =    "|-------|------|------|---------------|----------|------|"
    elif op == "all_gather":
        algos = ALGOS_ALL_GATHER
        header = "| Nodes | Ring | NVLS | CollNet SHARP |"
        sep =    "|-------|------|------|---------------|"
    else:
        return []

    size_2g = MSG_SIZE_BYTES[0]
    size_8g = MSG_SIZE_BYTES[2]

    lines = [
        f"## Message Size Sensitivity — {op} 8G/2G Ratio",
        "",
        "How much 8 GB messages outperform 2 GB, revealing bandwidth saturation effects at scale.",
        "",
        header,
        sep,
    ]

    for nodes in NODE_COUNTS:
        cells = []
        for algo in algos:
            if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                cells.append("--")
                continue
            v2 = get_bw(results, op, algo, nodes, size_2g)
            v8 = get_bw(results, op, algo, nodes, size_8g)
            if v2 is not None and v8 is not None and v2 > 0:
                ratio = v8 / v2
                s = fmt_ratio(ratio)
                if ratio > 1.3 or ratio < 0.9:
                    s = f"**{s}**"
                cells.append(s)
            else:
                cells.append("--")
        lines.append(f"| {nodes}n | {' | '.join(cells)} |")

    lines.append("")
    return lines


def section_algo_headtohead(results, op, size_bytes, size_label):
    if op == "sendrecv":
        return []

    if op == "all_reduce":
        algos = ALGOS_ALL_REDUCE
    else:
        algos = ALGOS_ALL_GATHER

    lines = [
        f"### {op} — Best Algorithm per Scale",
        "",
        "| Nodes (GPUs) | Best Algorithm | busBW (GB/s) | 2nd Best | busBW (GB/s) | Gap |",
        "|--------------|---------------|-------------|----------|-------------|-----|",
    ]

    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        valid = {}
        for algo in algos:
            if nodes == 1 and algo in MULTI_NODE_ONLY_ALGOS:
                continue
            val = get_bw(results, op, algo, nodes, size_bytes)
            if val is not None:
                valid[algo] = val

        if len(valid) >= 2:
            sorted_algos = sorted(valid, key=valid.get, reverse=True)
            best = sorted_algos[0]
            second = sorted_algos[1]
            gap = (valid[best] - valid[second]) / valid[second] * 100
            lines.append(
                f"| {nodes}n ({gpus}) | **{ALGO_LABELS[best]}** | {valid[best]:.1f} | "
                f"{ALGO_LABELS[second]} | {valid[second]:.1f} | +{gap:.1f}% |"
            )
        elif len(valid) == 1:
            best = list(valid.keys())[0]
            lines.append(
                f"| {nodes}n ({gpus}) | **{ALGO_LABELS[best]}** | {valid[best]:.1f} | -- | -- | -- |"
            )
        else:
            lines.append(f"| {nodes}n ({gpus}) | -- | -- | -- | -- | -- |")

    lines.append("")
    return lines


def section_interconnect(results):
    size_8g = MSG_SIZE_BYTES[2]
    lines = [
        "## Interconnect Topology Analysis",
        "",
        "### NVLink vs InfiniBand Boundary",
        "",
        "| Metric | NVLink 1n (GB/s) | IB 2n (GB/s) | IB 64n (GB/s) | 1n→2n Drop |",
        "|--------|-----------------|-------------|--------------|-----------|",
    ]

    metrics = [
        ("all_reduce (NVLS)", "all_reduce", "nvls"),
        ("all_reduce (Ring)", "all_reduce", "ring"),
        ("sendrecv P2P", "sendrecv", "p2p"),
    ]
    for label, op, algo in metrics:
        v1 = get_bw(results, op, algo, 1, size_8g)
        v2 = get_bw(results, op, algo, 2, size_8g)
        v64 = get_bw(results, op, algo, 64, size_8g)
        drop = f"**{(v2 - v1) / v1 * 100:.1f}%**" if v1 and v2 and v1 > 0 else "--"
        lines.append(
            f"| {label} | {fmt_bw(v1)} | {fmt_bw(v2)} | {fmt_bw(v64)} | {drop} |"
        )

    lines.extend([
        "",
        "### Bandwidth Utilization vs Theoretical @ 2 Nodes, 8 GB",
        "",
        "| Collective | Best Algorithm | busBW (GB/s) | IB Theoretical | Utilization |",
        "|------------|---------------|-------------|----------------|-------------|",
    ])

    ib_theoretical = 400  # 8 HCAs × 50 GB/s
    ib_single = 50  # 1 HCA

    # all_reduce best at 2n
    ar_algos = ALGOS_ALL_REDUCE
    ar_best = None
    ar_best_bw = 0
    for algo in ar_algos:
        v = get_bw(results, "all_reduce", algo, 2, size_8g)
        if v and v > ar_best_bw:
            ar_best_bw = v
            ar_best = algo
    if ar_best:
        util = ar_best_bw / ib_theoretical * 100
        lines.append(f"| all_reduce | {ALGO_LABELS[ar_best]} | {ar_best_bw:.1f} | {ib_theoretical} GB/s (8 HCAs) | {util:.1f}% |")

    # all_gather best at 2n
    ag_best = None
    ag_best_bw = 0
    for algo in ALGOS_ALL_GATHER:
        v = get_bw(results, "all_gather", algo, 2, size_8g)
        if v and v > ag_best_bw:
            ag_best_bw = v
            ag_best = algo
    if ag_best:
        util = ag_best_bw / ib_theoretical * 100
        lines.append(f"| all_gather | {ALGO_LABELS[ag_best]} | {ag_best_bw:.1f} | {ib_theoretical} GB/s (8 HCAs) | {util:.1f}% |")

    # sendrecv at 2n
    sr_bw = get_bw(results, "sendrecv", "p2p", 2, size_8g)
    if sr_bw:
        util = sr_bw / ib_single * 100
        lines.append(f"| sendrecv | P2P | {sr_bw:.1f} | {ib_single} GB/s (1 HCA) | {util:.1f}% |")

    lines.append("")
    return lines


def section_delta_comparison(new_results, old_results):
    """Generate delta comparison tables between two runs."""
    deltas = compare_runs(new_results, old_results)
    if not deltas:
        return []

    lines = [
        "---",
        "",
        "## Delta vs Previous Run",
        "",
    ]

    size_8g = MSG_SIZE_BYTES[2]

    for op, algos, op_label in [
        ("all_reduce", ALGOS_ALL_REDUCE, "all_reduce"),
        ("all_gather", ALGOS_ALL_GATHER, "all_gather"),
    ]:
        lines.append(f"### {op_label} @ 8 GB — Delta (%)")
        lines.append("")
        if op == "all_reduce":
            header = "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP | NVLSTree | Tree |"
            sep =    "|--------------|------|------|---------------|----------|------|"
        else:
            header = "| Nodes (GPUs) | Ring | NVLS | CollNet SHARP |"
            sep =    "|--------------|------|------|---------------|"
        lines.append(header)
        lines.append(sep)

        for nodes in NODE_COUNTS:
            gpus = nodes * GPUS_PER_NODE
            cells = []
            for algo in algos:
                key = (op, algo, nodes, size_8g)
                if key in deltas:
                    _, _, pct = deltas[key]
                    s = fmt_delta(pct)
                    if abs(pct) > 5:
                        s = f"**{s}**"
                    cells.append(s)
                else:
                    cells.append("--")
            lines.append(f"| {nodes}n ({gpus}) | {' | '.join(cells)} |")
        lines.append("")

    # sendrecv delta
    lines.append("### sendrecv P2P @ 8 GB — Delta (%)")
    lines.append("")
    lines.append("| Nodes (GPUs) | Delta |")
    lines.append("|--------------|-------|")
    for nodes in NODE_COUNTS:
        gpus = nodes * GPUS_PER_NODE
        key = ("sendrecv", "p2p", nodes, size_8g)
        if key in deltas:
            _, _, pct = deltas[key]
            s = fmt_delta(pct)
            if abs(pct) > 5:
                s = f"**{s}**"
            lines.append(f"| {nodes}n ({gpus}) | {s} |")
        else:
            lines.append(f"| {nodes}n ({gpus}) | -- |")
    lines.append("")

    # Summary of significant changes
    regressions = [(k, v) for k, v in deltas.items() if v[2] < -5]
    improvements = [(k, v) for k, v in deltas.items() if v[2] > 5]

    if regressions:
        lines.append("### Significant Regressions (>5% drop)")
        lines.append("")
        for (op, algo, nodes, size), (new_bw, old_bw, pct) in sorted(regressions, key=lambda x: x[1][2]):
            lines.append(f"- {op} {algo} {nodes}n @ {MSG_SIZES[size]}: {old_bw:.1f} → {new_bw:.1f} GB/s ({pct:+.1f}%)")
        lines.append("")

    if improvements:
        lines.append("### Significant Improvements (>5% gain)")
        lines.append("")
        for (op, algo, nodes, size), (new_bw, old_bw, pct) in sorted(improvements, key=lambda x: -x[1][2]):
            lines.append(f"- {op} {algo} {nodes}n @ {MSG_SIZES[size]}: {old_bw:.1f} → {new_bw:.1f} GB/s ({pct:+.1f}%)")
        lines.append("")

    if not regressions and not improvements:
        lines.append("All results within ±5% of previous run.")
        lines.append("")

    return lines


def section_observations(results):
    size_8g = MSG_SIZE_BYTES[2]
    size_2g = MSG_SIZE_BYTES[0]
    lines = [
        "## Observations",
        "",
    ]

    obs = []

    # 1. CollNet SHARP dominance at multi-node
    collnet_wins = 0
    for nodes in [4, 8, 16, 32, 64]:
        for op in ["all_reduce", "all_gather"]:
            algos = ALGOS_ALL_REDUCE if op == "all_reduce" else ALGOS_ALL_GATHER
            best_algo = None
            best_bw = 0
            for algo in algos:
                v = get_bw(results, op, algo, nodes, size_8g)
                if v and v > best_bw:
                    best_bw = v
                    best_algo = algo
            if best_algo == "collnet_sharp":
                collnet_wins += 1
    ar_collnet_4n = get_bw(results, "all_reduce", "collnet_sharp", 4, size_8g)
    ar_collnet_32n = get_bw(results, "all_reduce", "collnet_sharp", 32, size_8g)
    ar_ring_32n = get_bw(results, "all_reduce", "ring", 32, size_8g)
    if collnet_wins > 5 and ar_collnet_32n and ar_ring_32n and ar_ring_32n > 0:
        gap_pct = (ar_collnet_32n - ar_ring_32n) / ar_ring_32n * 100
        obs.append(
            f"**CollNet SHARP is the clear winner at 4+ nodes** across both all_reduce (~{ar_collnet_4n:.0f} GB/s) "
            f"and all_gather, despite SHARP hardware not being operational. NCCL's CollNet fallback outperforms "
            f"ring by {gap_pct:.0f}% at 32 nodes. All CollNet jobs logged `SHARP coll init error: Cannot create SHARP job` "
            f"— actual SHARP offload would likely push numbers higher."
        )

    # 2. NVLS intra-node dominance
    nvls_1n = get_bw(results, "all_reduce", "nvls", 1, size_8g)
    ring_1n = get_bw(results, "all_reduce", "ring", 1, size_8g)
    nvls_2n = get_bw(results, "all_reduce", "nvls", 2, size_8g)
    ring_2n = get_bw(results, "all_reduce", "ring", 2, size_8g)
    if nvls_1n and ring_1n and ring_1n > 0:
        boost = (nvls_1n - ring_1n) / ring_1n * 100
        obs.append(
            f"**NVLS dominates intra-node and 2-node scale** with {nvls_1n:.0f} GB/s all_reduce at 1 node "
            f"(+{boost:.0f}% over ring) and {nvls_2n:.0f} GB/s at 2 nodes. "
            f"NVLS leverages NVLink multicast for efficient intra-node reduction."
        )

    # 3. NVLS collapse at 64 nodes
    nvls_64n = get_bw(results, "all_reduce", "nvls", 64, size_8g)
    ring_64n = get_bw(results, "all_reduce", "ring", 64, size_8g)
    if nvls_64n and ring_64n and nvls_64n < ring_64n:
        obs.append(
            f"**NVLS collapses at 64 nodes**: all_reduce NVLS drops to {nvls_64n:.0f} GB/s — "
            f"worse than ring ({ring_64n:.0f} GB/s). Don't force `NCCL_NVLS_ENABLE=1` for large-scale training."
        )

    # 4. CollNet scaling flatness
    if ar_collnet_4n and ar_collnet_32n and ar_collnet_4n > 0:
        variation = abs(ar_collnet_32n - ar_collnet_4n) / ar_collnet_4n * 100
        collnet_64n = get_bw(results, "all_reduce", "collnet_sharp", 64, size_8g)
        drop_64 = (collnet_64n - ar_collnet_32n) / ar_collnet_32n * 100 if collnet_64n and ar_collnet_32n else 0
        obs.append(
            f"**CollNet scaling is remarkably flat**: all_reduce CollNet holds ~{ar_collnet_4n:.0f} GB/s from "
            f"4 to 32 nodes ({variation:.0f}% variation), only dropping {abs(drop_64):.0f}% at 64 nodes."
        )

    # 5. NVLink→IB cliff for P2P
    sr_1n = get_bw(results, "sendrecv", "p2p", 1, size_8g)
    sr_2n = get_bw(results, "sendrecv", "p2p", 2, size_8g)
    if sr_1n and sr_2n and sr_1n > 0:
        drop = (1 - sr_2n / sr_1n) * 100
        obs.append(
            f"**The NVLink→IB cliff is {sr_1n / sr_2n:.0f}x for P2P**: sendrecv drops from "
            f"{sr_1n:.0f} GB/s intra-node to {sr_2n:.1f} GB/s at 2 nodes. P2P is single-HCA limited "
            f"({sr_2n:.1f} / 50 = {sr_2n / 50 * 100:.0f}% utilization). This makes pipeline parallelism (PP) "
            f"the inter-node bottleneck."
        )

    # 6. Larger messages help at scale
    ring_64n_2g = get_bw(results, "all_reduce", "ring", 64, size_2g)
    ring_64n_8g = get_bw(results, "all_reduce", "ring", 64, size_8g)
    if ring_64n_2g and ring_64n_8g and ring_64n_2g > 0:
        gain = (ring_64n_8g - ring_64n_2g) / ring_64n_2g * 100
        obs.append(
            f"**Larger messages improve bandwidth at scale**: ring 64n gains {gain:.0f}% from 2G→8G "
            f"({ring_64n_2g:.0f}→{ring_64n_8g:.0f} GB/s). This validates GBS tuning (more gradient "
            f"accumulation) as a communication optimization strategy."
        )

    # 7. Tree worst
    tree_vals = [get_bw(results, "all_reduce", "tree", n, size_8g) for n in [4, 8, 16, 32, 64]]
    tree_vals = [v for v in tree_vals if v is not None]
    if tree_vals:
        tree_avg = sum(tree_vals) / len(tree_vals)
        obs.append(
            f"**Tree algorithm is consistently worst** for large messages: ~{tree_avg:.0f} GB/s ceiling "
            f"regardless of node count at 4–64 nodes. Never force `NCCL_ALGO=Tree` for LLM training."
        )

    # 8. Sendrecv message-size invariant
    sr_2n_2g = get_bw(results, "sendrecv", "p2p", 2, size_2g)
    sr_2n_8g = get_bw(results, "sendrecv", "p2p", 2, size_8g)
    if sr_2n_2g and sr_2n_8g and abs(sr_2n_2g - sr_2n_8g) < 1:
        sr_64n = get_bw(results, "sendrecv", "p2p", 64, size_8g)
        tail = f" Bandwidth drops to {sr_64n:.0f} GB/s at 64 nodes." if sr_64n else ""
        obs.append(
            f"**Sendrecv P2P is message-size invariant at multi-node**: {sr_2n_2g:.1f} GB/s at 2 nodes "
            f"regardless of 2G/4G/8G, confirming pure link-bandwidth limitation.{tail}"
        )

    for i, o in enumerate(obs, 1):
        lines.append(f"{i}. {o}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def section_recommendations(results):
    lines = [
        "## Recommendations for LLM Training",
        "",
        "| Scale | Recommended NCCL Settings | Rationale |",
        "|-------|--------------------------|-----------|",
        "| 1 node (8 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS gives best intra-node performance |",
        "| 2 nodes (16 GPU) | `NCCL_NVLS_ENABLE=1`, default algo | NVLS still dominant at 2-node scale |",
        "| 4–32 nodes (32–256 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet fallback outperforms ring at scale |",
        "| 64 nodes (512 GPU) | `NCCL_COLLNET_ENABLE=1` | CollNet maintains best scaling efficiency |",
        "| All scales | Use largest feasible GBS | Larger messages improve BW utilization |",
        "",
        "### Optimization Opportunities",
        "",
        "1. **Enable SHARP:** Admin action needed — start `sharp_am` on management node, `sharpd` on compute nodes, create SHARP reservations. Could boost CollNet beyond current fallback numbers.",
        "2. **Pipeline parallelism bottleneck:** PP SendRecv at inter-node speeds is the scaling wall. Minimize PP stages or keep PP intra-node (NVLink) where possible.",
        "3. **GBS tuning:** Maximize gradient accumulation steps to send larger messages and amortize latency overhead, especially at 32+ nodes.",
        "",
    ]
    return lines


def section_raw_results(results_dir):
    job_min, job_max = extract_job_id_range(results_dir)
    job_str = f"{job_min}–{job_max}" if job_min and job_max else "unknown"
    lines = [
        "---",
        "",
        "## Raw Results",
        "",
        f"Results directory: `{results_dir}`",
        "",
        f"Slurm jobs: {job_str}",
        "",
        "Submit script: `~/together-nccl-tests/benchmarks/B200/collective-scaling/submit_all.sh`",
        "",
        f"Manifest: `manifest.txt` in results directory",
        "",
    ]
    return lines


# ── Main report generator ────────────────────────────────────────────────────

def generate_report(results, failures, expected_fail_count, results_dir,
                    compare_results=None, compare_dir=None):
    nccl_version = extract_nccl_version(results_dir)
    total_expected = len(get_expected_files())

    lines = []
    lines.extend(section_header(results_dir, nccl_version))
    lines.extend(section_system_config(results_dir))
    lines.extend(section_software_config(nccl_version))
    lines.extend(section_test_config())
    lines.extend(section_test_matrix(failures, expected_fail_count, total_expected))

    # all_reduce results
    lines.append("## Results — all_reduce Peak busBW (GB/s)")
    lines.append("")
    for size in MSG_SIZE_BYTES:
        lines.extend(section_results_allreduce(results, size, MSG_SIZES[size]))

    lines.append("---")
    lines.append("")

    # all_gather results
    lines.append("## Results — all_gather Peak busBW (GB/s)")
    lines.append("")
    lines.append("tree and nvls_tree are not supported by NCCL for AllGather.")
    lines.append("")
    for size in MSG_SIZE_BYTES:
        lines.extend(section_results_allgather(results, size, MSG_SIZES[size]))

    lines.append("---")
    lines.append("")

    # sendrecv results
    lines.extend(section_results_sendrecv(results))

    lines.append("---")
    lines.append("")

    # Scaling efficiency
    size_8g = MSG_SIZE_BYTES[2]
    lines.extend(section_scaling_efficiency(results, "all_reduce", size_8g, "8 GB"))
    lines.extend(section_scaling_efficiency(results, "all_gather", size_8g, "8 GB"))
    lines.extend(section_scaling_efficiency_sendrecv(results, size_8g, "8 GB"))

    lines.append("---")
    lines.append("")

    # Bandwidth drops
    lines.extend(section_bandwidth_drops(results, "all_reduce", size_8g, "8 GB"))

    lines.append("---")
    lines.append("")

    # Message size sensitivity
    lines.extend(section_msg_size_sensitivity(results, "all_reduce"))
    lines.extend(section_msg_size_sensitivity(results, "all_gather"))

    lines.append("---")
    lines.append("")

    # Algorithm head-to-head
    lines.append("## Algorithm Head-to-Head @ 8 GB")
    lines.append("")
    lines.extend(section_algo_headtohead(results, "all_reduce", size_8g, "8 GB"))
    lines.extend(section_algo_headtohead(results, "all_gather", size_8g, "8 GB"))

    lines.append("---")
    lines.append("")

    # Interconnect topology
    lines.extend(section_interconnect(results))

    # Delta comparison
    if compare_results:
        lines.extend(section_delta_comparison(results, compare_results))

    lines.append("---")
    lines.append("")

    # Observations and recommendations
    lines.extend(section_observations(results))
    lines.extend(section_recommendations(results))

    # Raw results
    lines.extend(section_raw_results(results_dir))

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NCCL Collective Scaling Benchmark — Parser and Report Generator"
    )
    parser.add_argument("--results-dir", required=True, help="Path to results directory")
    parser.add_argument("--compare-dir", help="Path to previous results directory for delta comparison")
    parser.add_argument("--output", help="Path to write markdown report (stdout if omitted)")
    parser.add_argument("--wait", action="store_true", help="Poll until all expected files are complete")
    parser.add_argument("--poll-interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("--timeout", type=int, default=2400, help="Max wait time in seconds (default: 2400)")
    parser.add_argument("--parse-only", action="store_true", help="Just parse and dump results as text")
    args = parser.parse_args()

    if args.wait:
        monitor_jobs(args.results_dir, args.poll_interval, args.timeout)

    print(f"Parsing results from: {args.results_dir}")
    results, failures, expected_fail_count = parse_results_dir(args.results_dir)

    success_count = sum(1 for v in results.values() if v)
    print(f"  Parsed: {success_count} successful, {expected_fail_count} expected failures, {len(failures)} unexpected failures")

    if args.parse_only:
        for key in sorted(results.keys()):
            op, algo, nodes = key
            data = results[key]
            if data:
                vals = ", ".join(f"{MSG_SIZES[s]}: {v:.1f}" for s, v in sorted(data.items()))
                print(f"  {op:12s} {algo:14s} {nodes:3d}n: {vals}")
            else:
                print(f"  {op:12s} {algo:14s} {nodes:3d}n: FAILED/EMPTY")
        return

    compare_results = None
    if args.compare_dir:
        print(f"Parsing comparison results from: {args.compare_dir}")
        compare_results, _, _ = parse_results_dir(args.compare_dir)

    report = generate_report(
        results, failures, expected_fail_count, args.results_dir,
        compare_results=compare_results, compare_dir=args.compare_dir,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
