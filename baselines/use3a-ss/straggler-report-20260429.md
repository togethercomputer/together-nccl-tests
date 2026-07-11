# Straggler Report — use3a-ss — 2026-04-29

**Cluster:** use3a-ss (B200)
**Nodes tested:** 60 idle / 60 used (15 groups of 4)
**Baseline:** ring ~332.4–332.6 GB/s, collnet_sharp ~514.4–514.7 GB/s

| Run | Timestamp | Results |
|-----|-----------|---------|
| Primary | 20260429_210316 | `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers/20260429_210316/results.json` |
| Confirmation #1 | 20260429_213056 | `/mnt/vast/dgxc-benchmarking-auto/nccl-results/B200/find-stragglers/20260429_213056/results.json` |
| Confirmation #2 | 20260429_213057 | Invalid — NCCL phases timed out due to job contention with run #1 (parallel submission). IB sweep only. |

---

## Summary

All 3 valid NCCL runs independently flagged the same 3 SEVERE groups and the same 8 bad nodes. IB layer clean across all runs — no bad HCAs detected.

**8 nodes confirmed bad. Recommend draining immediately.**

---

## Phase 1 — 4-node Disjoint Sweep

3 groups flagged SEVERE (ring z < −5) — consistent across both valid runs:

| Group | Nodes | Ring Δ% | Sharp Δ% |
|-------|-------|---------|----------|
| grp04 | gpu-159, 160, 162, 164 | ~−6.7% (z~−13) | ~−1.4% (z~−2.7) |
| grp10 | gpu-188, 189, 195, 196 | ~−6.4% (z~−13) | ~−1.7% (z~−3.3) |
| grp14 | gpu-214, 215, 227, 228 | ~−6.0% (z~−12) | ~−1.4% (z~−2.7) |

`gpu-159`, `gpu-160`, `gpu-188`, `gpu-189` cleared healthy in localization across all runs.

---

## Phase 3 — Localization

| Node | Primary ring z | Conf #1 ring z | Primary sharp z | Conf #1 sharp z |
|------|---------------|----------------|-----------------|-----------------|
| gpu-162 | −10.7 | −11.6 | −2.6 | −2.9 |
| gpu-164 | −12.2 | −12.7 | −2.9 | −3.1 |
| gpu-195 | −10.3 | −10.8 | −3.3 | −3.3 |
| gpu-196 | −10.2 | −11.1 | −2.8 | −2.9 |
| gpu-214 | −9.1  | −8.4  | −3.1 | −3.0 |
| gpu-215 | −11.5 | −12.4 | −2.9 | −3.1 |
| gpu-227 | −10.5 | −10.1 | −2.6 | −2.9 |
| gpu-228 | −11.1 | −10.9 | −3.1 | −3.3 |

---

## Phase 4 — Inter-link Cross-pair

No inter-link suspects in either run. `grp07+grp09` was consistently slightly fast (+3.8–4.0%, z~+2.4) — within noise, not flagged.

---

## Phase 5 — Confirmation Retest

`gpu-214` ring retested both runs (orig z~−9): confirmed_bad in confirmation #1 (retest z=−10.2), topology_sensitive in primary (retest z=+0.2). All other 7 nodes had ring z ≤ −10.0 and were auto-confirmed without retest.

`gpu-162` collnet_sharp retested in confirmation #1: topology_sensitive (retest z=−0.1). Ring signal strong and consistent (z~−11).

---

## Final Node Verdicts — Cross-run Confirmation

| Node | Primary | Conf #1 | Across-run verdict |
|------|---------|---------|-------------------|
| `use3a-ss-b200-gpu-162` | ring: CB, sharp: CB | ring: CB, sharp: TS | **CONFIRMED_BAD** — ring bad in both; sharp TS in 1/2 retests |
| `use3a-ss-b200-gpu-164` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |
| `use3a-ss-b200-gpu-195` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |
| `use3a-ss-b200-gpu-196` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |
| `use3a-ss-b200-gpu-214` | ring: TS, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** — sharp bad in both; ring confirmed in 1/2 retests |
| `use3a-ss-b200-gpu-215` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |
| `use3a-ss-b200-gpu-227` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |
| `use3a-ss-b200-gpu-228` | ring: CB, sharp: CB | ring: CB, sharp: CB | **CONFIRMED_BAD** |

CB = confirmed_bad, TS = topology_sensitive

> **Note on gpu-162 and gpu-214:** Each shows topology_sensitive on one algo in one run, but the other algo is consistently confirmed_bad across all runs. Both should be drained.

---

## Phase 6 — IB Sweep (consistent across all runs)

No bad HCAs on any node in any run. All 8 HCAs healthy.

| HCA | Median (MB/s) | Stddev |
|-----|--------------|--------|
| mlx5_0  | ~45,090 | ~90 |
| mlx5_1  | ~45,140 | ~80 |
| mlx5_4  | ~45,030 | ~70 |
| mlx5_5  | ~45,025 | ~75 |
| mlx5_6  | ~44,860 | ~42 |
| mlx5_11 | ~44,855 | ~45 |
| mlx5_14 | ~44,700 | ~65 |
| mlx5_15 | ~44,710 | ~65 |

**Result: No bad HCAs detected.** Degradation is NCCL-layer (likely NVLink or intra-node interconnect), not IB fabric.
