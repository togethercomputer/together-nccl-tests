---
name: slinky cluster environment
description: Hardware, software, and configuration details for the slinky Slurm cluster — needed for running NCCL benchmarks. Includes confirmed node health status (20 IB-broken nodes) and SHARP inactive status as of 2026-05-04.
type: project
originSessionId: 959acd8b-aa46-4d21-b222-be7352c270dc
---
## Cluster Identity
- **Name**: flapping-airplanes (Slurm partition: `slinky`, distinct from use3a-ss B200 DGXC cluster)
- **Slurm partition**: `slinky` (also aliased as `all`)
- **Nodes**: 64 total, contiguous `slinky-[0-63]`, no gaps (verified 2026-05-11). Previously 51 with gaps at 14/17-18/46-49/51/55/57/59 — the missing nodes came online between 2026-05-09 and 2026-05-11.
- **TotalCPUs**: 10240 (64 × 160)

## Hardware (per node)
- **GPU**: NVIDIA B200, 8x per node, 183GB each, driver 575.57.08
- **CPU**: 160 threads (2x 40-core sockets)
- **RAM**: ~1.7TB
- **RDMA IB HCAs (12 — corrected 2026-05-09)**: mlx5_0..7, mlx5_9..12
  - All 12 are real IB devices, all show `IB/SHARP` capability in NCCL net plugin
  - mlx5_8 = NVSwitch internal; mlx5_13 = RoCE (used as one of 13 net devices in NCCL log)
  - Prior note saying mlx5_0..3 were "non-RDMA / NVSwitch" was WRONG — verified by passed K8s 62n run + 60n sbatch (job 1992) which used all 12 successfully
- **No bond0** interface (unlike use3a-ss)

## Software
- **NCCL**: 2.28.9+cuda12.9, installed at `/usr/lib/x86_64-linux-gnu/libnccl.so.2`
  - Note: /etc/environment still shows old package refs (2.26.5) — actual lib is 2.28.9
- **NCCL test binaries**: `/opt/nccl-tests/build/` — already in PATH, all *_perf binaries present
- **HPC-X**: `/opt/hpcx/hpcx-init.sh`, OpenMPI 4.1.7a1, all HPC-X dirs in env
- **SHARP**: **ACTIVE as of 2026-05-12** (job 2644, 64n: 534.4 GB/s @ 32 GiB busbw, +37.6% vs ring; 24,576 COLLNET/SHARP/GDRDMA markers, 0 errors). Was inactive 2026-05-11 — operations enabled it overnight. See [[SHARP-active-on-slinky-2026-05-12]] for working env vars and gotchas (CollnetChain is allreduce-only).
- **CUDA**: 12.9.0, nvcc at `/usr/local/cuda/bin/`
- **LD_LIBRARY_PATH**: already set system-wide (includes NCCL, HPC-X, NCCL RDMA plugin)

## Filesystems
- `/data` — 64TB shared NFS/network FS, use for results: `/data/home/johnson/`
- `/scratch` — 14TB node-local SSD
- No `/mnt/vast` (that was the old dgxc cluster)
- **`/home/johnson` == `/data/home/johnson`** — same inode/device (confirmed via `stat`). No need to copy files between them; everything in `~` is already on the persistent NFS.

## Key Differences from use3a-ss (old B200 DGXC cluster)
| Item | use3a-ss | slinky |
|------|----------|--------|
| Partition | `batch` | `slinky` |
| Shared FS | `/mnt/vast` | `/data` |
| NCCL binaries | `/mnt/vast/dgxc-benchmarking-auto/nccl-tests/build` | `/opt/nccl-tests/build` |
| NCCL lib | `/mnt/vast/dgxc-benchmarking-auto/nccl-libs/...` | system (no override needed) |
| IB HCAs | mlx5_0,1,4,5,6,11,14,15 | mlx5_4,5,6,7,9,10,11,12 |
| HPC-X staging | needed (var/tmp workaround) | not needed (works from /opt/hpcx) |
| bond0 | yes | no (eth0 only) |
| NCCL version | 2.29.7 | 2.28.9 |

## NCCL Environment Variables for This Cluster
```bash
# IB HCAs (8 real RDMA devices)
export NCCL_IB_HCA="=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=300
# No NCCL_SOCKET_IFNAME=bond0 (bond0 doesn't exist here)
# No custom LD_LIBRARY_PATH for NCCL (already in system LD_LIBRARY_PATH)
```

## run_slurm.sh Command Template
```bash
cd ~/together-nccl-tests
RESULTS_BASE=/data/home/johnson/nccl-results/flapping-airplanes \
./benchmarks/run_slurm.sh \
  --partition slinky \
  --gpu-type B200 \
  --binary-path ~/together-nccl-tests/build \
  --hpcx /opt/hpcx/hpcx-init.sh \
  [--node slinky-0 | --nodes slinky-0,slinky-1 | --all]
```

## UCX Fix (required for this cluster)
K8s pods enforce `ulimit -l 8192` — UCX/IB cannot lock memory. Fixed in `run_slurm.sh` by setting:
```bash
export UCX_TLS=self,sm,cuda_ipc,cuda_copy,tcp
export UCX_NET_DEVICES=eth0
```
NCCL handles all data movement via NVLink/IB directly; MPI only needs shared-mem/TCP for bootstrap.

## Node Health — end of 2026-05-12: 13 of 64 nodes flagged bad

**End of day (15:41 UTC): 13 nodes failed during real LLM training today** (20% of cluster). See [[2026-05-12-256gpu-sweep-recovery]] for full incident.

**Hard NODE_FAIL today**: slinky-5, 9, 11, 12, 14, 21, 26, 40 (slinky-40: 6 NODE_FAILs alone — likely hardware-faulty)
**Pair-sweep flagged**: slinky-2, 37, 56 (never reproduced under LLM load — conservatively excluded)
**Pyxis container fail**: slinky-18
**Conservative exclude**: slinky-20 (rank-0 of CANCELLED job)

**Diagnostic clue**: short NCCL tests (64n all_reduce 8 GiB at 13:51 job 2925: 325 GB/s, 0 errors) and pair-sweep at 10:55 (715 GB/s, all 62 nodes PASS) declared cluster healthy — yet the same nodes hit hard NODE_FAIL within 2-5 min of LLM training start. **NCCL connectivity tests are necessary-but-not-sufficient validators for sustained-training stability on this cluster.**

**Multi-node-IB-only fault hypothesis**: slinky-9 and slinky-14 hosted user `theoh`'s single-node MoE training successfully even while failing my multi-node LLM jobs. Failure mode is most likely an inter-node IB QP or fabric-manager fault, not per-node GPU/compute hardware.

**Recommended for ops**: drain/reboot slinky-40 (6 fails today), slinky-5, 11, 12, 21 (1 fail each, today's first failure). slinky-9, 14, 26 likely intermittent.

### Earlier waypoints
- **Morning (09:00 UTC): all 64 healthy** — pair-sweep on 62 nodes 31/31 PASS @ 715 GB/s.
- **By 10:30 UTC**: 10 nodes flagged across 90 min of LLM sweep attempts (SHARP-attempted; aborted 10:28). See [[2026-05-12-512gpu-sweep-aborted]].
- **By 12:15 UTC**: 256-GPU sweep main run done — 4/8 workloads with valid data, 4 lost to NODE_FAIL/PREFLIGHT_FAIL cascades.
- **By 15:41 UTC**: retry v3 finished — 7/8 valid (only nemotronh abandoned). See [[2026-05-12-256gpu-sweep-recovery]].

| Node | Mode | When |
|---|---|---|
| slinky-9 | NODE_FAIL × 2 (LLM jobs 2742, 2776) | 09:54, 10:07 |
| slinky-14 | NODE_FAIL (job 2810 rank-0) | 10:21 |
| slinky-18 | pyxis container failure (job 2829) | 10:27 |
| slinky-26 | NODE_FAIL (job 2774 fallback) | 09:59 |
| slinky-2, 11, 13, 28, 37, 56 | pair-sweep flagged | 09:54-10:26 |

**Diagnostic clue:** slinky-9 was simultaneously running user `theoh`'s single-node job 2843 (MoE EP=8 training) successfully at 10:30 with no failure. So slinky-9's failure mode is **multi-node IB-related, not single-node GPU/compute**. Suggests fabric/SHARP-daemon-state issue, not individual node hardware.

**No node exclusions needed for slinky as of 2026-05-12.** → STALE. Multi-node 256+ GPU jobs need at minimum `--exclude=slinky-9,slinky-14,slinky-18,slinky-26` until ops investigates the fabric.

## Morning healed status (09:00 UTC, before degradation)

**slinky-9 and slinky-26 were HEALED** (jobs 2656/2657, 2026-05-12 ~08:55 UTC):
- slinky-0 + slinky-9 (2n, 8 GiB ring): **716.39 GB/s**, 0 NCCL WARN/ERROR
- slinky-1 + slinky-26 (2n, 8 GiB ring): **688.46 GB/s**, 0 NCCL WARN/ERROR
- The all-64n SHARP allreduce (job 2644, 534 GB/s) and the 9-test 64n matrix completed on all 64 nodes with no failures.
- → Health was VERIFIED at 09:00 then DEGRADED by 10:00. Cluster IB stability is short-lived on slinky.

### Historical degradation (2026-05-11, now resolved)
slinky-9 and slinky-26 failed pair-sweep 2n tests at 12:02 UTC after the llama405b_nvfp4 @ 512 NODE_FAIL at 12:01.
- slinky-9 was rank-0 of an earlier (09:54) nemotronh_fp8 TCPStore bootstrap failure — likely degraded gradually from sweep start.
- A third unidentified node hit pyxis container failures at 12:24 during the llama405b_nvfp4 fallback.
- Both have since recovered (likely after node reboots / pod cycling overnight).

**Stale-as-of-start-of-day-cluster bad nodes (now stale by lunch)**: previously slinky-43 and slinky-62 were the only flagged bad nodes; today's pre-sweep verification confirmed those two are healed.

**slinky-43 and slinky-62 are HEALED** (jobs 2410/2411, 2026-05-11):
- slinky-0 + slinky-43 (2n, 8 GiB ring): **716.16 GB/s**, no `ibv_reg_mr` errors
- slinky-1 + slinky-62 (2n, 8 GiB ring): **713.73 GB/s**, no `ibv_reg_mr` errors
- Full 64-node ring allreduce 32 GiB (job 2413, no exclusions): **388.50 GB/s** — better than the 60n baseline (367.36) and matches the 32n flat-scaling line (388.30). All 64 nodes pass together.
- **No need for `--exclude=slinky-43,slinky-62` anymore.** Re-add only if specific failures resurface.

### Historical (2026-05-09, now obsolete)
- slinky-43, slinky-62 were failing with `ibv_reg_mr_iova2 failed` / `ibv_create_qp ... Cannot allocate memory` in 2-node tests.
- Workaround was `--exclude=slinky-43,slinky-62`; 60n with exclusion got 367.36 GB/s (job 1992).
- "20 broken nodes" claim from 2026-05-04 was already retracted on 2026-05-09 (TCP-OOB launcher confusion, not hardware).

**Earlier "20 broken nodes" claim from 2026-05-04 is OUTDATED.** Re-tested 2026-05-09: 60-node run including all of those alleged broken nodes (except slinky-43, slinky-62) PASSED at 367 GB/s. The earlier diagnosis conflated TCP-OOB launcher issues with hardware problems.

**Benchmark partition**: `exeamplar-benchmark` (32 nodes: healthy + broken subset; distinct from full `slinky`/`all` partition)

**Pair sweep script**: `/data/home/johnson/nccl-results/run_pair_sweep.sh`
- Submits 16 disjoint 2-node pairs in parallel; 3-min timeout per pair; ring all-reduce 8G+16G
- Results: `/data/home/johnson/nccl-results/flapping-airplanes/pair_sweep_<YYYYMMDD_HHMMSS>/`

## 1-Node Baseline (2026-05-01, slinky-0)
Results: `/data/home/johnson/nccl-results/flapping-airplanes/B200/20260501_002644_slurm_1nodes/`
| Test | Ring | NVLS | CollNet/SHARP |
|------|------|------|---------------|
| all_reduce | 693.2 GB/s | **838.7 GB/s** | 693.1 GB/s |
| all_gather | 658.6 GB/s | 657.7 GB/s | 658.5 GB/s |
| reduce_scatter | 689.1 GB/s | 687.1 GB/s | 688.2 GB/s |
Matches use3a-ss baseline (NVLS +21% all_reduce). SHARP inactive at 1-node (expected).

## Baseline Performance Reference (from use3a-ss B200, 2026-04-27)
These are the targets to compare against for a healthy new B200 cluster:
- **1 node**: all_reduce NVLS ~836 GB/s, Ring ~687 GB/s
- **2 nodes**: all_reduce NVLS ~713 GB/s, SHARP ~543 GB/s
- **4+ nodes**: all_reduce SHARP ~542-545 GB/s (flat), Ring ~311 GB/s
- **SHARP enabled at 4+ nodes** gives ~75-88% improvement over ring

**Why:** This is a new cluster with same GPU type (B200) as use3a-ss. The baselines need to be established fresh since cluster topology may differ.
