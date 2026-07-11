---
name: feedback-new-cluster-adaptation
description: "Porting checklist — what constants to change in find_stragglers.py, run_slurm.sh, and sbatch templates when adapting tooling to a new GPU cluster. Includes HCA identification recipe."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 40ae4eee-9e1c-40ba-92a1-c2fca3af82b8
---

When tooling that was developed on cluster A is run on cluster B, these are the things that ALWAYS need to change. Skip none — silent failures are common when a constant is left over from the old cluster.

## In `~/together-nccl-tests/stragglers/find_stragglers.py`

| Constant | Set to |
|---|---|
| `PARTITION` | New cluster's Slurm partition |
| `GPUS_PER_NODE` | Usually 8, but verify with `nvidia-smi -L` |
| `BINARY_PATH` | Path where `all_reduce_perf` lives (per-cluster build) |
| `NCCL_LIB` | LD path for NCCL lib (varies: `/usr/lib/x86_64-linux-gnu` vs cluster install path) |
| `HPCX_INIT` | `/opt/hpcx/hpcx-init.sh` (slinky) vs `/var/tmp/hpcx-X.Y/hpcx-init.sh` (some clusters) |
| `RESULTS_ROOT` | Per-cluster results path under user home or shared FS |
| `CLUSTER_NAME` | Used for `baselines/<cluster>/` — keep consistent with other tools |

In the sbatch body:
- `NCCL_SOCKET_IFNAME`, `UCX_NET_DEVICES`, `OMPI_MCA_btl_tcp_if_include` — set to the actual NIC (`eth0` on slinky, `bond0` on use3a-ss).
- `ulimit -l unlimited` — keep on every cluster, cheap and prevents memlock failures.
- `UCX_TLS=self,sm,cuda_ipc,cuda_copy,tcp` — keep. Avoid IB for MPI bootstrap.
- Do NOT set `NCCL_IB_HCA` in `find_stragglers.py` sbatch — auto-detect is fine for the ring straggler test. (For production `run_slurm.sh`, see below.)

## In `~/together-nccl-tests/benchmarks/run_slurm.sh`

`NCCL_IB_HCA` must list only RDMA-capable HCAs and exclude:
- NVSwitch internal HCAs (NVL-class nodes have several — they break SHARP)
- RoCE HCAs unless explicitly testing RoCE
- HCAs absent on this node generation

**HCA identification recipe** (run on one cluster node):
```bash
ibstat                                        # ports + state + rate
ibv_devinfo                                   # link_layer: InfiniBand vs Ethernet
# Cross-reference: NVSwitch internal HCAs usually:
#   - show as InfiniBand BUT can't talk off-node
#   - have port state=Active but routed only intra-node
#   - on B200 slinky: mlx5_0,1,2,3 are NVSwitch (excluded)
#   - on B200 slinky: mlx5_4-7, 9-12 are the 8 RDMA HCAs
#   - on B200 slinky: mlx5_8 absent, mlx5_13 is RoCE
```

Validate by running 2-node NCCL with the proposed list. Then validate SHARP separately (`NCCL_COLLNET_ENABLE=1`) — if SHARP hangs but ring works, narrow the HCA list further.

**Why this matters:** On flapping-airplanes/slinky, leaving mlx5_0-3 in the HCA list passes ring tests but breaks SHARP entirely (commit 12a864a, 2026-05-13). The reverse — narrowing too aggressively — drops bandwidth. Get this right per cluster.

## In `~/.bashrc` (per-cluster)

- HPCX to PATH if `source /opt/hpcx/hpcx-init.sh && hpcx_load` is needed interactively
- `ENROOT_DATA_PATH`, `ENROOT_RUNTIME_PATH`, `ENROOT_CACHE_PATH` to a persistent FS (WekaFS/VAST) — default `/run` is per-boot
- `LLMB_INSTALL` to a persistent FS (per-user is fine)

## Common install hazards (regardless of cluster)

See `together-dgxc-benchmarking/worklog/2026-05-01_dgxc_installation.md` for the authoritative list with fixes:
- `git lfs` not found → static binary to `~/.local/bin`
- System Python too old → uv-managed Python 3.12
- `enroot import` timeout (35-min default) → patch `llmb_install/downloads/image.py`
- `enroot import` whiteout failure → clear leftover 0-byte sqsh
- `python3-venv` missing → `apt-get install python3.X-venv`
- pip 22.x AssertionError in venv → patch `venv_manager.py` to upgrade pip

## How to apply

When prompted to run anything on a new cluster, FIRST verify these constants match the new cluster — don't assume. Symptoms of stale constants: hangs on first srun, "no such partition", missing binary, NCCL bootstrap timeout. See [[cluster-bringup]] for the broader bring-up workflow, [[project_straggler_finder]] for the current state of the straggler tool.
