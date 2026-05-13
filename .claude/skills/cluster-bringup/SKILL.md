---
name: cluster-bringup
description: Walk through the bring-up of a new GPU/HPC cluster for LLM training benchmarking — discover hardware/fabric, install tooling, validate NCCL multi-node, detect stragglers, run a first smoke-test sweep. Use when the user starts work on a new cluster, asks "how do I onboard this cluster", references a new partition/site, or says they want to repeat the bring-up they did on a previous cluster.
---

# Cluster bring-up

Use this when starting on a fresh GPU cluster. Reuses tooling from `~/together-nccl-tests` and `~/together-dgxc-benchmarking`. Each phase has a "gather facts" step and an "act" step — gather first, then ask the user before making cluster-wide changes.

## Phase 0 — Establish basics

Ask the user (or check) before running discovery:
- Cluster name (used for `baselines/<cluster>/`, `CLUSTER_NAME`, file naming)
- Scheduler: Slurm, k8s, or both
- Container runtime: pyxis (Slurm), enroot, Docker, or k8s native
- GPU/NIC generation (B200, H200, H100, GB200) — sets expected baselines

## Phase 1 — Discover

Run in parallel where possible. Capture outputs for the worklog.

```bash
# Slurm topology
sinfo -h -o "%P %D %t"                            # partitions, node states
sinfo -N -h -o "%N %t %G"                         # per-node GPU info
scontrol show partition <name>

# Single-node hardware
nvidia-smi -L                                     # GPUs
ip -br link                                       # NICs (look for eth0/bond0/ens*)
ibstat                                            # IB ports + rate + state
ibv_devinfo                                       # link_layer: InfiniBand vs Ethernet (RoCE)
ls /opt/hpcx /var/tmp 2>/dev/null | grep -i hpcx  # HPCX install location
df -h | grep -E 'weka|vast|lustre|gpfs'           # parallel filesystems
```

**HCA identification** (critical for `NCCL_IB_HCA` later):
- `link_layer: InfiniBand` → RDMA-capable, IB fabric
- `link_layer: Ethernet` → RoCE, only usable if explicitly configured
- On NVL-class nodes (B200/GB200), several mlx5_NN are typically NVSwitch internal — exclude them. Verify by running NCCL with each excluded subset.

See [[feedback_new_cluster_adaptation]] for the full list of constants to change.

## Phase 2 — Install

```bash
# nccl-tests
cd ~ && git clone https://github.com/togethercomputer/together-nccl-tests
cd together-nccl-tests && bash benchmarks/setup_and_build.sh

# dgxc-benchmarking
cd ~ && git clone https://github.com/togethercomputer/together-dgxc-benchmarking
cd together-dgxc-benchmarking
export LLMB_INSTALL=/data/home/$USER/llmb            # adjust path
bash install.sh                                      # install all workloads
```

Common install hazards (check `together-dgxc-benchmarking/worklog/2026-05-01_dgxc_installation.md` for the full list with fixes):
- `git lfs` missing → install static binary
- System Python rejected → uv-managed Python 3.12 to PATH
- `enroot import` timeout for >30GB images → patch `llmb_install/downloads/image.py` `"35"` → `"120"`
- `enroot import` whiteout failure → remove leftover 0-byte `.sqsh`, fix `ENROOT_SQUASH_OPTIONS`
- pip too old in venv → patch `venv_manager.py` to upgrade pip after venv creation
- memlock too low → add `ulimit -l unlimited` to sbatch

## Phase 3 — Validate NCCL multi-node

Start small, scale up. Test plan:

| Step | Nodes | What to check |
|---|---|---|
| 1 | 2 | ring all_reduce 8G runs at expected pairwise rate (B200: ~700+ GB/s) |
| 2 | 8 | ring 32G ≈ single-pair rate / 2 ratio holds |
| 3 | 16 | no hangs in `MPI_Init` or `ncclCommInit` |
| 4 | 32, 64 | busbw stays flat (~388 GB/s for B200 ring) — drop indicates straggler |
| 5 | 4 | SHARP `NCCL_COLLNET_ENABLE=1` — should hit ~530+ GB/s on B200, otherwise SHARP not active |

Use `~/together-nccl-tests/benchmarks/run_slurm.sh` — adapt per [[feedback_new_cluster_adaptation]] (partition, HCA list, NIC).

**If anything hangs**, invoke [[feedback_nccl_multinode_hang_debug]] before continuing.

## Phase 4 — Detect stragglers

```bash
cd ~/together-nccl-tests/stragglers
python3 find_stragglers.py                # 2-round disjoint 2-node ring
# outputs: /data/home/$USER/nccl-results/<cluster>/find-stragglers/<ts>/results.json
```

The first run also seeds the baseline JSON in `baselines/<cluster>/healthy_2n_ring.json`. See [[project_straggler_finder]] for constants to change for the new cluster, and [[feedback_new_cluster_adaptation]] for the full porting checklist.

Re-run after every cluster maintenance window — bad nodes change.

## Phase 5 — First LLM sweep (smoke test)

Recommended first workload (well-characterized, fast feedback):

```bash
cd ~/together-dgxc-benchmarking
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 64
```

Expected on B200: ~1500-1600 TFLOPS/GPU at 64 GPUs (verified on slinky 2026-05-10).

File the report per [[feedback-worklog-routing]] — for a smoke test, a short note in `together-dgxc-benchmarking/worklog/YYYY-MM-DD_<cluster>_smoke_test.md` is enough.

## After bring-up

- Save a `project_<cluster>_cluster.md` memory with: partition name, node count, healthy node list, bad node inventory, key paths, expected baselines.
- Update `feedback_worklog_routing` if you discovered a new sub-directory pattern.
- If you found a porting gap not covered in [[feedback_new_cluster_adaptation]], update that memory.
