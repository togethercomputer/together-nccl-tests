---
name: feedback-nccl-multinode-hang-debug
description: "Diagnostic ladder for \"NCCL hangs at N nodes when it worked at N/2\". Work the steps in order — most hangs come from the first 3 causes."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 40ae4eee-9e1c-40ba-92a1-c2fca3af82b8
---

When a multi-node NCCL run hangs, work this ladder in order. Most real hangs come from steps 1-3; don't skip them.

## Step 1 — Get the stuck rank's last line

```bash
export NCCL_DEBUG=INFO
```

Re-submit, then check the per-rank logs for the last operation logged. The phase of the hang tells you which sub-system to look at:

| Phase logged | Hang location | Go to step |
|---|---|---|
| Before any NCCL output | `MPI_Init` | 2 |
| `ncclCommInitRank` started, then silence | NCCL bootstrap | 3 |
| `IB` or `Channel` lines, then silence | HCA / IB fabric | 4 |
| `comm.cuda.allreduce` started, then silence after first iter | Straggler / IB fabric fault | 5 |
| SHARP-specific lines (`sharp_job_id`) | SHARP unavailable | 6 |

## Step 2 — MPI_Init hang (hcoll/ucc collectives)

HPC-X enables hcoll/ucc collectives by default. They wedge `MPI_Init` at scale on most clusters we've seen.

```bash
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll=^hcoll,ucc
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=eth0      # ← match actual NIC
```

If still hung in `MPI_Init`:
- Switch from `mpirun` to `srun --mpi=pmix`. mpirun ORTE OOB hangs at >=12 nodes on slinky.
- Do NOT switch to `--mpi=pmi2` — HPCX OpenMPI 4.1.x usually lacks PMI2 build (`OPAL ERROR: Unreachable in ext3x_client.c`).

## Step 3 — NCCL bootstrap hang

NCCL needs to discover peers over a working IP network before it touches IB.

- `NCCL_SOCKET_IFNAME` set to a NIC that exists and is up? (`ip -br link`)
- `OMPI_MCA_btl_tcp_if_include` set to the same?
- Firewall / network ACL between nodes on the bootstrap port? (rare on Slurm clusters, common on k8s)

Quick test: 2 nodes, single GPU each, `all_reduce_perf -b 1M -e 1M -g 1`. If this works but 8-node doesn't, it's a bootstrap-fan-out / k8s networking issue, not NCCL itself.

## Step 4 — HCA misselection

Symptom: `ncclCommInitRank` started, all peers eventually print "Channel" lines but the run never returns.

```bash
# First, try without NCCL_IB_HCA at all — let NCCL auto-detect
unset NCCL_IB_HCA

# If auto-detect picks too many HCAs (NVSwitch internal HCAs on NVL nodes), list explicitly:
export NCCL_IB_HCA="=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1"
```

On B200 slinky: mlx5_0-3 are NVSwitch (break SHARP). On other clusters, identify per the recipe in [[feedback-new-cluster-adaptation]].

## Step 5 — Straggler / IB fabric fault

Symptom: first iteration completes, then hang. Or: bandwidth drops well below baseline at large scale.

```bash
cd ~/together-nccl-tests/stragglers
python3 find_stragglers.py --partition <name>
```

Exclude bad nodes from the next run: `#SBATCH --exclude=<comma-separated>`.

**Important:** short NCCL tests can pass while the cluster's IB fabric still has a multi-node fault that only shows up under sustained LLM training load. If straggler test passes but LLM training keeps NODE_FAILing, fall back to manual node-by-node exclusion based on per-job failure attribution (see `~/.claude/projects/-data-home-johnson/memory/project_slinky_cluster.md`).

## Step 6 — SHARP-specific hang

Symptom: ring all_reduce works fine, but `NCCL_COLLNET_ENABLE=1` hangs or runs at ring speed.

- Verify SHARP is provisioned on this cluster (`sharp_am` daemon running on aggregation nodes).
- HCA list must exclude NVSwitch internals — see step 4.
- Fall back to `NCCL_COLLNET_ENABLE=0` for the run. Note in the worklog and revisit.
- SHARP can help allreduce but **hurts FSDP-heavy workloads** (commit 6f1234, 2026-05-12 SHARP sweep aborted). Apply per-workload, not globally.

## Things that did NOT work (do not retry without a reason)

- `srun --mpi=pmi2` on HPCX 4.1.x — almost always missing PMI2 build
- `mpirun --map-by ppr:1:node` at >=12 nodes — ORTE OOB hangs
- Leaving NVSwitch-internal HCAs in `NCCL_IB_HCA` — works for ring, breaks SHARP
- Setting `NCCL_DEBUG=TRACE` everywhere — drowns the per-rank logs, makes the hang location harder to find. Use `INFO`.

## See also

- [[project_nccl_working_config]] — the verified working slinky sbatch (with the SHARP-correct 8-HCA list)
- [[feedback-new-cluster-adaptation]] — HCA identification recipe for new hardware
- [[project_straggler_finder]] — current state of find_stragglers.py
