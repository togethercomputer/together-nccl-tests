---
name: sharp-active-on-slinky-2026-05-12
description: "SHARP/CollNet offload verified working at 64n on slinky as of 2026-05-12. Reaches 534 GB/s at 32 GiB allreduce, within 1.4% of use3a-ss reference. Supersedes \"SHARP inactive\" notes in [[slinky-cluster-environment]] and [[nccl-multi-node-working-config]]."
metadata: 
  node_type: memory
  type: project
  originSessionId: 37fcaaff-3bd6-46be-b13d-31ec2edca995
---

## Verified result — job 2644, 2026-05-12 08:32 UTC, 64 nodes / 512 GPUs

**State change:** SHARP was inactive on 2026-05-11 (job 2412: 4n CollNet got 390 GB/s = ring fallback). On 2026-05-12 it works end-to-end at 64n with real GDRDMA SHARP offload.

| Size | busbw (GB/s) |
|---|---|
| 4 GiB | 523.18 |
| 8 GiB | 521.88 |
| 16 GiB | 533.96 |
| **32 GiB** | **534.44** |
| Avg (1..32G) | 486.94 |

- 0 NCCL WARN/ERROR, 0 SHARP errors
- 24,576 `COLLNET/SHARP/.../GDRDMA` init markers in log — confirms real offload
- 32 GiB result is within 1.4% of use3a-ss SHARP baseline (542.1 GB/s)
- vs ring 64n on same cluster (388.5 GB/s, 2026-05-11): **+37.6%**

## Working sbatch
`/home/johnson/nccl-ar-sharp-64n-b200.sbatch` — combines base [[nccl-multi-node-working-config]] env with SHARP enablement.

**SHARP-specific env vars that worked:**
```bash
# For all_reduce only:
export NCCL_ALGO=CollnetChain  # CollnetChain supports ONLY allreduce
export NCCL_NVLS_ENABLE=1
export NCCL_COLLNET_ENABLE=1

# For all_gather / reduce_scatter / mixed workloads:
unset NCCL_ALGO  # let NCCL auto-pick CollnetDirect (or whatever fits)
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=1

# Common:
export SHARP_COLL_ENABLE_SAT=1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

**Gotcha:** Forcing `NCCL_ALGO=CollnetChain` for all_gather or reduce_scatter aborts with
`NCCL WARN Error : no algorithm/protocol available for function AllGather/ReduceScatter with datatype ncclInt8. NCCL_ALGO was set to CollnetChain.` — CollnetChain is allreduce-only. Unset NCCL_ALGO for those collectives.

All base config (hcoll/ucc disabled, srun --mpi=pmix, 12 IB HCAs, etc.) still required per [[nccl-multi-node-working-config]].

## 64n full matrix (2026-05-12, jobs 2644-2655), 32 GiB busbw OOP GB/s
| Test           | Ring  | NVLS  | SHARP | Winner            |
|----------------|-------|-------|-------|-------------------|
| all_reduce     | 388.6 | 389.0 | **534.4** | SHARP +37.6%   |
| all_gather     | 381.6 | 379.9 | 383.6 | tie (IB-bound)    |
| reduce_scatter | 388.8 | 384.6 | 387.8 | tie (IB-bound)    |

SHARP wins **only** for all_reduce. all_gather and reduce_scatter have no reduction step for the SHARP AM to offload, so all three algos converge at IB bandwidth limit (~385 GB/s). Matches use3a-ss behavior.

Results dir: `/data/home/johnson/nccl-results/flapping-airplanes/B200/20260512_084003_slurm_64nodes/`

## Result log
`/home/johnson/nccl-ar-sharp-64n-2644.log`

**Why:** Cluster operations enabled the SHARP daemon between 2026-05-11 evening and 2026-05-12 morning. The `NCCL_ALGO=CollnetChain` explicit selection (vs leaving algo unset and relying on NCCL_COLLNET_ENABLE alone) is what user supplied and what was tested.

**How to apply:** Use this config when running any large-message allreduce at 4+ nodes on slinky. For tensor-parallel or smaller messages, NVLS may still win (NVLS_ENABLE=1 is already set; NCCL picks per-size). Re-verify if SHARP daemon state changes — slinky has shown SHARP can come/go.

## SHARP env HURTS FSDP-heavy LLM workloads (2026-05-12 finding)

**Result:** llama70b_fp8 @ 512 GPU with SHARP env (NCCL_COLLNET_ENABLE=1, NCCL_NVLS_ENABLE=1, NCCL_ALGO unset) → **1412.2 TFLOPS/GPU** (job 2740). vs 2026-05-11 ring-only at same scale: **1614.1 TFLOPS/GPU**. Net: **-12.5%**.

**Why:** llama70b FP8 uses TP=2, PP=4 with FSDP (DP=64). Dominant collectives are FSDP all_gather + reduce_scatter — NOT allreduce. SHARP only offloads allreduce; for all_gather/reduce_scatter NCCL still picks CollnetDirect or falls back to IB ring, and the CollNet path setup (tree allocation, GDRDMA buffers) costs overhead without giving any speedup.

**How to apply:** Apply SHARP env **per-workload**, not globally:
- ✅ SHARP on: allreduce-bound workloads (TP+PP+CP with DP DPN multi-node, esp. Llama 405B FP8)
- ❌ SHARP off: FSDP-heavy workloads (Llama 70B at any dtype, anything where DP > PP×TP×CP)
- Rule of thumb: if NCCL_DEBUG logs show allreduce as dominant collective, keep SHARP. If allgather/reduce_scatter dominant, set NCCL_COLLNET_ENABLE=0.

**Evidence:** Single data point so far (one workload, one comparison). Validate on more FSDP workloads before treating as universal rule. See [[2026-05-12-512gpu-sweep-aborted]] for full incident context.
