---
name: slinky LLMB-run / pyxis container workaround stack (2026-05-10)
description: Patch stack that makes llmb-run training jobs work on slinky cluster despite K8s pod RLIMIT_MEMLOCK=8KB, pyxis path perms, pod cycling, and Triton/ptxas-blackwell missing.
type: project
originSessionId: 05b58e05-7df9-496b-aeaf-eaa70c54af83
---
## Verified results (2026-05-10)
- Job 2143: **Llama 70B FP8 mx @ 256 GPU (32 nodes), pretrain_llama3.1**: COMPLETED 10/10 iters in 6:05, **1794 MODEL_TFLOPS/GPU mean iter 3-9** (+19% vs MD1, +10% vs target 1624). Required NCCL preflight (see below).
- Job 2139: Llama 70B FP8 cs @ 128 GPU (16 nodes): COMPLETED 10/10 iters in 6:45, 1584 MODEL_TFLOPS/GPU.
- Job 2116: Llama 70B FP8 cs @ 128 GPU (16 nodes), 50 iters: 1582 MODEL_TFLOPS/GPU.

## Auto-sweep orchestrator (2026-05-10 06:02 UTC, in progress)
- Script: `/home/johnson/auto_sweep_256gpu.sh`
- Pattern: preflight (32-node NCCL all_reduce, ~30s) → if busbw extracted, submit LLM 256GPU MAX_STEPS=10 → wait + parse mean iter 3-9 MODEL_TFLOPS → next workload.
- 13 workloads queued (skipping nemotron4-15b: not installed; skipping nemotron-h-56b: container missing ptxas-blackwell).
- Results TSV: `/home/johnson/auto_sweep_results.tsv`
- Log: `/home/johnson/auto_sweep_logs/run.log` and `orchestrator.log`

### Sweep results (rolling, 2026-05-10)
| # | Workload | mean iter 3-9 (TFLOPS/GPU) | Job | Wall | Notes |
|---|---|---|---|---|---|
| 1 | llama70b_fp8 | 1831.0 | 2145 | 6:03 | +12.7% target 1624, +21.8% MD1 1503 |
| 2 | llama70b_nvfp4 | 2062.6 | 2147 | 5:31 | +2.5% target 2013 |
| 3 | llama405b_fp8 | 1022.9 | 2149 | 28:25 | −40.6% target — comm-bound, no SHARP |
| 4 | llama405b_nvfp4 | NODE_FAIL | 2151 | 2:06 | retry queued at end of v2 list |
| 5 | n4340b_bf16 | 853.1 | 2159 | 4:50 | NeMo TFLOPS_per_GPU pattern (mean iter 3-9) |
| 6 | dsv3_fp8 (2160) | FAILED@15:24 | 2160 | 15:24 | NCCL collective timeout in PIPELINE_MODEL_PARALLEL_GROUP. Originally misdiagnosed as DeepEP-specific; actually a deterministic PP deadlock (see below). |
| 7 | dsv3_bf16 (2163) | FAILED@14:52 | 2163 | 14:52 | **ptxas-blackwell missing** in container 26.02.01 (torch.compile/Inductor needs it for B200 SM100 codegen) |
| 8 | qwen3_bf16 | 478.9 | 2167 | 16:35 | NeMo path, no issues |
| 9 | nemotronh_fp8 | **1116.9** | 2174 | 6:36 | **Container 26.02.01 NO LONGER blocked by ptxas-blackwell** — runs fine. Memory's earlier "known broken" claim was wrong. |
| 10 | n4340b_fp8 | **1096.4** (orch parsed 141.6 wrong) | 2165 | 25:12 | Trained 10/10 OK; orch parser failed on scientific notation `1.142e+03`; slurm wall hit. Actual mean: 1096.4. |
| 11 | dsv3_fp8 (retry 2214) | FAILED@15:12 | 2214 | 15:12 | Same PP deadlock as 2160 — confirms deterministic |
| 12 | dsv3_bf16 (retry 2314) | FAILED@13:31 | 2314 | 13:31 | ptxas fix worked (no FileNotFoundError, reached "Starting training loop"), then SAME PP timeout as fp8. **Deadlock is NOT fp8/DeepEP-specific.** |
| 13 | grok1_fp8/bf16 | UNRUNNABLE | 2170, 2172, 2218, 2220, 2222, 2316, 2318 | varies | **No compatible container exists.** 25.09.00: OMPI MPI_Init via OPAL→pmix3x_client → Unreachable. 25.07.01: fixes MPI but TypeError `DistributedDataParallelConfig.__init__() got unexpected kwarg 'keep_fp8_transpose_cache'` — recipe newer than NeMo's API. 26.02.01: no MPI issue but TypeError `no_weight_decay_cond` removed. The grok1 recipe in NeMo HEAD wants APIs not present in any released container. **Skip grok1 from sweep until NVIDIA ships a compatible container.** |
| 14 | llama405b_nvfp4 | NODE_FAIL ×3 | 2151, 2176, 2224 | ~2:00 each | Always NODE_FAIL early. Possibly intrinsic to nvfp4 405B at 256 GPU or persistent cluster issue with this specific config. |

## CRITICAL FINDING: dsv3 DDP grad-sync hang at 256 GPU (NOT a PP deadlock)
Earlier diagnosis was wrong. Job 2324 (TP=2/PP=8/EP=8/DP=16) reproduced the SAME ~13 min failure as the original TP=1/PP=16/EP=8 — proving it's not PP-specific. The actual hang is in `reduce_scatter_tensor_coalesced` from `start_grad_sync` (Megatron-Bridge `param_and_grad_buffer.py` line 433, distributed-optimizer's DP grad sync). Failure mode: `RuntimeError: NCCL Error 2: unhandled system error` — typically network/IB-related, not a deadlock per se. May be K8s pod cycling / IB-MR failure under sustained DP load.

**Tested parallelism configs (all hang at ~13 min mark):**
- TP=1/PP=16/EP=8/DP=2 (default): jobs 2160, 2163, 2214, 2314 — FAILED at 13-15 min
- TP=2/PP=8/EP=8/DP=16 (alt): job 2324 — FAILED at 13:12 (NCCL Error 2 in DDP reduce_scatter)

**Workarounds attempted and DIDN'T fix it:** retry on different nodes, ptxas-blackwell mount, fresh whitelist, alt parallelism TP=2/PP=8.
**How to inject alt parallelism:** edited `/data/home/johnson/llmb/llmb_repo/deepseek_v3/pretrain/megatron_bridge/launch.sh` to read `EXTRA_OVERRIDES` env var (after the h100/fp8_recipe block, before the run command). Set `export EXTRA_OVERRIDES="-tp 2 -pp 8 -ep 8 -vp None"` then `llmb-run submit ...`.
**To try next:** NCCL_DEBUG=INFO to identify the exact failing rank/connection; or test if dsv3 at 128 GPU works (smaller DP world).

## ptxas-blackwell bind-mount fix (2026-05-10)

Container `nvidia+nemo+26.02.01` is missing `/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas-blackwell`. Host has identical 34.9MB binary (Triton 3.6.0, ptxas built May 27 2025) at `/data/home/johnson/llmb/venvs/venv_*/lib/python3.10/site-packages/triton/backends/nvidia/bin/ptxas-blackwell` — verified runs standalone.

**Fix:** stage at `/data/home/johnson/triton_fixes/ptxas-blackwell` and bind-mount into container at the expected path. One mount line added to `pretrain_deepseek-v3/Megatron-Bridge/scripts/performance/utils/executors.py` line 86:
```python
mounts.append("/data/home/johnson/triton_fixes/ptxas-blackwell:/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas-blackwell")  # slinky: container 26.02.01 missing this
```
**How to apply:** for nemotron-h and qwen3 (if Inductor is hit), add the same line to their respective `executors.py`. Same fix pattern.

### Workloads that will hit ptxas-blackwell blocker (container 26.02.01)
- nemotron-h (Mamba/Triton)
- deepseek-v3 BF16 (torch.compile/Inductor — Megatron-Bridge route)
- Possibly qwen3 BF16/FP8 (TBD — depends on whether their Megatron-Bridge config hits Inductor)

### Workloads that sidestep ptxas-blackwell (TransformerEngine / NeMo)
- llama70b/405b (TransformerEngine, no Inductor)
- nemotron4-340b (NeMo, no Inductor)
- grok1 (NeMo, expected)

### Parser pattern: TWO log formats
- **Megatron-Bridge** (llama, dsv3, qwen3, nemotron-h): `GPU utilization: NNN.N MODEL_TFLOP/s/GPU`. Mean = `awk 'NR>=3 && NR<=9'` of matches.
- **NeMo** (grok1, nemotron4-340b): `TFLOPS_per_GPU: NNN.N`. Mean = `awk 'NR>=4 && NR<=10'` (iter index is 0-based, so iter 3 = 4th match).
Orchestrator `wait_and_parse()` tries Megatron first, falls back to NeMo.

### Bug recovery (2026-05-10)
- 5 workloads (deepseek-v3, qwen3, gpt_oss, grok1, nemotron4-340b) had Python SyntaxError in patched executors.py: bulk patcher inserted literal newline inside multi-line string concatenation. Fix: replace literal `'\n"` block with escaped `'\\n"`. Validate with `python3 -c "import ast; ast.parse(open(f).read())"`.

## CRITICAL: 32-node NCCL preflight before any 256-GPU LLM submit
At 32-node scale, 9/9 naïve `llmb-run submit ... --scale 256` attempts failed at NCCL P2P setup with `ncclSystemError` (random K8s pod IB-MR failures). Solution: **always do a 32-node NCCL `all_reduce_perf` preflight (~30s) on the candidate node set immediately before LLM submission**, on the SAME nodelist. Bad pods are not deterministic across hours.

Reproducer at `/home/johnson/preflight_then_llm.sh` — runs 32-node NCCL allreduce, parses `Avg bus bandwidth`, and only submits LLM if preflight returned a busbw value.

## Why: structural blockers on slinky
1. K8s pod hard `RLIMIT_MEMLOCK=8192` (8 KB). Even `sudo prlimit` + `slurm --propagate=MEMLOCK` doesn't fully bypass — PyTorch's default `file_descriptor` shm strategy in `_new_using_fd_cpu` hits EAGAIN.
2. `/usr/share/enroot/enroot-data` is `root:root 0755` on each pod after restart — pyxis can't extract squashfs.
3. Pyxis ignores `ENROOT_DATA_PATH` env var (uses `/etc/enroot/enroot.conf` static value).
4. K8s pod cycling/eviction kills random nodes mid-job (NCCL `ncclRemoteError`, "remote process exited prematurely").
5. Nemo container `26.02.01` is missing `triton/backends/nvidia/bin/ptxas-blackwell` — breaks Mamba/Triton kernels (Nemotron-H). Llama (TransformerEngine) sidesteps this.

## The patch stack (all in `executors.py` + `run_script.py`)

### 1. `set_sharing_strategy('file_system')` — THE critical fix
At top of `run_script.py` (BEFORE any DataLoader instantiation):
```python
import torch.multiprocessing as _mp
try:
    _mp.set_sharing_strategy('file_system')
except Exception:
    pass
```
Switches PyTorch tensor sharing from shm-fd (which hits RLIMIT_MEMLOCK) to mkstemp in `/tmp` (no memlock needed).

### 2. `slurm_executor()` (in `Megatron-Bridge/scripts/performance/utils/executors.py`)
- `mounts.append("/dev/shm:/dev/shm")` — bind host's 16GB tmpfs.
- `srun_args` adds: `--propagate=MEMLOCK,STACK,NOFILE`, `--container-remap-root`, `--container-writable`.
- `setup_lines`: pre-srun chmod + sudo prlimit on the sbatch shell:
  ```
  srun --ntasks-per-node=1 --nodes=$SLURM_JOB_NUM_NODES bash -c 'sudo -n chmod 1777 /usr/share/enroot/enroot-data 2>/dev/null || true; sudo -n prlimit --memlock=unlimited:unlimited --pid=$$ 2>/dev/null || true'
  sudo -n prlimit --memlock=unlimited:unlimited --pid=$$ 2>/dev/null || true
  ```
- `custom_bash_cmds.insert(0, "ulimit -l unlimited || true; ulimit -n 524288 || true")` — pre_cmds (DO NOT use `2>/dev/null` — Jinja escapes `>` to `&gt;`).

### 3. Launch wrapper (`/data/home/johnson/llmb/llmb_repo/<model>/launch.sh`)
Adds `--custom_env_vars=ENROOT_DATA_PATH=...,ENROOT_CACHE_PATH=...,ENROOT_TEMP_PATH=...` so the host's enroot redirect env vars propagate to sbatch (NeMo Run filters env by default). **Note**: must use `=` form for `--custom_srun_args=--prolog=...`, not space, or argparse eats the value.

### 4. Slurm submission flags (set as environment before `llmb-run submit`)
```bash
export ADDITIONAL_SLURM_PARAMS="exclude=slinky-43,slinky-54,slinky-62"
# (slinky-29 is also frequently missing — drop from exclude on the day if it's down)
```
Format is `key=value` (no `--` prefix) per setup_experiment.py's `parse_additional_slurm_params`.

### 5. Choose Llama (not Nemotron-H) for B200
Nemotron-H 56B uses Mamba/Triton kernels, container missing `ptxas-blackwell`. Llama 70B uses TransformerEngine and works.

### 6. Choose ≥16 nodes for Llama 70B
At 8 nodes, FSDP all_gather/reduce_scatter is heavy enough that pod-cycling eviction kills the run before training starts (~3:58 mark). At 16 nodes, smaller per-rank FSDP shards finish setup before any pod cycles.

## Reproduce
```bash
export LLMB_INSTALL=/data/home/johnson/llmb GPU_TYPE=b200 SBATCH_ACCOUNT=root SBATCH_PARTITION=all
export ENROOT_DATA_PATH=/data/home/johnson/.local/share/enroot
export ENROOT_CACHE_PATH=/data/home/johnson/.cache/enroot
export ENROOT_TEMP_PATH=/data/home/johnson/.cache/enroot/tmp
export ADDITIONAL_SLURM_PARAMS="exclude=slinky-43,slinky-54,slinky-62"  # add slinky-29 if missing today
source /data/home/johnson/llmb_venv/bin/activate
cd /data/home/johnson/llmb
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 128
```

## Files modified (host-side only — patches applied to mounted paths)
- `/data/home/johnson/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/utils/executors.py`
- `/data/home/johnson/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/perf_plugins.py`
- `/data/home/johnson/llmb/workloads/pretrain_llama3.1/Megatron-Bridge/scripts/performance/run_script.py`
- (analogous files under `/data/home/johnson/llmb/workloads/pretrain_nemotron-h/...` — Nemotron-H still blocked by ptxas-blackwell, but pyxis/shm path is fixed)
- `/data/home/johnson/llmb/llmb_repo/nemotron-h/launch.sh` (env propagation)
- `/home/johnson/enroot_chmod_prolog.sh` (host helper used by `--prolog`)

**How to apply:** When user reinstalls llmb workloads or upgrades container, RE-APPLY this patch stack to the new install paths. The container itself doesn't need patching — patches live on host and bind-mount in.

## Open issues for admin (not blocking, but ideal fix)
1. Raise pod `RLIMIT_MEMLOCK` to unlimited (matches MD1 cluster).
2. Update `/etc/enroot/enroot.conf` `ENROOT_DATA_PATH` to a user-writable path.
3. Add `ptxas-blackwell` to next Nemo container (`26.02.02`+) for Nemotron-H/Mamba kernels.
4. Investigate K8s pod-cycling eviction policy — single-node death kills 70B FSDP training during init.
