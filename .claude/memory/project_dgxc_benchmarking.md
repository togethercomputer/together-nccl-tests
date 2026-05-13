---
name: dgxc-benchmarking installation on slinky cluster
description: NVIDIA dgxc-benchmarking install state, paths, fixes applied, and how to run Llama3.1 benchmarks on the B200 slinky cluster
type: project
originSessionId: 15250214-8077-40e3-8ecd-7b9b7fcbeaed
---
## Installation

- **Repo**: `/data/home/johnson/dgxc-benchmarking` (also a copy at `/data/home/johnson/together-dgxc-benchmarking`)
- **Install root** (`LLMB_INSTALL`): `/data/home/johnson/llmb`
- **Installer venv**: `/data/home/johnson/llmb_venv` (Python 3.12, via uv)
- **Workload venvs**: `/data/home/johnson/llmb/venvs/` (Python 3.10 with upgraded pip)
- **Container images** (`/data/home/johnson/llmb/images/`):
  - `nvidia+nemo+25.07.01.sqsh` (27 GB)
  - `nvidia+nemo+25.09.00.sqsh` (30 GB)
  - `nvidia+nemo+26.02.00.sqsh` (36 GB) ← used for B200
  - `nvidia+nemo+26.02.01.sqsh` (36 GB)
- **Installed workloads**: pretrain_deepseek-v3, pretrain_gpt_oss, pretrain_nemotron-h, pretrain_qwen3, pretrain_grok1, pretrain_llama3.1, pretrain_nemotron4-340b
- **cluster_config.yaml**: generated at `/data/home/johnson/llmb/cluster_config.yaml`
- **Completion date**: 2026-05-01

## Enroot Configuration

Override env vars added to `/home/johnson/.bashrc`:
```bash
export ENROOT_DATA_PATH=/data/home/johnson/.local/share/enroot
export ENROOT_CACHE_PATH=/data/home/johnson/.cache/enroot
export ENROOT_TEMP_PATH=/data/home/johnson/.cache/enroot/tmp
export ENROOT_SQUASH_OPTIONS="-b 262144 -noD -noI"
export LLMB_INSTALL=/data/home/johnson/llmb
export PATH="/home/johnson/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin:$HOME/.local/bin:$PATH"
```

**Why:** ENROOT_DATA_PATH default (`/usr/share/enroot/enroot-data`) is on ephemeral overlay (pod restart = lost). All enroot paths redirected to WekaFS for persistence.

## Fixes Applied to Installer (in llmb_venv site-packages)

1. **`llmb_install/downloads/image.py` line ~207**: Changed srun time limit from `"35"` → `"120"` minutes. **Why:** Large NeMo images (36 GB, 507K files) exceeded 35-min limit during mksquashfs.
2. **`llmb_install/environment/venv_manager.py` after venv creation**: Added pip upgrade step after `python3 -m venv --clear`. **Why:** System pip 22.0.2 has AssertionError bug in resolvelib when resolving nemo-toolkit dependencies; needs pip 23+.

## Running Benchmarks

```bash
source /data/home/johnson/llmb_venv/bin/activate
cd /data/home/johnson/llmb

# Llama3.1 8B — 1 node (8 GPUs), synthetic data, 50 steps
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 8

# Llama3.1 70B — 8 nodes (64 GPUs)
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 64

# Llama3.1 405B — 32 nodes (256 GPUs)
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
```

**B200 FP8 configs (key parallelism):**
| Model | GPUs | TP | PP | CP |
|-------|------|----|----|-----|
| 8B | 8–128 | 1 | 1 | 1 |
| 70B (small) | 64–128 | 1 | 1 | 1 |
| 70B (large) | 256–1024 | 2 | 4 | 1 |
| 405B | 256–1024 | 4 | 8 | 2 |

## Performance Analysis

After job completes, parse results:
```bash
$LLMB_INSTALL/llmb_repo/common/parse_train_timing_mbridge.sh
# Results in $LLMB_INSTALL/workloads/pretrain_llama3.1/experiments/
```

Metric: `TFLOPS_per_GPU` (steps 35–44). MFU = TFLOPS_per_GPU / 4900 (B200 peak FP8).
