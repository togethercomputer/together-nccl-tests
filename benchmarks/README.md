# NCCL Benchmarks

Benchmark scripts and results for comparing NCCL communication algorithms
(Ring, NVLS, CollNet/SHARP) across GPU types and cluster backends.

---

## Directory Structure

```
benchmarks/
├── run_k8s.sh            # Benchmark suite — Kubernetes Jobs
├── run_slurm.sh          # Benchmark suite — Slurm sbatch
├── run_mpi.sh            # Benchmark suite — direct MPI (no scheduler)
├── lib/
│   └── common.sh         # Shared: test matrix, GPU detection, summary table
├── run_nccl_k8s.sh       # Single-test utility — Kubernetes
├── run_nccl_native.sh    # Single-test utility — Slurm, native binaries + HPC-X
├── run_nccl_test.sh      # Single-test utility — Slurm, Pyxis/enroot container
├── setup_and_build.sh    # Build nccl-tests from source on shared filesystem
├── B200/results/         # B200 results (auto-created)
└── H100/results/         # H100 results (auto-created)
```

GPU type is auto-detected from `nvidia-smi`. Results are saved to
`benchmarks/<GPU_TYPE>/results/<timestamp>_<backend>/`.

---

## Backend Architecture

MPI is the process coordination layer. K8s and Slurm are the schedulers that
sit beneath it. All three entry scripts run the same 9-test matrix.

```
┌─────────────────────────────────────────────┐
│              MPI (process layer)             │
│   mpirun -np 64 -N 8 all_reduce_perf -g 1   │
└───────────────────┬─────────────────────────┘
                    │ runs on top of
       ┌────────────┼────────────┐
       │            │            │
  ┌────▼────┐  ┌────▼────┐  ┌───▼──────┐
  │  Slurm  │  │   K8s   │  │  Bare    │
  │ sbatch  │  │  Job    │  │  Metal   │
  └─────────┘  └─────────┘  └──────────┘
```

---

## Registry Credentials

All three benchmark scripts need to pull
`togethercomputer/training-performance:nccl-tests-ub22.04-cuda12.9-v0.1`
from Docker Hub. Credentials are picked up automatically — set them once and
every script handles auth before submitting jobs.

### Setup

**Option 1 — `.env` file** (persists across sessions, gitignored):

```bash
cp .env.example .env
# edit .env:
#   DOCKER_USER=johnsontogether
#   DOCKER_TOKEN=dckr_pat_xxxx
#   DOCKER_REGISTRY=docker.io   # optional, default: docker.io
```

**Option 2 — shell export** (current session only):

```bash
export DOCKER_USER=johnsontogether
export DOCKER_TOKEN=dckr_pat_xxxx
```

Variables already in the environment always take precedence over `.env`.
If neither is set, scripts print a reminder with the exact commands above.

### What each script does with the credentials

| Script | Auth action |
|--------|-------------|
| `run_k8s.sh` | Creates / updates the `dockerhub-nccl-tests` imagePullSecret in the target namespace before submitting jobs. Idempotent — safe to run repeatedly. |
| `run_slurm.sh --container` | Runs `docker login` on the local host so enroot can pull the image. No-op in native binary mode. |
| `run_mpi.sh --hostfile` | Runs `docker login` on every node in the hostfile via SSH (token passed over stdin, never exposed in process listings). |

### Create a Docker Hub token

1. Log in to [hub.docker.com](https://hub.docker.com)
2. Account Settings → Security → **New Access Token**
3. Scopes: `Read` (pull only) is sufficient for benchmark runs

---

## Benchmark Entry Scripts

All three share the same test matrix and flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-b/--min` | Min message size | `1M` |
| `-B/--max` | Max message size | `8G` |
| `-f/--factor` | Step factor | `2` |
| `-i/--iters` | Iterations | `20` |
| `-w/--warmup` | Warmup iters | `5` |
| `--baseline DIR` | Baseline dir for comparison | auto |
| `--set-baseline` | Mark this run as new baseline | — |
| `--gpu-type TYPE` | Override GPU type label | auto-detect |
| `--dry-run` | Print commands/YAMLs without running | — |

---

### `run_k8s.sh` — Kubernetes

Submits one `batch/v1 Job` per test config. Streams logs and saves results locally.
Image is pulled from Docker Hub via an `imagePullSecret`.

```bash
# Run with defaults (auto-detects GPU type, uses dockerhub-nccl-tests secret)
./benchmarks/run_k8s.sh

# Custom image or namespace
./benchmarks/run_k8s.sh --image myrepo/nccl-tests:latest --namespace prod

# Inspect generated YAMLs without submitting
./benchmarks/run_k8s.sh --dry-run

# Mark this run as the new baseline
./benchmarks/run_k8s.sh --set-baseline
```

Additional flags: `--image`, `--namespace`, `--pull-secret`, `--ttl`

---

### `run_slurm.sh` — Slurm

Generates and submits one `sbatch` job per test config (`sbatch --wait`).
Supports native binaries or Pyxis/enroot containers.

```bash
# Single node
./benchmarks/run_slurm.sh --node gpu-node-1

# Explicit node list
./benchmarks/run_slurm.sh --nodes gpu-node-1,gpu-node-2

# All nodes in partition
./benchmarks/run_slurm.sh --all --partition gpu

# All nodes except specified
./benchmarks/run_slurm.sh --exclude gpu-node-bad

# With Pyxis container
./benchmarks/run_slurm.sh --nodes gpu-node-1,gpu-node-2 \
    --container /path/to/nccl-tests.sqsh

# With HPC-X MPI (native binary path on shared fs)
./benchmarks/run_slurm.sh --nodes gpu-node-1,gpu-node-2 \
    --binary-path /mnt/vast/nccl-tests/build \
    --hpcx /opt/hpcx/hpcx-init.sh

# Dry-run — print sbatch scripts without submitting
./benchmarks/run_slurm.sh --nodes gpu-node-1 --dry-run
```

Additional flags: `--partition`, `--account`, `--time`, `--binary-path`,
`--container`, `--hpcx`, `--gpus-per-node`

---

### `run_mpi.sh` — Direct MPI

Runs `mpirun` directly on the local host or across nodes via a hostfile.
No job scheduler required — requires passwordless SSH to all worker nodes.

```bash
# Single node (8 GPUs, localhost)
./benchmarks/run_mpi.sh

# Multi-node with hostfile
./benchmarks/run_mpi.sh --hostfile hosts.txt

# Dry-run — print mpirun commands without running
./benchmarks/run_mpi.sh --hostfile hosts.txt --dry-run
```

Hostfile format (OpenMPI):
```
node1 slots=8
node2 slots=8
```

The script checks SSH connectivity to all hosts before running.

Additional flags: `--hostfile`, `--gpus-per-node`, `--mpirun-extra`

---

## Test Matrix

9 tests run per benchmark: 3 collective operations × 3 NCCL algorithm configs.

| Config | `NCCL_ALGO` | `NCCL_NVLS_ENABLE` | `NCCL_COLLNET_ENABLE` | Description |
|--------|-------------|--------------------|-----------------------|-------------|
| `ring` | `RING` | 0 | 0 | Ring allreduce — cluster default |
| `nvls` | (unset) | 1 | 0 | NVSwitch SHARP — intra-node NVLink reduction |
| `collnet_sharp` | (unset) | 0 | 1 | IB SHARP — inter-node InfiniBand reduction |

Operations: `all_reduce`, `all_gather`, `reduce_scatter`

> **Note:** For non-ring configs, `NCCL_ALGO` is explicitly unset in case the
> cluster sets `NCCL_ALGO=RING` globally via `/etc/environment`.

---

## Baseline Comparison

Each backend maintains its own baseline symlink:

```
B200/results/baseline_k8s   -> 20260409_185746_k8s
B200/results/baseline_mpi   -> (set with --set-baseline)
B200/results/baseline_slurm -> (set with --set-baseline)
```

When a baseline exists the summary table gains comparison columns:

```
Test                   Config             Baseline(GB/s)  Current(GB/s) Delta(GB/s) Delta(%)
----                   ------             -------------   ------------  ----------  --------
all_reduce             nvls                       842.96         843.53      +0.57    +0.1%
all_reduce             ring                       684.87         685.35      +0.48    +0.1%
...
```

Set a new baseline after a clean run:
```bash
./benchmarks/run_k8s.sh --set-baseline
```

---

## Results: B200

**Node**: 1× K8s worker, 8× B200 GPUs  
**Image**: `togethercomputer/training-performance:nccl-tests-ub22.04-cuda12.9-v0.1`  
**NCCL**: 2.26.5+cuda12.9 | **CUDA**: 12.9  
**Test**: float32 sum, 1M–8G, 5 warmup + 20 iters  
**Results**: `B200/results/20260409_201511_k8s/` (baseline: `20260409_185746_k8s`)

### 1-Node (8 GPUs)

| Test | Config | Peak busBW (GB/s) |
|------|--------|:-----------------:|
| all_reduce | **nvls** | **843.53** |
| all_reduce | ring | 685.35 |
| all_reduce | collnet_sharp | 684.89 |
| all_gather | **nvls** | **662.67** |
| all_gather | ring | 660.48 |
| all_gather | collnet_sharp | 660.62 |
| reduce_scatter | **nvls** | **685.91** |
| reduce_scatter | ring | 684.93 |
| reduce_scatter | collnet_sharp | 684.56 |

### Key Findings

- **NVLS gives +23% on all_reduce** (843 vs 685 GB/s). NVSwitch SHARP reduces
  all 8 GPUs locally within the NVSwitch fabric before any inter-node traffic.
- **all_gather and reduce_scatter show minimal NVLS gain** (~0.3%) on single node —
  expected, as these ops don't benefit from the same NVSwitch optimization path.
- **CollNet/SHARP has no effect** — IB SHARP is an inter-node optimization;
  single-node results are identical to ring.

---

## Results: H100 80GB HBM3

**Cluster**: 2× nodes (gpu-dp-2jvzl-pqq7p, gpu-dp-2jvzl-ddjjs), 8× H100 per node  
**NCCL**: 2.28.3+cuda13.0 | **Driver**: 570.195.03  
**Test**: AllReduce, float32 sum, 1M–8G, 5 warmup + 20 iters

### 1-Node (8 GPUs)

| Config | Peak busBW (GB/s) | vs Ring |
|--------|:-----------------:|:-------:|
| ring | 368.55 | baseline |
| **nvls** | **477.80** | **+29.7%** |
| collnet_sharp | 367.95 | ~0% |
| nvls_collnet | 478.72 | +29.9% |

### 2-Node (16 GPUs)

| Config | Peak busBW (GB/s) | vs Ring |
|--------|:-----------------:|:-------:|
| ring | 368.24 | baseline |
| **nvls** | **479.81** | **+30.3%** |
| collnet_sharp | 368.27 | ~0% |
| nvls_collnet | 479.03 | +30.1% |

### Key Findings

- **NVLS gives +30%** on both 1-node and 2-node. NVSwitch reduces all 8 GPUs
  locally before transmitting over IB, cutting inter-node traffic significantly.
- **IB SHARP (CollNet) is inactive** on this cluster. The SHARP plugin
  (`libnccl-net.so` v10) is present but does not support `float32` AllReduce.
- **Recommendation**: set `NCCL_NVLS_ENABLE=1` and leave `NCCL_ALGO` unset.
