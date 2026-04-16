#!/usr/bin/env python3
"""
FA4 per-node benchmark — runs each GPU sequentially within a node.

Benchmarks FlashAttention-4 (CuTeDSL, SM100 Blackwell) forward pass:
  causal=True, hdim=128, seqlen=8192, batch=4, BF16

Output: CSV with node,gpu,time_ms,tflops,mfu
B200 BF16 peak = 2250 TFLOPS.

Launch via sbatch/srun:
  srun -N<nodes> --ntasks-per-node=1 --gpus-per-node=8 python fa4_bench_node.py
"""
import os
import socket

# Redirect all caches to /tmp BEFORE importing torch/triton.
# /home is not accessible on compute nodes.
os.environ["HOME"] = "/tmp"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_home"

import torch

hostname = socket.gethostname()
# Strip common Together AI B200 hostname prefixes
for prefix in ("use3a-ss-b200-", "use3a-ss-h100-"):
    hostname = hostname.replace(prefix, "")
hostname = hostname.replace(".cloud.together.ai", "")

ngpus = torch.cuda.device_count()

# Compile kernels once on GPU 0
torch.cuda.set_device(0)
from flash_attn.cute import flash_attn_func
from triton.testing import do_bench

# Benchmark config
BATCH = 4
SEQLEN = 8192
NHEADS = 16
HEADDIM = 128
DTYPE = torch.bfloat16
CAUSAL = True
PEAK_TFLOPS = 2250.0  # B200 BF16

fwd_flops = 4 * BATCH * SEQLEN**2 * NHEADS * HEADDIM // 2

# Warmup compile on GPU 0
q0 = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device="cuda:0", dtype=DTYPE)
k0 = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device="cuda:0", dtype=DTYPE)
v0 = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device="cuda:0", dtype=DTYPE)
_ = flash_attn_func(q0, k0, v0, causal=CAUSAL)
del q0, k0, v0
torch.cuda.empty_cache()

for gpu_id in range(ngpus):
    torch.cuda.set_device(gpu_id)
    q = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device=f"cuda:{gpu_id}", dtype=DTYPE)
    k = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device=f"cuda:{gpu_id}", dtype=DTYPE)
    v = torch.randn(BATCH, SEQLEN, NHEADS, HEADDIM, device=f"cuda:{gpu_id}", dtype=DTYPE)

    def fwd_fn():
        return flash_attn_func(q, k, v, causal=CAUSAL)

    ms = do_bench(fwd_fn, warmup=10, rep=30) * 1e-3
    tflops = fwd_flops / ms / 1e12
    mfu = tflops / PEAK_TFLOPS * 100

    print(f"{hostname},GPU{gpu_id},{ms*1e3:.3f},{tflops:.0f},{mfu:.1f}%", flush=True)

    del q, k, v
    torch.cuda.empty_cache()
