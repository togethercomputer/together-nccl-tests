# Memory Index

- [flapping-airplanes cluster environment](project_slinky_cluster.md) — 13 of 64 nodes failed during LLM training 2026-05-12 (slinky-40 worst: 6 fails alone). Multi-node IB fabric fault; short NCCL tests don't predict LLM stability.
- [2026-05-11 512-GPU LLM sweep](project_512gpu_sweep.md) — First full-cluster sweep: 8 workloads, 1614 TFLOPS/GPU llama70b_fp8, 88-94% scaling eff, fallback design caught real bad nodes
- [2026-05-12 SHARP sweep ABORTED](project_512gpu_sharp_2026-05-12_aborted.md) — Only 1/8 workloads finished (llama70b_fp8 1412.2 = **-12.5% vs 2026-05-11**); cluster IB collapsed mid-sweep. Lesson: SHARP env hurts FSDP-heavy workloads — apply per-workload, not globally.
- [2026-05-12 256-GPU sweep RECOVERY](project_256gpu_sweep_2026-05-12.md) — Post-SHARP-abort 256-GPU run: 7/8 valid. **N4 340b workloads +24%/+8% unexplained**. Llama 405B NVFP4 first clean run. Auto-exclude-on-NODE_FAIL pattern proved valuable. Nemotronh abandoned (TCPStore).
- [NCCL multi-node working config](project_nccl_working_config.md) — sbatch for 8/16/32/64n: srun --mpi=pmix + hcoll/ucc disabled. busbw 390/389/388/388.5 (64n, 2026-05-11). SHARP needs 8-HCA list (drop mlx5_0-3, 2026-05-13)
- [SHARP active on slinky](project_sharp_active_slinky.md) — 2026-05-12 job 2644: 64n CollnetChain SHARP 534 GB/s @ 32 GiB allreduce, +37.6% vs ring. NEW: SHARP env hurts FSDP-heavy workloads — apply per-workload only.
- [slinky LLMB-run / pyxis container workarounds](project_slinky_llmb_workarounds.md) — Patch stack for llmb-run on slinky (memlock, pyxis perms, sharing strategy). Verified Llama 70B FP8 1582 TFLOPS/GPU @ 128 GPU 2026-05-10
- [NCCL benchmark scripts and workflow](project_nccl_benchmark_scripts.md) — run_slurm.sh usage, test matrix, straggler detection, B200 baselines
- [find_stragglers.py — straggler detection tool](project_straggler_finder.md) — 2-round disjoint 2-node ring sweep on slinky (rewrite 2026-05-13). Round 1 watchdog flags bad groups, Round 2 pairs each candidate with healthy H-node
- [dgxc-benchmarking installation](project_dgxc_benchmarking.md) — Install paths, enroot config, installer patches, llmb-run commands for Llama3.1 on B200 slinky cluster
- [worklog routing convention](feedback_worklog_routing.md) — Where to file new reports: LLM training → dgxc repo's worklog/, NCCL → nccl-tests baselines/<cluster>/, YYYY-MM-DD_ prefix
- [new-cluster adaptation checklist](feedback_new_cluster_adaptation.md) — Constants to change in find_stragglers.py / run_slurm.sh for a new cluster + HCA identification recipe + install hazards
- [NCCL multi-node hang debug ladder](feedback_nccl_multinode_hang_debug.md) — Ordered diagnostic ladder for hangs at scale: NCCL_DEBUG=INFO → hcoll/ucc disable → bootstrap NIC → HCA list → straggler → SHARP
