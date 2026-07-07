"""CLI runner for GPU end-to-end prefill/decode benchmarking.

Usage::

    python -m vllm.patches.batch_decode_scheduler.perf_test_runner \
        --model facebook/opt-125m \
        --mode prefill \
        --batch-sizes 1,4 \
        --seq-lens 128,256

    # DP=2 decode benchmark (each rank gets batch_size // dp_size requests)
    python -m vllm.patches.batch_decode_scheduler.perf_test_runner \
        --model facebook/opt-125m \
        --mode decode \
        --batch-sizes 4,8 \
        --seq-lens 128 \
        --dp-size 2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


def _nullcontext():
    return contextlib.nullcontext()

from vllm.patches.batch_decode_scheduler.perf_test_harness import BenchHarness


@dataclass
class BenchResult:
    mode: str
    batch_size: int
    seq_len: int
    # per-step forward_ms stats (≈ RTP-LLM decode_time_per_token)
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    num_samples: int
    # per-request cost_time stats (≈ RTP-LLM cost_time_us)
    cost_p50_ms: float = 0.0
    cost_p90_ms: float = 0.0
    cost_mean_ms: float = 0.0
    num_cost_samples: int = 0
    # decode-mode breakdown (≈ RTP-LLM first_token_cost_time / decode_time_per_token)
    prefill_mean_ms: float = 0.0
    decode_per_token_mean_ms: float = 0.0


def run_prefill_bench(
    harness: BenchHarness,
    batch_size: int,
    seq_len: int,
    num_iters: int,
    num_warmup_iters: int = 1,
    profile: bool = False,
    profile_steps: int = 3,
) -> BenchResult:
    latencies: list[float] = []
    cost_times: list[float] = []
    total = num_warmup_iters + num_iters
    profiled = 0
    for i in range(total):
        harness.submit(batch_size, seq_len, max_tokens=1)
        harness.mark_batch_start()
        if profile and i >= num_warmup_iters and profiled < profile_steps:
            if profiled == 0:
                harness.start_profiling()
            profiled += 1
        stat = harness.run_step()
        harness.assert_phase(stat, "prefill")
        if profile and profiled == profile_steps:
            harness.stop_profiling()
            profiled += 1  # prevent re-stop
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            latencies.append(stat.forward_ms)
            cost_times.append(cost_ms)
        harness.drain()
    return _summarize("prefill", batch_size, seq_len, latencies, cost_times)


def run_decode_bench(
    harness: BenchHarness,
    batch_size: int,
    kv_len: int,
    num_decode_steps: int,
    num_iters: int,
    num_warmup_iters: int = 1,
    profile: bool = False,
    profile_steps: int = 3,
    skip_prefill_forward: bool = False,
) -> BenchResult:
    latencies: list[float] = []
    cost_times: list[float] = []
    prefill_times: list[float] = []
    total = num_warmup_iters + num_iters
    profiled = 0
    for i in range(total):
        if skip_prefill_forward:
            harness.submit_decode_only(
                batch_size, kv_len, num_decode_steps,
            )
            harness.mark_batch_start()
        else:
            harness.submit(
                batch_size, kv_len,
                max_tokens=num_decode_steps + 1,
                ignore_eos=True,
            )
            harness.mark_batch_start()
            setup_stat = harness.run_step()
            harness.assert_phase(setup_stat, "prefill")
            if i >= num_warmup_iters:
                prefill_times.append(setup_stat.forward_ms)
        for step in range(num_decode_steps):
            if (profile and i >= num_warmup_iters
                    and profiled < profile_steps):
                if profiled == 0:
                    harness.start_profiling()
                profiled += 1
            stat = harness.run_step()
            harness.assert_phase(stat, "decode")
            if profile and profiled == profile_steps:
                harness.stop_profiling()
                profiled += 1
            if i >= num_warmup_iters:
                latencies.append(stat.forward_ms)
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            cost_times.append(cost_ms)
        harness.drain()
    result = _summarize("decode", batch_size, kv_len, latencies, cost_times)
    if prefill_times:
        result.prefill_mean_ms = float(np.mean(prefill_times))
        result.decode_per_token_mean_ms = result.mean_ms
    return result


def _summarize(
    mode: str,
    batch_size: int,
    seq_len: int,
    latencies: list[float],
    cost_times: list[float] | None = None,
) -> BenchResult:
    arr = np.array(latencies)
    result = BenchResult(
        mode=mode,
        batch_size=batch_size,
        seq_len=seq_len,
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        num_samples=len(latencies),
    )
    if cost_times:
        ct = np.array(cost_times)
        result.cost_p50_ms = float(np.percentile(ct, 50))
        result.cost_p90_ms = float(np.percentile(ct, 90))
        result.cost_mean_ms = float(np.mean(ct))
        result.num_cost_samples = len(cost_times)
    return result


def print_table(results: list[BenchResult]) -> None:
    print("=== Per-Step Forward (≈ RTP-LLM decode_time_per_token) ===")
    header = (
        f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
        f"{'p50(ms)':>9} {'p90(ms)':>9} {'p99(ms)':>9} {'mean(ms)':>9} {'N':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
            f"{r.p50_ms:>9.2f} {r.p90_ms:>9.2f} {r.p99_ms:>9.2f} "
            f"{r.mean_ms:>9.2f} {r.num_samples:>5}"
        )
    if any(r.num_cost_samples > 0 for r in results):
        print()
        print("=== Per-Request Cost Time (≈ RTP-LLM cost_time_us) ===")
        cost_header = (
            f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
            f"{'p50(ms)':>9} {'p90(ms)':>9} {'mean(ms)':>9} {'N':>5}"
        )
        print(cost_header)
        print("-" * len(cost_header))
        for r in results:
            print(
                f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
                f"{r.cost_p50_ms:>9.2f} {r.cost_p90_ms:>9.2f} "
                f"{r.cost_mean_ms:>9.2f} {r.num_cost_samples:>5}"
            )
    decode_results = [r for r in results
                      if r.mode == "decode" and r.prefill_mean_ms > 0]
    if decode_results:
        print()
        print("=== Decode Breakdown "
              "(≈ RTP-LLM first_token_cost_time / decode_time_per_token) ===")
        bd_header = (
            f"{'BS':>4} {'SeqLen':>7} "
            f"{'prefill(ms)':>12} {'decode/tok(ms)':>15}"
        )
        print(bd_header)
        print("-" * len(bd_header))
        for r in decode_results:
            print(
                f"{r.batch_size:>4} {r.seq_len:>7} "
                f"{r.prefill_mean_ms:>12.2f} {r.decode_per_token_mean_ms:>15.2f}"
            )


def write_csv(results: list[BenchResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "batch_size", "seq_len",
            "step_p50_ms", "step_p90_ms", "step_p99_ms", "step_mean_ms",
            "num_step_samples",
            "cost_p50_ms", "cost_p90_ms", "cost_mean_ms",
            "num_cost_samples",
            "prefill_mean_ms", "decode_per_token_mean_ms",
        ])
        for r in results:
            writer.writerow([
                r.mode, r.batch_size, r.seq_len,
                f"{r.p50_ms:.2f}", f"{r.p90_ms:.2f}", f"{r.p99_ms:.2f}",
                f"{r.mean_ms:.2f}", r.num_samples,
                f"{r.cost_p50_ms:.2f}", f"{r.cost_p90_ms:.2f}",
                f"{r.cost_mean_ms:.2f}", r.num_cost_samples,
                f"{r.prefill_mean_ms:.2f}", f"{r.decode_per_token_mean_ms:.2f}",
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM GPU E2E bench (prefill-only / decode-only)"
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--mode", required=True, choices=["prefill", "decode"],
        help="Benchmark mode",
    )
    parser.add_argument(
        "--batch-sizes", default="1,4,16",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--seq-lens", default="128,512,1024",
        help="Comma-separated sequence lengths (prefill=seq_len, decode=kv_len)",
    )
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-decode-steps", type=int, default=20)
    parser.add_argument(
        "--num-warmup-iters", type=int, default=1,
        help="Warmup iterations per grid point (results discarded)",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Pipeline parallel size")
    parser.add_argument("--dp-size", type=int, default=1,
                        help="Data parallel size (spawns N processes, "
                        "each rank gets batch_size // dp_size requests)")
    parser.add_argument(
        "--enable-expert-parallel", action="store_true",
        help="Enable expert parallelism for MoE models (split experts "
        "across DP ranks with all-to-all communication)",
    )
    parser.add_argument(
        "--skip-prefill-forward", action="store_true",
        help="Skip prefill forward in decode mode (hack KV blocks, align with RTP-LLM)",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable profiling (nsys cudaProfilerApi or torch.profiler)",
    )
    parser.add_argument(
        "--profile-mode", default="nsys", choices=["nsys", "torch"],
        help="Profile backend: nsys (requires nsys wrapper) or torch (Kineto trace)",
    )
    parser.add_argument(
        "--profile-steps", type=int, default=3,
        help="Number of steps to capture in profile window",
    )
    parser.add_argument(
        "--profile-output", default=None,
        help="Output dir for torch profiler traces (default: ./profile_output)",
    )
    return parser.parse_args()


def _run_bench_grid(
    args: argparse.Namespace,
    batch_sizes: list[int],
    seq_lens: list[int],
    dp_barrier: multiprocessing.Barrier | None = None,
) -> list[BenchResult]:
    """Run the full benchmark grid on the current process. Returns results."""
    max_batch_size = max(batch_sizes)
    print(f"Initializing harness (model={args.model}, "
          f"max_batch_size={max_batch_size}) ...")
    t0 = time.time()
    harness = BenchHarness(
        model=args.model,
        max_batch_size=max_batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        enable_expert_parallel=args.enable_expert_parallel,
        dp_barrier=dp_barrier,
    )
    print(f"Harness ready in {time.time() - t0:.1f}s")

    min_seq = min(seq_lens)
    print(f"Global warmup (bs=1, seq_len={min_seq}) ...")
    harness.submit(1, min_seq, max_tokens=2, ignore_eos=True)
    harness.run_step_no_timing()
    harness.run_step_no_timing()
    harness.drain()
    print("Global warmup done")

    use_nsys_profile = args.profile and args.profile_mode == "nsys"
    use_torch_profile = args.profile and args.profile_mode == "torch"
    profile_output = args.profile_output or "./profile_output"

    results: list[BenchResult] = []
    for bs in batch_sizes:
        for seq_len in seq_lens:
            label = f"{args.mode} bs={bs} seq_len={seq_len}"
            print(f"Running {label} ...")

            trace_name = f"vllm_{args.mode}_bs{bs}_seq{seq_len}"
            ctx = (harness.torch_profile(profile_output, trace_name)
                   if use_torch_profile
                   else _nullcontext())

            with ctx:
                if args.mode == "prefill":
                    r = run_prefill_bench(
                        harness, bs, seq_len,
                        args.num_iters, args.num_warmup_iters,
                        profile=use_nsys_profile,
                        profile_steps=args.profile_steps,
                    )
                else:
                    r = run_decode_bench(
                        harness, bs, seq_len,
                        args.num_decode_steps,
                        args.num_iters, args.num_warmup_iters,
                        profile=use_nsys_profile,
                        profile_steps=args.profile_steps,
                        skip_prefill_forward=args.skip_prefill_forward,
                    )
            results.append(r)
            print(f"  {label}: step_p50={r.p50_ms:.2f}ms "
                  f"cost_p50={r.cost_p50_ms:.2f}ms")
            if use_torch_profile:
                print(f"  Trace: {profile_output}/{trace_name}.json")
    return results


def _detect_moe(model: str) -> bool:
    """Check if the model is a MoE model by reading its config."""
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        for cfg in (config, getattr(config, "text_config", None)):
            if cfg is None:
                continue
            num_experts = getattr(cfg, "num_local_experts", 0) or \
                          getattr(cfg, "num_experts", 0)
            if num_experts > 0:
                return True
        return False
    except Exception:
        return False


def _dp_worker(
    rank: int,
    dp_size: int,
    tp_size: int,
    master_port: int,
    use_dp_env: bool,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue,
    args: argparse.Namespace,
    batch_sizes: list[int],
    seq_lens: list[int],
) -> None:
    """Worker process for one DP rank."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    if use_dp_env:
        # Cross-DP EP mode: let vLLM handle GPU assignment via
        # init_device()'s dp_local_rank * tp_pp_world_size + local_rank.
        # Setting CUDA_VISIBLE_DEVICES here would conflict with that.
        os.environ["VLLM_DP_RANK"] = str(rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    else:
        # Independent DP mode (no EP): isolate each rank's GPUs.
        gpu_start = rank * tp_size
        gpus = ",".join(str(gpu_start + i) for i in range(tp_size))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    local_batch_sizes = [bs // dp_size for bs in batch_sizes]
    try:
        t_start = time.time()
        results = _run_bench_grid(
            args, local_batch_sizes, seq_lens,
            dp_barrier=barrier if use_dp_env else None,
        )
        elapsed = time.time() - t_start
        for r in results:
            r.batch_size *= dp_size
        print(f"[DP rank {rank}] "
              f"step_p50={[f'{r.p50_ms:.2f}' for r in results]} "
              f"({elapsed:.1f}s total incl. init)")
        barrier.wait()
        result_queue.put((rank, results))
    except Exception as e:
        import traceback
        traceback.print_exc()
        barrier.wait()
        result_queue.put((rank, e))


def main() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    dp_size = args.dp_size

    if dp_size <= 1:
        results = _run_bench_grid(args, batch_sizes, seq_lens)
        print()
        print_table(results)
        if args.output:
            write_csv(results, args.output)
            print(f"\nCSV written to {args.output}")
        return

    for bs in batch_sizes:
        if bs < dp_size or bs % dp_size != 0:
            raise ValueError(
                f"batch_size {bs} must be >= dp_size ({dp_size}) "
                f"and divisible by it"
            )

    use_dp_env = args.enable_expert_parallel and _detect_moe(args.model)
    print(f"DP mode: {'EP (VLLM_DP env vars)' if use_dp_env else 'independent (CUDA_VISIBLE_DEVICES)'}, "
          f"dp_size={dp_size}")

    from vllm.utils.network_utils import get_open_port
    master_port = get_open_port() if use_dp_env else 0
    barrier = multiprocessing.Barrier(dp_size)
    result_queue = multiprocessing.Queue()

    procs = []
    for rank in range(dp_size):
        p = multiprocessing.Process(
            target=_dp_worker,
            args=(rank, dp_size, args.tp_size, master_port, use_dp_env,
                  barrier, result_queue, args, batch_sizes, seq_lens),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)
        if p.exitcode is None:
            print(f"Killing worker pid={p.pid} (timeout)")
            p.kill()

    rank_results: dict[int, list[BenchResult]] = {}
    while not result_queue.empty():
        rank, data = result_queue.get_nowait()
        if isinstance(data, Exception):
            print(f"DP rank {rank} failed: {data}")
            continue
        rank_results[rank] = data

    if 0 not in rank_results:
        print("ERROR: rank 0 did not return results")
        sys.exit(1)

    results = rank_results[0]

    if len(rank_results) > 1:
        for i, r0 in enumerate(results):
            peer_p50s = [rank_results[r][i].p50_ms
                         for r in rank_results if r != 0]
            for r, p50 in zip(
                [r for r in rank_results if r != 0], peer_p50s
            ):
                diff_pct = abs(p50 - r0.p50_ms) / max(r0.p50_ms, 1e-6) * 100
                if diff_pct > 10:
                    print(f"WARNING: rank {r} p50={p50:.2f}ms vs "
                          f"rank 0 p50={r0.p50_ms:.2f}ms "
                          f"({diff_pct:.0f}% diff) for "
                          f"bs={r0.batch_size} seq={r0.seq_len}")

    print()
    print(f"=== DP={dp_size} (showing rank 0 results, "
          f"batch_size is global) ===")
    print_table(results)
    if args.output:
        write_csv(results, args.output)
        print(f"\nCSV written to {args.output}")


if __name__ == "__main__":
    main()
