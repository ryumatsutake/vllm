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
import csv
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


from vllm.patches.batch_decode_scheduler.perf_test_harness import BenchHarness


@dataclass
class BenchResult:
    """One grid cell, aggregated RTP-LLM style.

    Every ``*_ms`` is a trimmed mean over rounds (sort the per-round values,
    drop min and max, average the rest — see ``_trimmed_mean``), matching
    RTP-LLM ``batch_perf_impl.run``'s ``measurements[1:-1]``. Every
    ``*_mean_ms`` is the raw mean over the same rounds, for reference.
    The sample unit is the round (one per ``num_iters``), not the step.
    """

    mode: str
    batch_size: int
    seq_len: int
    num_rounds: int = 0
    # per-round total wall time, begin -> last token (≈ RTP cost_time)
    cost_ms: float = 0.0
    cost_mean_ms: float = 0.0
    # per-round prefill / first-token wall time (≈ RTP first_token_cost_time)
    prefill_ms: float = 0.0
    prefill_mean_ms: float = 0.0
    # per-round (cost - prefill) / decode_steps (≈ RTP decode_time_per_token)
    per_token_ms: float = 0.0
    per_token_mean_ms: float = 0.0

    @property
    def primary_ms(self) -> float:
        """Headline metric: decode_time_per_token for decode, else prefill."""
        return self.per_token_ms if self.mode == "decode" else self.prefill_ms


def _trimmed_mean(samples: list[float]) -> float:
    """RTP-LLM aggregation: sort, drop min and max, average the rest.

    Falls back to plain mean when fewer than 3 samples (nothing to trim).
    Mirrors batch_perf_impl.run: measurements.sort(); measurements[1:-1].
    """
    if not samples:
        return 0.0
    if len(samples) < 3:
        return float(np.mean(samples))
    trimmed = sorted(samples)[1:-1]
    return float(np.mean(trimmed))


def _agg(samples: list[float]) -> tuple[float, float]:
    """Return (trimmed_mean, raw_mean) over rounds."""
    if not samples:
        return 0.0, 0.0
    return _trimmed_mean(samples), float(np.mean(samples))


def run_prefill_bench(
    harness: BenchHarness,
    batch_size: int,
    seq_len: int,
    num_iters: int,
    num_warmup_iters: int = 1,
    profile: bool = False,
    profile_steps: int = 3,
) -> BenchResult:
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
        # Wall time of the single prefill step (≈ RTP first_token_cost_time).
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            cost_times.append(cost_ms)
        harness.drain()
    cost_trim, cost_mean = _agg(cost_times)
    return BenchResult(
        mode="prefill",
        batch_size=batch_size,
        seq_len=seq_len,
        num_rounds=len(cost_times),
        cost_ms=cost_trim,
        cost_mean_ms=cost_mean,
        prefill_ms=cost_trim,        # prefill == total for a single step
        prefill_mean_ms=cost_mean,
    )


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
    cost_times: list[float] = []
    prefill_times: list[float] = []
    per_token_times: list[float] = []
    total = num_warmup_iters + num_iters
    profiled = 0
    for i in range(total):
        prefill_ms = 0.0
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
            # Wall time to first token (≈ RTP first_token_cost_time), on the
            # same batch-start clock as cost below.
            prefill_ms = harness.mark_lap()
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
        # Whole round, begin -> last token (≈ RTP cost_time).
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            cost_times.append(cost_ms)
            prefill_times.append(prefill_ms)
            # Per-round decode_time_per_token = (cost - prefill) / steps,
            # matching RTP dataclass.ResponseInfo.decode_time_per_token.
            per_token_times.append((cost_ms - prefill_ms) / num_decode_steps)
        harness.drain()

    cost_trim, cost_mean = _agg(cost_times)
    prefill_trim, prefill_mean = _agg(prefill_times)
    pt_trim, pt_mean = _agg(per_token_times)
    return BenchResult(
        mode="decode",
        batch_size=batch_size,
        seq_len=kv_len,
        num_rounds=len(cost_times),
        cost_ms=cost_trim,
        cost_mean_ms=cost_mean,
        prefill_ms=prefill_trim,
        prefill_mean_ms=prefill_mean,
        per_token_ms=pt_trim,
        per_token_mean_ms=pt_mean,
    )


def print_table(results: list[BenchResult]) -> None:
    prefill = [r for r in results if r.mode == "prefill"]
    decode = [r for r in results if r.mode == "decode"]

    if prefill:
        print("=== Prefill (trimmed mean over rounds, "
              "≈ RTP-LLM first_token_cost_time) ===")
        header = (
            f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
            f"{'prefill(ms)':>12} {'mean(ms)':>9} {'rounds':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in prefill:
            print(
                f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
                f"{r.prefill_ms:>12.2f} {r.prefill_mean_ms:>9.2f} "
                f"{r.num_rounds:>7}"
            )

    if decode:
        if prefill:
            print()
        print("=== Decode (trimmed mean over rounds, "
              "≈ RTP-LLM grid_perf_test) ===")
        header = (
            f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
            f"{'cost(ms)':>9} {'prefill(ms)':>12} "
            f"{'per_token(ms)':>14} {'mean(ms)':>9} {'rounds':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in decode:
            print(
                f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
                f"{r.cost_ms:>9.2f} {r.prefill_ms:>12.2f} "
                f"{r.per_token_ms:>14.2f} {r.per_token_mean_ms:>9.2f} "
                f"{r.num_rounds:>7}"
            )


def write_csv(results: list[BenchResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "batch_size", "seq_len", "num_rounds",
            "cost_ms", "cost_mean_ms",
            "prefill_ms", "prefill_mean_ms",
            "per_token_ms", "per_token_mean_ms",
        ])
        for r in results:
            writer.writerow([
                r.mode, r.batch_size, r.seq_len, r.num_rounds,
                f"{r.cost_ms:.2f}", f"{r.cost_mean_ms:.2f}",
                f"{r.prefill_ms:.2f}", f"{r.prefill_mean_ms:.2f}",
                f"{r.per_token_ms:.2f}", f"{r.per_token_mean_ms:.2f}",
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
        "--disable-mm", action="store_true",
        help="Text-only decode of a VL model: zero multimodal slots so the "
        "engine skips vision profiling (aligns with RTP-LLM's text benchmark).",
    )
    parser.add_argument(
        "--vllm-scopes", action="store_true",
        help="Enable vLLM's built-in engine-phase record_function scopes "
        "(gpu_model_runner: forward/sample/..., schedule: ...) as user_annotation "
        "in the torch trace. Sets VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 AND "
        "VLLM_USE_V2_MODEL_RUNNER=0 — the gpu_model_runner: scopes exist ONLY in "
        "the legacy V1 runner; the V2 runner (default for Qwen3/Llama/Mistral/... "
        "per DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES) has none. Eager only; "
        "negligible CPU overhead so use with --profile.",
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
    parser.add_argument(
        "--analyze", action="store_true",
        help="After a torch-profile pass, print the per-category kernel "
        "breakdown (perf_test_timeline).",
    )
    parser.add_argument(
        "--rtp-trace", default=None,
        help="Path to an RTP-LLM chrome trace; prints a per-category "
        "per-step vLLM-vs-RTP diff after profiling.",
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
        disable_mm=args.disable_mm,
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

            # Timing pass (nsys profiling, if any, piggybacks here).
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
            metric = "per_token" if args.mode == "decode" else "prefill"
            print(f"  {label}: {metric}={r.primary_ms:.2f}ms "
                  f"cost={r.cost_ms:.2f}ms")

            # Dedicated torch-profile pass for per-component breakdown.
            if use_torch_profile:
                steps = args.num_decode_steps if args.mode == "decode" else 1
                trace = harness.profile_run(
                    bs, seq_len, args.mode, steps,
                    profile_output,
                    skip_prefill_forward=args.skip_prefill_forward,
                )
                print(f"  Trace: {trace}")
                if args.analyze or args.rtp_trace:
                    from vllm.patches.batch_decode_scheduler.perf_test_timeline import (  # noqa: E501
                        analyze_file, compare,
                    )
                    if args.analyze:
                        analyze_file(trace, steps)
                    if args.rtp_trace:
                        compare(trace, args.rtp_trace, "vLLM", "RTP-LLM",
                                steps_a=steps)
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
    if args.vllm_scopes:
        os.environ.setdefault("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

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
              f"primary={[f'{r.primary_ms:.2f}' for r in results]} "
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
    if args.vllm_scopes:
        os.environ.setdefault("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "1")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
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
            base = r0.primary_ms
            for r in (r for r in rank_results if r != 0):
                peer = rank_results[r][i].primary_ms
                diff_pct = abs(peer - base) / max(base, 1e-6) * 100
                if diff_pct > 10:
                    print(f"WARNING: rank {r} primary={peer:.2f}ms vs "
                          f"rank 0 primary={base:.2f}ms "
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
