"""Chrome-trace timeline analyzer for per-component GPU time.

Parses a Kineto / torch.profiler chrome-trace JSON and attributes GPU kernel
time to high-level categories (Attention, MoE, Dense GEMM, Norm, RoPE,
Sampling, Communication, ...). The category taxonomy is kept identical to
RTP-LLM's ``analyze_timeline.py`` so a vLLM trace and an RTP-LLM trace can be
compared category-by-category.

Two views:

  * ``category_breakdown`` — group GPU kernels by name-pattern category. Works
    in both eager and CUDA-graph modes (kernels are always visible in the
    trace), so this is the primary, always-on path.
  * ``scope_breakdown`` — group ``user_annotation`` ranges (record_function /
    at::RecordFunction scopes) by name. Only meaningful for eager runs, where
    the vLLM harness injects ``attn`` / ``moe`` / ``norm`` / ``rope`` scopes
    that line up with RTP-LLM's RecordFunction scopes.

Usage::

    python -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
        traces/vllm_decode_bs4_seq128_steps10.json

    # side-by-side vLLM vs RTP-LLM
    python -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
        --compare vllm.json rtp.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Ordered (pattern, category). First match wins, so put specific before generic.
# RTP-LLM patterns (from chrome-timeline-analysis/analyze_timeline.py) are kept
# so the categories line up; vLLM-specific kernel families are added alongside.
_KERNEL_PATTERNS: list[tuple[str, str]] = [
    # --- Attention (RTP: flashinfer/flash; vLLM: flash_attn, triton, flashinfer) ---
    (r"BatchDecodeWithPagedKVCacheKernel|BatchPrefill|PrefillWithKVCacheKernel"
     r"|flash_fwd|MergeStates|sm90_fp8_mqa_logits|mla_combine"
     r"|AppendPagedKVCacheKernel"
     r"|unified_attention|attn_fwd|_fwd_kernel|flashinfer|paged_attention"
     r"|reshape_and_cache", "Attention"),
    # --- MLA (deepseek) ---
    (r"fast_hadamard_transform|concat_mla_k|concat_and_cache_ds_mla"
     r"|indexer_k_quant_and_cache", "MLA"),
    # --- MoE GEMM / experts ---
    (r"deep_gemm::sm90_fp8_gemm.*GemmType\)2"
     r"|fused_moe_kernel|fused_moe|grouped_gemm|group_gemm|moe_mm"
     r"|cutlass.*moe|marlin_moe|batched_gemm", "MoE GEMM"),
    # --- MoE routing / align ---
    (r"moeSoftmax|moeTopK|fakeBalanceExpert"
     r"|topk_softmax|moe_align|moe_sum|sgl_moe", "MoE Routing"),
    # --- MoE communication (EP) ---
    (r"deep_ep::|_fwd_kernel_ep_scatter|_fwd_kernel_ep_gather", "MoE Communication"),
    # --- Dense GEMM (FP8 / low-precision / cutlass / cublas) ---
    (r"deep_gemm::sm90_fp8_gemm|deep_gemm::transpose", "Dense GEMM (FP8)"),
    (r"nvjet_|cublas|splitKreduce|dot_kernel|reduce_1Block|sm80_xmma_gemm"
     r"|sm90.*gemm|sm80.*gemm|cutlass|scaled_mm|machete|marlin"
     r"|gptq_gemm|awq_gemm|gemm_kernel|GemmUniversal|s16816gemm|ampere.*gemm"
     r"|wgmma", "Dense GEMM"),
    # --- Activation (incl. torch.compile fused silu/gelu) ---
    (r"_silu_mul_masked|act_and_mul|SigmoidGateScaleAdd|_silu_and_mul"
     r"|silu_and_mul|gelu_and_mul|gelu_kernel|act_kernel"
     r"|silu|fused_mul_silu|fused.*gelu", "Activation"),
    # --- Quantization ---
    (r"per_token_group_quant|computeFP8Quantize|_tma_align_input_scale"
     r"|scaled_fp8_quant|dynamic_scaled_int8_quant|quant_kernel"
     r"|awq_dequantize|gptq_marlin_repack|awq_marlin_repack", "Quantization"),
    # --- Normalization ---
    (r"RMSNorm|FusedAddRMSNorm|fusedQkRmsNorm|generalLayerNorm|layer_norm_fwd"
     r"|rms_norm|fused_add_rms_norm|layernorm|LayerNorm", "Normalization"),
    # --- RoPE ---
    (r"decode_add_fusedQKV.*rope|rope_cache|ApplyRotaryPosIds"
     r"|rotary_embedding|rotary_kernel|apply_rope", "RoPE"),
    # --- Embedding ---
    (r"embedding_lookup|embedding_kernel|EmbeddingBag", "Embedding"),
    # --- Sampling ---
    (r"TopPRenormProb|TopKRenormProb|TopKTopPSampling|topk_transform_prefill"
     r"|addBiasSoftMax|batchApplyPenalty|batchApplyTemperaturePenalty"
     r"|ChainSpeculativeSamp"
     r"|top_k|top_p|argmax|random_sample|_sample|penalt|softmax_kernel", "Sampling"),
    # --- Linear / recurrent attention ---
    (r"fused_recurrent_gated_delta_rule|causal_conv1d_update"
     r"|gated_delta|chunk_gated|recurrent", "Linear Attention"),
    # --- Collective communication ---
    (r"all_reduce|all_gather|ncclDev|multimem_all_gather|memcpy32_post|Broadcast"
     r"|nccl|reduce_scatter|custom_all_reduce|one_shot_all_reduce", "Communication"),
    # --- Memory ---
    (r"Memcpy|memcpy", "Memory Copy"),
    (r"Memset|memset", "Memory Set"),
    # --- Index / offset / KV-metadata prep (decode bookkeeping kernels) ---
    (r"ConvertOffset|BlockArray|convert_req_index"
     r"|block_tables|slot_mapping|compute_slot|gather_block|prepare_pos"
     r"|seq_lens|post_update_kernel|apply_write_kernel", "Index/Offset"),
    (r"transposeAxis01|lookupHiddenStateOfLastToken", "Elementwise/Copy"),
    (r"vectorized_elementwise|elementwise_kernel|unrolled_elementwise|CatArray"
     r"|FillFunctor|CUDAFunctor_add|direct_copy|sigmoid_kernel|BinaryFunctor"
     r"|SoftMaxForward|reduce_kernel|distribution_elementwise"
     r"|copy_kernel|fill_kernel|cat_kernel|index_kernel|gather_kernel"
     r"|scatter_kernel", "Elementwise/Copy"),
    # --- torch.compile-fused kernels (generic inductor names, ambiguous).
    # Kept as a distinct bucket rather than "Other" so it's clear these are
    # fused ops (RoPE/norm/residual) that lost their semantic name under
    # torch.compile. Run --enforce-eager --scopes for a semantic breakdown. ---
    (r"triton_poi_fused|triton_red_fused|triton_per_fused|triton_.*_fused",
     "Fused (compile)"),
]

_COMPILED = [(re.compile(p), c) for p, c in _KERNEL_PATTERNS]


def classify_kernel(name: str) -> str:
    """Map a GPU kernel name to a high-level category (first match wins)."""
    for pattern, category in _COMPILED:
        if pattern.search(name):
            return category
    return "Other"


def _parse_context(path: str) -> dict:
    """Recover mode/bs/seq/steps from the harness trace-name convention.

    Harness names torch traces ``vllm_<mode>_bs<BS>_seq<SEQ>_steps<N>``.
    """
    stem = Path(path).stem
    ctx: dict = {}
    for key, pat in (
        ("mode", r"vllm_(prefill|decode)"),
        ("batch_size", r"bs(\d+)"),
        ("seq_len", r"seq(\d+)"),
        ("num_steps", r"steps(\d+)"),
    ):
        m = re.search(pat, stem)
        if m:
            val = m.group(1)
            ctx[key] = val if key == "mode" else int(val)
    return ctx


def parse_trace(path: str) -> dict:
    """Load a chrome-trace JSON and bucket complete (``ph=='X'``) events.

    Returns a dict with lists ``kernel`` / ``gpu_memcpy`` / ``gpu_memset`` /
    ``cuda_runtime`` / ``user_annotation`` (each event is the raw dict) plus the
    filename-derived ``context``.
    """
    with open(path) as f:
        data = json.load(f)
    buckets: dict[str, list] = defaultdict(list)
    for ev in data.get("traceEvents", []):
        if ev.get("ph") != "X":
            continue
        buckets[ev.get("cat", "")].append(ev)
    return {
        "kernel": buckets.get("kernel", []),
        "gpu_memcpy": buckets.get("gpu_memcpy", []),
        "gpu_memset": buckets.get("gpu_memset", []),
        "cuda_runtime": buckets.get("cuda_runtime", []),
        "user_annotation": buckets.get("user_annotation", []),
        "context": _parse_context(path),
    }


def category_breakdown(trace: dict, num_steps: int | None = None) -> dict:
    """Aggregate GPU kernel + memcpy/memset time per category.

    Returns ``{"categories": {cat: {total_us, count, per_step_us, pct}},
    "total_gpu_us", "num_steps"}``. ``per_step_us`` divides by ``num_steps``
    (from arg or trace context; falls back to 1).
    """
    if num_steps is None:
        num_steps = trace.get("context", {}).get("num_steps") or 1
    gpu_events = trace["kernel"] + trace["gpu_memcpy"] + trace["gpu_memset"]
    total_gpu = sum(e["dur"] for e in gpu_events) or 1.0

    cats: dict[str, dict] = defaultdict(
        lambda: {"total_us": 0.0, "count": 0}
    )
    for e in trace["kernel"]:
        c = classify_kernel(e["name"])
        cats[c]["total_us"] += e["dur"]
        cats[c]["count"] += 1
    # memcpy / memset already carry their own category name
    for e in trace["gpu_memcpy"]:
        cats["Memory Copy"]["total_us"] += e["dur"]
        cats["Memory Copy"]["count"] += 1
    for e in trace["gpu_memset"]:
        cats["Memory Set"]["total_us"] += e["dur"]
        cats["Memory Set"]["count"] += 1

    for c in cats.values():
        c["per_step_us"] = c["total_us"] / num_steps
        c["pct"] = c["total_us"] / total_gpu * 100
    return {
        "categories": dict(cats),
        "total_gpu_us": total_gpu,
        "num_steps": num_steps,
    }


def scope_breakdown(trace: dict, num_steps: int | None = None) -> dict:
    """Aggregate ``user_annotation`` (record_function) ranges by name.

    Only leaf/short scope names are kept meaningful; the harness injects
    ``attn`` / ``moe`` / ``norm`` / ``rope``. Note: user_annotation durations
    are CPU-range wall times, useful as a semantic cross-check, not exact GPU
    time.
    """
    if num_steps is None:
        num_steps = trace.get("context", {}).get("num_steps") or 1
    scopes: dict[str, dict] = defaultdict(lambda: {"total_us": 0.0, "count": 0})
    for e in trace["user_annotation"]:
        name = e["name"]
        scopes[name]["total_us"] += e["dur"]
        scopes[name]["count"] += 1
    for s in scopes.values():
        s["per_step_us"] = s["total_us"] / num_steps
    return {"scopes": dict(scopes), "num_steps": num_steps}


def _print_category_table(bd: dict, title: str) -> None:
    print(f"\n=== {title} (num_steps={bd['num_steps']}) ===")
    header = (
        f"{'Category':<22} {'total(us)':>11} {'per_step(us)':>13} "
        f"{'%kernel':>8} {'count':>7}"
    )
    print(header)
    print("-" * len(header))
    rows = sorted(
        bd["categories"].items(), key=lambda kv: kv[1]["total_us"], reverse=True
    )
    for cat, s in rows:
        print(
            f"{cat:<22} {s['total_us']:>11.1f} {s['per_step_us']:>13.2f} "
            f"{s['pct']:>7.1f}% {s['count']:>7}"
        )
    print("-" * len(header))
    print(f"{'TOTAL GPU':<22} {bd['total_gpu_us']:>11.1f} "
          f"{bd['total_gpu_us'] / bd['num_steps']:>13.2f}")


def _print_scope_table(sb: dict) -> None:
    if not sb["scopes"]:
        return
    print(f"\n=== Semantic scopes (record_function CPU wall time, not GPU; "
          f"num_steps={sb['num_steps']}) ===")
    header = f"{'Scope':<32} {'total(us)':>11} {'per_step(us)':>13} {'count':>7}"
    print(header)
    print("-" * len(header))
    rows = sorted(
        sb["scopes"].items(), key=lambda kv: kv[1]["total_us"], reverse=True
    )
    for name, s in rows:
        print(
            f"{name:<16} {s['total_us']:>11.1f} {s['per_step_us']:>13.2f} "
            f"{s['count']:>7}"
        )


def compare(
    path_a: str,
    path_b: str,
    label_a: str = "vLLM",
    label_b: str = "RTP-LLM",
    steps_a: int | None = None,
    steps_b: int | None = None,
) -> None:
    """Print a side-by-side per-category per-step comparison of two traces."""
    ba = category_breakdown(parse_trace(path_a), steps_a)
    bb = category_breakdown(parse_trace(path_b), steps_b)
    cats = sorted(
        set(ba["categories"]) | set(bb["categories"]),
        key=lambda c: -(ba["categories"].get(c, {}).get("total_us", 0)
                        + bb["categories"].get(c, {}).get("total_us", 0)),
    )
    print(f"\n=== Per-step per-category: {label_a} vs {label_b} "
          f"(us/step; steps {ba['num_steps']} vs {bb['num_steps']}) ===")
    header = (
        f"{'Category':<22} {label_a:>12} {label_b:>12} "
        f"{'diff':>10} {'diff%':>8}"
    )
    print(header)
    print("-" * len(header))
    for c in cats:
        a = ba["categories"].get(c, {}).get("per_step_us", 0.0)
        b = bb["categories"].get(c, {}).get("per_step_us", 0.0)
        diff = a - b
        diffpct = (diff / b * 100) if b else float("inf")
        pct_str = f"{diffpct:>7.0f}%" if b else "     n/a"
        print(f"{c:<22} {a:>12.2f} {b:>12.2f} {diff:>10.2f} {pct_str}")
    print("-" * len(header))
    ta = ba["total_gpu_us"] / ba["num_steps"]
    tb = bb["total_gpu_us"] / bb["num_steps"]
    print(f"{'TOTAL GPU/step':<22} {ta:>12.2f} {tb:>12.2f} {ta - tb:>10.2f}")


def analyze_file(path: str, num_steps: int | None = None) -> None:
    trace = parse_trace(path)
    ctx = trace["context"]
    if ctx:
        print(f"Trace: {Path(path).name}  context={ctx}")
    bd = category_breakdown(trace, num_steps)
    _print_category_table(bd, "Kernel category breakdown")
    _print_scope_table(scope_breakdown(trace, num_steps))
    other = bd["categories"].get("Other")
    if other and other["pct"] > 5:
        print(f"\nWARNING: 'Other' = {other['pct']:.1f}% of kernel time — "
              f"consider adding patterns to classify_kernel().")


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM/RTP timeline analyzer")
    parser.add_argument("trace", nargs="?", help="chrome-trace JSON to analyze")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override num decode steps (else parsed from name)")
    parser.add_argument("--compare", nargs=2, metavar=("VLLM", "RTP"),
                        help="Compare two traces per-category per-step")
    parser.add_argument("--labels", nargs=2, default=["vLLM", "RTP-LLM"],
                        help="Labels for --compare")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1],
                args.labels[0], args.labels[1])
    elif args.trace:
        analyze_file(args.trace, args.steps)
    else:
        parser.error("provide a trace path or --compare A B")


if __name__ == "__main__":
    main()
