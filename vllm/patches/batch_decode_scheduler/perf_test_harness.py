"""BenchHarness: GPU end-to-end benchmark aligned with RTP-LLM BatchDecodeScheduler.

Drives vLLM's EngineCore in schedule → execute → update steps,
timing aligned with RTP-LLM decode_time_per_token (dataclass.py).

forward_ms per step ≈ RTP-LLM's per-step execution (forward + sampler +
dispatch), excluding schedule(). BatchDecodeScheduler's inter-step
scheduling overhead is negligible (<100μs), so the two are comparable.

Known differences vs RTP-LLM perf test:
- No FAKE_BALANCE_EXPERT: RTP-LLM forces uniform expert routing in MoE
  models for stable benchmarks. vLLM has no equivalent; expert routing
  follows the model's gating network. This may cause variance in MoE
  decode latency across runs.
- DP via runner only: harness itself is single-process (InprocClient).
  DP is supported by the runner, which spawns one harness per DP rank.
  See perf_test_runner.py --dp-size for DP+EP benchmarks.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus


def _force_vllm_scopes() -> None:
    """Force vLLM's built-in record_function scopes on, robustly.

    vLLM gates its ``gpu_model_runner:`` / ``schedule:`` scopes behind
    ``record_function_or_nullcontext`` (vllm/v1/utils.py), which freezes its
    choice into a module global ``_PROFILER_FUNC`` on the FIRST call. If that
    first call happened before VLLM_CUSTOM_SCOPES_FOR_PROFILING was visible, the
    global sticks at ``nullcontext`` forever. We overwrite ``_PROFILER_FUNC`` so
    all scopes emit real record_function ranges regardless of first-call timing.

    NOTE: the ``gpu_model_runner:`` scopes only exist in vLLM's *legacy V1* model
    runner (vllm/v1/worker/gpu_model_runner.py). The V2 runner
    (vllm/v1/worker/gpu/model_runner.py, default for archs in
    DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES — Qwen3/Llama/Mistral/...) has no
    scopes. The ``--vllm-scopes`` runner flag sets VLLM_USE_V2_MODEL_RUNNER=0 so
    the scoped V1 runner is used; this function only handles the on/off gating.
    """
    import vllm.v1.utils as U
    from torch.autograd.profiler import record_function

    U._PROFILER_FUNC = record_function


@dataclass
class StepStat:
    forward_ms: float
    num_scheduled_tokens: int
    num_reqs: int
    phase: Literal["prefill", "decode", "mixed"]


class BenchHarness:
    """Wrap EngineCore for phase-separated GPU benchmarking.

    Lifecycle::

        harness = BenchHarness(model, max_batch_size=16, ...)
        harness.submit(batch_size=4, seq_len=1024, max_tokens=1)
        stat = harness.run_step()
        harness.assert_phase(stat, "prefill")
        harness.drain()
    """

    def __init__(
        self,
        model: str,
        max_batch_size: int,
        *,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        dp_barrier=None,
        disable_mm: bool = False,
    ):
        self._dp_barrier = dp_barrier

        if dp_barrier is not None:
            self._patch_executor_for_dp_sync(dp_barrier)

        extra_kwargs = {}
        if disable_mm:
            # Text-only decode of a VL model: zero the multimodal slots so the
            # engine's memory-profiling skips the (huge) vision dummy batch and
            # only the language model is exercised — aligns with RTP-LLM, which
            # benchmarks these as text (qwen_3_moe).
            extra_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

        self.llm = LLM(
            model=model,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=max_batch_size * max_model_len,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            additional_config={"gdn_prefill_backend": "triton"},
            **extra_kwargs,
        )

        if dp_barrier is not None:
            self._unpatch_executor()

        client = self.llm.llm_engine.engine_core
        engine_core = getattr(client, 'engine_core', client)
        self.scheduler = engine_core.scheduler
        self.executor = engine_core.model_executor

        self.block_size = self.scheduler.block_size
        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(self.block_size, sha256)

    def submit(
        self,
        batch_size: int,
        seq_len: int,
        max_tokens: int = 1,
        ignore_eos: bool = True,
    ) -> list[str]:
        """Add batch_size requests to the scheduler. Returns request IDs."""
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            temperature=0.0,
        )
        req_ids = []
        for _ in range(batch_size):
            req_id = str(uuid.uuid4())
            prompt_ids = [i % 128 + 1 for i in range(seq_len)]
            request = Request(
                request_id=req_id,
                prompt_token_ids=prompt_ids,
                sampling_params=sampling_params,
                pooling_params=None,
                block_hasher=self._block_hasher,
            )
            self.scheduler.add_request(request)
            req_ids.append(req_id)
        return req_ids

    @staticmethod
    def _patch_executor_for_dp_sync(barrier):
        """Monkey-patch Executor to add DP barriers during engine init.

        With DP+EP, engine init stages (profiling, CUDA graph capture) involve
        NCCL collectives across DP ranks. Independent DP rank processes may
        reach these stages at different times, causing NCCL mismatch deadlocks.
        Patching adds a barrier before each stage so all ranks enter together.
        """
        from vllm.v1.executor.abstract import Executor
        Executor._orig_determine_available_memory = (
            Executor.determine_available_memory)
        Executor._orig_initialize_from_config = (
            Executor.initialize_from_config)

        def _synced_determine(self):
            barrier.wait()
            return Executor._orig_determine_available_memory(self)

        def _synced_initialize(self, kv_cache_configs):
            barrier.wait()
            return Executor._orig_initialize_from_config(self, kv_cache_configs)

        Executor.determine_available_memory = _synced_determine
        Executor.initialize_from_config = _synced_initialize

    @staticmethod
    def _unpatch_executor():
        from vllm.v1.executor.abstract import Executor
        Executor.determine_available_memory = (
            Executor._orig_determine_available_memory)
        Executor.initialize_from_config = (
            Executor._orig_initialize_from_config)
        del Executor._orig_determine_available_memory
        del Executor._orig_initialize_from_config

    def _sync_dp(self):
        """Barrier across DP ranks before execute_model.

        With EP, all DP ranks must enter execute_model together so NCCL
        collectives (EP all-to-all, DP sync in dispatch_cg_and_sync_dp)
        and CUDA graph replays proceed in lockstep.
        """
        if self._dp_barrier is not None:
            self._dp_barrier.wait()

    def _execute_and_sample(self, scheduler_output):
        """Run execute_model + sample_tokens if needed (V2 model runner).

        Mirrors EngineCore.step() flow: execute_model may return None when
        using V2 model runner (deferred sampling), in which case we call
        sample_tokens separately. Both calls use non_block=True + .result()
        to properly unwrap AsyncModelRunnerOutput.
        """
        future = self.executor.execute_model(
            scheduler_output, non_block=True
        )
        model_output = future.result()
        if model_output is None:
            grammar_output = self.scheduler.get_grammar_bitmask(
                scheduler_output
            )
            future = self.executor.sample_tokens(
                grammar_output, non_block=True
            )
            model_output = future.result()
        return model_output

    def run_step(self) -> StepStat:
        """Schedule + timed execute + update. Returns step statistics.

        Timing covers execute_model + update_from_output (forward +
        sampler + dispatch), excluding schedule().
        """
        scheduler_output = self.scheduler.schedule()
        self._sync_dp()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model_output = self._execute_and_sample(scheduler_output)
        self.scheduler.update_from_output(scheduler_output, model_output)
        forward_ms = (time.perf_counter() - t0) * 1000

        return StepStat(
            forward_ms=forward_ms,
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_reqs=len(scheduler_output.num_scheduled_tokens),
            phase=_detect_phase(scheduler_output),
        )

    def run_step_no_timing(self) -> StepStat:
        """Schedule + execute + update without GPU sync timing."""
        scheduler_output = self.scheduler.schedule()
        self._sync_dp()
        model_output = self._execute_and_sample(scheduler_output)
        self.scheduler.update_from_output(scheduler_output, model_output)

        return StepStat(
            forward_ms=0.0,
            num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            num_reqs=len(scheduler_output.num_scheduled_tokens),
            phase=_detect_phase(scheduler_output),
        )

    def assert_phase(self, stat: StepStat, expected: str) -> None:
        if stat.phase != expected:
            raise AssertionError(
                f"Expected phase '{expected}', got '{stat.phase}' "
                f"(num_scheduled_tokens per req: "
                f"{stat.num_scheduled_tokens}/{stat.num_reqs})"
            )

    def mark_batch_start(self) -> None:
        """Record batch start time (after cuda sync). Aligns with RTP-LLM resetBeginTime."""
        torch.cuda.synchronize()
        self._batch_start = time.perf_counter()

    def mark_batch_end(self) -> float:
        """Record batch end time, return cost_time_ms. Aligns with RTP-LLM cost_time_us."""
        torch.cuda.synchronize()
        return (time.perf_counter() - self._batch_start) * 1000

    def mark_lap(self) -> float:
        """Elapsed ms since batch start (after cuda sync), without resetting.

        Splits a decode round into prefill (first token) vs decode without
        disturbing the batch-start clock, so prefill and cost share one
        origin. Aligns with RTP-LLM first_token_cost_time.
        """
        torch.cuda.synchronize()
        return (time.perf_counter() - self._batch_start) * 1000

    def start_profiling(self) -> None:
        """Signal nsys to start capture (requires --capture-range=cudaProfilerApi)."""
        torch.cuda.cudart().cudaProfilerStart()

    def stop_profiling(self) -> None:
        """Signal nsys to stop capture."""
        torch.cuda.cudart().cudaProfilerStop()

    @contextmanager
    def torch_profile(self, output_dir: str, trace_name: str = "vllm_bench"):
        """Wrap steps with torch.profiler for Kineto Chrome Trace output."""
        os.makedirs(output_dir, exist_ok=True)
        if os.environ.get("VLLM_CUSTOM_SCOPES_FOR_PROFILING") == "1":
            _force_vllm_scopes()
            if os.environ.get("SCOPE_DEBUG") == "1":
                self._debug_print_runner()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as prof:
            yield prof
        trace_path = os.path.join(output_dir, f"{trace_name}.json")
        prof.export_chrome_trace(trace_path)

    def _debug_print_runner(self) -> None:
        """Print the active model-runner class — V1 (scoped) vs V2 (no scopes)."""
        try:
            mr = self.executor.driver_worker.model_runner
            cls = type(mr)
            is_v1 = cls.__module__ == "vllm.v1.worker.gpu_model_runner"
            print(f"[scope] model_runner={cls.__module__}.{cls.__qualname__} "
                  f"({'V1 scoped' if is_v1 else 'V2 NO scopes'})")
        except Exception as e:
            print(f"[scope] runner probe failed: {e}")

    def profile_run(
        self,
        batch_size: int,
        seq_len: int,
        mode: str,
        num_steps: int,
        output_dir: str,
        skip_prefill_forward: bool = False,
    ) -> str:
        """Dedicated profiling pass — warmup, then capture a clean trace.

        For decode, prefill runs OUTSIDE the profiler window so the trace holds
        exactly ``num_steps`` decode steps (per-step averages are then exact).
        With ``skip_prefill_forward`` the prefill forward is skipped entirely
        (fake KV via submit_decode_only) — needed at large BS where a real
        prefill would OOM, and matches RTP-LLM's decode-only setup. For prefill,
        ``num_steps`` is 1. Trace name encodes mode/bs/seq/steps so
        perf_test_timeline can recover num_steps. Returns the trace path.
        """
        trace_name = f"vllm_{mode}_bs{batch_size}_seq{seq_len}_steps{num_steps}"

        if mode == "prefill":
            # warmup
            self.submit(batch_size, seq_len, max_tokens=1)
            self.run_step_no_timing()
            self.drain()
            self.submit(batch_size, seq_len, max_tokens=1)
            with self.torch_profile(output_dir, trace_name):
                self.run_step_no_timing()
            self.drain()
        elif skip_prefill_forward:
            # Decode-only: fake KV, no prefill forward at all.
            self.submit_decode_only(batch_size, seq_len, num_steps)
            for _ in range(min(2, num_steps)):
                self.run_step_no_timing()
            self.drain()
            self.submit_decode_only(batch_size, seq_len, num_steps)
            with self.torch_profile(output_dir, trace_name):
                for _ in range(num_steps):
                    self.run_step_no_timing()
            self.drain()
        else:
            # warmup: a full prefill + a couple decode steps
            self.submit(batch_size, seq_len,
                        max_tokens=num_steps + 1, ignore_eos=True)
            self.run_step_no_timing()
            for _ in range(min(2, num_steps)):
                self.run_step_no_timing()
            self.drain()
            # measured: prefill outside window, decode steps inside
            self.submit(batch_size, seq_len,
                        max_tokens=num_steps + 1, ignore_eos=True)
            self.run_step_no_timing()  # prefill (excluded from trace)
            with self.torch_profile(output_dir, trace_name):
                for _ in range(num_steps):
                    self.run_step_no_timing()
            self.drain()

        return os.path.join(output_dir, f"{trace_name}.json")

    def submit_decode_only(
        self,
        batch_size: int,
        seq_len: int,
        num_decode_steps: int,
    ) -> list[str]:
        """Submit requests and fast-forward past prefill without running forward.

        Allocates KV blocks via normal scheduling, registers requests in
        model_runner via _update_states, but skips the actual prefill forward.
        KV cache content is zeroed (not computed), matching RTP-LLM's
        setIsContextStream(false) behavior.
        """
        req_ids = self.submit(
            batch_size, seq_len,
            max_tokens=num_decode_steps + 1,
            ignore_eos=True,
        )

        scheduler_output = self.scheduler.schedule()

        self._register_without_forward(scheduler_output)
        self._fake_update_from_output(scheduler_output)

        return req_ids

    def _register_without_forward(self, scheduler_output):
        """Register requests in model_runner without running forward.

        Handles both V1 (_update_states) and V2 (finish/add/update) model runners.
        Only works with InprocExecutor (TP=1). MultiprocExecutor (TP>1) runs
        workers in subprocesses where driver_worker is not directly accessible.
        """
        if not hasattr(self.executor, 'driver_worker'):
            raise RuntimeError(
                "--skip-prefill-forward is not supported with TP>1. "
                "MultiprocExecutor does not expose driver_worker. "
                "Use normal prefill (remove --skip-prefill-forward) instead."
            )
        worker = self.executor.driver_worker
        model_runner = worker.model_runner
        if hasattr(model_runner, '_update_states'):
            model_runner._update_states(scheduler_output)
        else:
            model_runner.finish_requests(scheduler_output)
            model_runner.free_states(scheduler_output)
            model_runner.add_requests(scheduler_output)
            model_runner.update_requests(scheduler_output)
            model_runner.block_tables.apply_staged_writes()

    def _fake_update_from_output(self, scheduler_output):
        """Fabricate ModelRunnerOutput so scheduler advances past prefill."""
        from vllm.v1.outputs import ModelRunnerOutput

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        fake_token_ids = [[1]] * len(req_ids)

        fake_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=fake_token_ids,
        )
        self.scheduler.update_from_output(scheduler_output, fake_output)

    def drain(self) -> None:
        """Force-finish all requests and let model_runner clean up."""
        req_ids = list(self.scheduler.requests.keys())
        if req_ids:
            self.scheduler.finish_requests(req_ids, RequestStatus.FINISHED_ABORTED)
        # Must run schedule+execute even when no tokens are scheduled,
        # because model_runner.execute_model processes finished_req_ids
        # at the start (V2: finish_requests(); V1: _update_states())
        # to free request slots. Skipping this leaves stale slots.
        while self.scheduler.has_requests():
            so = self.scheduler.schedule()
            self._sync_dp()
            out = self._execute_and_sample(so)
            self.scheduler.update_from_output(so, out)


def _detect_phase(scheduler_output: SchedulerOutput) -> str:
    tokens = list(scheduler_output.num_scheduled_tokens.values())
    if not tokens:
        return "empty"
    if all(t == 1 for t in tokens):
        return "decode"
    if all(t > 1 for t in tokens):
        return "prefill"
    return "mixed"
