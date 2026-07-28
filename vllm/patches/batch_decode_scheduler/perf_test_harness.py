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
    from torch.autograd.profiler import record_function

    import vllm.v1.utils as U

    U._PROFILER_FUNC = record_function


# V2 model-runner scope injection (see _install_scope_patches).
#
# The V2 runner (vllm/v1/worker/gpu/model_runner.py, default for
# Qwen3/Llama/Mistral) has NO record_function scopes, unlike the legacy V1
# runner. We monkey-patch its granular methods to emit the SAME scope names the
# V1 runner uses, so perf_test_timeline's scope_breakdown/compare treat V1 and
# V2 traces identically. (name, owner, attr): owner "self" means the runner
# instance; otherwise a runner attribute holding the target object.
_V2_SCOPE_METHODS: list[tuple[str, str, str]] = [
    ("gpu_model_runner: preprocess", "self", "prepare_inputs"),
    ("gpu_model_runner: sample", "self", "sample"),
    ("gpu_model_runner: postprocess", "self", "postprocess_sampled"),
]
# forward is inlined in execute_model across three mutually-exclusive branches
# (FULL cudagraph / PIECEWISE / eager), so there is no single runner forward
# method. Wrap all three call targets with the same "forward" name; only one
# fires per step. run_pw_graph internally re-invokes model.forward, so the
# forward wrap is reentrancy-guarded to avoid a nested double-count.
_V2_FORWARD_TARGETS: list[tuple[str, str]] = [
    ("cudagraph_manager", "run_fullgraph"),
    ("cudagraph_manager", "run_pw_graph"),
    ("model", "forward"),
]


def _make_scoped(fn, scope_name: str, reentry_guard=None):
    """Wrap a bound method so its body runs inside record_function(scope_name).

    Args:
        fn: the original bound method.
        scope_name: the scope label to emit.
        reentry_guard: optional (obj, attr) pair. When already inside a
            same-name scope (attr truthy), the wrapper skips the scope and just
            calls fn — prevents nested double-count for the piecewise forward
            path (run_pw_graph -> model.forward).
    """
    import vllm.v1.utils as U

    def wrapper(*args, **kwargs):
        if reentry_guard is not None:
            obj, attr = reentry_guard
            if getattr(obj, attr, False):
                return fn(*args, **kwargs)
            setattr(obj, attr, True)
            try:
                with U.record_function_or_nullcontext(scope_name):
                    return fn(*args, **kwargs)
            finally:
                setattr(obj, attr, False)
        with U.record_function_or_nullcontext(scope_name):
            return fn(*args, **kwargs)

    wrapper._scoped = True
    wrapper._orig = fn
    return wrapper


def _install_scope_patches(worker, wrap_forward: bool = False) -> None:
    """Install record_function scopes inside a worker process.

    Runs on every rank via ``executor.collective_rpc`` (TP=1 live call, TP>1
    cloudpickle over spawn/fork). Two effects, both process-local:

    1. Force ``vllm.v1.utils._PROFILER_FUNC = record_function`` so scopes emit
       regardless of first-call timing (the module global freezes on first
       call; the driver-side _force_vllm_scopes can't reach worker processes).
    2. Monkey-patch the V2 runner's granular methods to wrap them in scopes
       mirroring the V1 names. The V1 runner already has native scopes, so we
       detect it and only do step 1 (never double-wrap).

    Idempotent. Must be module-level so cloudpickle can resolve it under spawn.
    """
    from torch.autograd.profiler import record_function

    import vllm.v1.utils as U

    U._PROFILER_FUNC = record_function

    mr = getattr(worker, "model_runner", None)
    if mr is None:
        return

    # V1 runner already emits gpu_model_runner: scopes natively.
    if type(mr).__module__ == "vllm.v1.worker.gpu_model_runner":
        return

    if getattr(mr, "_scopes_installed", False):
        return

    for scope_name, owner, attr in _V2_SCOPE_METHODS:
        target = mr if owner == "self" else getattr(mr, owner, None)
        if target is None:
            continue
        fn = getattr(target, attr, None)
        if fn is None or getattr(fn, "_scoped", False):
            continue
        setattr(target, attr, _make_scoped(fn, scope_name))

    # Output dispatch (D2H): mirror V1's "gpu_model_runner: ModelRunnerOutput"
    # scope. In V2 the output object + async D2H copy is set up in
    # AsyncOutput.__init__ (inlined in sample_tokens, not a runner method), so
    # wrap the class initializer — this is the analogue of RTP's dispatch_output.
    try:
        from vllm.v1.worker.gpu.async_utils import AsyncOutput
        if not getattr(AsyncOutput.__init__, "_scoped", False):
            AsyncOutput.__init__ = _make_scoped(
                AsyncOutput.__init__, "gpu_model_runner: ModelRunnerOutput"
            )
    except Exception:
        pass

    if wrap_forward:
        mr._in_forward_scope = False
        for owner, attr in _V2_FORWARD_TARGETS:
            target = mr if owner == "self" else getattr(mr, owner, None)
            if target is None:  # cudagraph_manager is None under enforce_eager
                continue
            fn = getattr(target, attr, None)
            if fn is None or getattr(fn, "_scoped", False):
                continue
            setattr(
                target, attr,
                _make_scoped(fn, "gpu_model_runner: forward",
                             reentry_guard=(mr, "_in_forward_scope")),
            )

    mr._scopes_installed = True


# Token id fed to the scheduler AND written into each rank's runner state as
# the "sampled" first token of the fake prefill (submit_decode_only).
_FAKE_TOKEN_ID = 1

# Token-budget cap for the fake-KV setup (submit_decode_only). Registration
# needs no forward, so the budget only sizes runner buffers (inputs_embeds is
# max_num_batched_tokens x hidden on BOTH pinned host and device memory) and
# the profile_run activation reservation. bs x seq at large shapes made those
# allocations fail before a single decode step could run; prompts register
# over multiple scheduler rounds instead. Cap at 64Ki tokens: this host
# rejects single pinned allocations above ~1 GiB (cudaHostAlloc invalid
# argument), and 64Ki x hidden 5120 x bf16 = 640 MiB stays safely below
# that; longer prompts are admitted through the max_seq_len floor below.
_FAKE_KV_TOKEN_BUDGET_CAP = 65536


def _register_requests_no_forward(worker, scheduler_output) -> None:
    """Register scheduled requests in a worker's model_runner without forward.

    Runs on every rank via ``executor.collective_rpc`` (TP=1 UniProc live
    call; TP>1 MultiprocExecutor ships this function via cloudpickle and
    invokes it as ``func(worker, scheduler_output)``). SchedulerOutput is
    already pickled over the same broadcast MQ on every normal step, so
    shipping it here is equally safe.

    Mirrors the state-update prologue of execute_model for both runners
    (V1: _update_states; V2: finish/free/add/update + staged block-table
    writes) so KV blocks are registered but no forward ever runs.

    CRITICAL: the first decode step reads the "sampled" token of the skipped
    prefill from runner-local state that is normally written by the sampler
    path we skipped. Leaving it unwritten reads uninitialized memory — V1's
    token_ids_cpu slot held heap garbage -> embedding index out-of-bounds
    (device-side assert, observed on TP=1). Both branches below therefore
    also mirror the post-sampling bookkeeping with _FAKE_TOKEN_ID.
    Must be module-level so cloudpickle can resolve it under spawn.
    """
    model_runner = worker.model_runner
    if hasattr(model_runner, '_update_states'):
        model_runner._update_states(scheduler_output)
        # Mirror gpu_model_runner's post-sampling bookkeeping (execute_model
        # writes sampled ids into input_batch on the last PP rank; the
        # scheduler never sends them back).
        input_batch = model_runner.input_batch
        for req_id in scheduler_output.num_scheduled_tokens:
            idx = input_batch.req_id_to_index[req_id]
            start = int(input_batch.num_tokens_no_spec[idx])
            input_batch.token_ids_cpu[idx, start] = _FAKE_TOKEN_ID
            input_batch.is_token_ids[idx, start] = True
            input_batch.num_tokens_no_spec[idx] = start + 1
            model_runner.requests[req_id].output_token_ids.append(_FAKE_TOKEN_ID)
    else:
        model_runner.finish_requests(scheduler_output)
        model_runner.free_states(scheduler_output)
        model_runner.add_requests(scheduler_output)
        model_runner.update_requests(scheduler_output)
        model_runner.block_tables.apply_staged_writes()
        # V2 keeps the decode input token in req_states.last_sampled_tokens
        # (zero-init, so it would read token 0 — valid but arbitrary). Write
        # the same fake token the scheduler was fed, for determinism.
        req_states = model_runner.req_states
        for req_id in scheduler_output.num_scheduled_tokens:
            idx = req_states.req_id_to_index[req_id]
            req_states.last_sampled_tokens[idx : idx + 1] = _FAKE_TOKEN_ID


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
        max_seq_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        fake_kv_registration: bool = False,
        dp_barrier=None,
        disable_mm: bool = False,
        worker_profiler_dir: str | None = None,
        inject_scopes: bool = False,
        scope_forward: bool = False,
    ):
        self._dp_barrier = dp_barrier
        self._shutdown = False

        # Token budget must admit the largest single-step batch without chunking:
        # both real prefill and the fake-KV decode-only setup go through a real
        # scheduler.schedule() that prefills batch * seq_len tokens in one step.
        # The bound is max_batch_size * max_seq_len (the largest seq_len across
        # the grid), NOT max_batch_size * max_model_len. profile_run does a
        # skip_attn dummy forward at exactly this many tokens (gpu_worker.py's
        # determine_available_memory), so an over-large budget inflates the
        # measured activation peak, steals KV memory, and can OOM profiling at
        # large batch. Floor at max_model_len: vLLM rejects
        # max_num_batched_tokens < max_model_len when chunked prefill is off
        # (config/scheduler.py verify_max_model_len).
        if max_seq_len is None:
            max_seq_len = max_model_len
        max_num_batched_tokens = max(max_batch_size * max_seq_len, max_model_len)
        if fake_kv_registration:
            # Decode-only (--partial 1): no prefill forward ever runs, so the
            # budget must merely admit one full prompt per scheduler round
            # alongside the already-running decode tokens. Keep it capped or
            # the runner's per-token buffers and profile_run's activation
            # reservation scale with bs x seq and OOM before decode starts.
            max_num_batched_tokens = max(
                min(max_num_batched_tokens, _FAKE_KV_TOKEN_BUDGET_CAP),
                max_model_len,
                max_seq_len + max_batch_size + 64,
            )

        if dp_barrier is not None:
            self._patch_executor_for_dp_sync(dp_barrier)

        extra_kwargs = {}
        if worker_profiler_dir:
            # Use vLLM's built-in WorkerProfiler (torch backend). Unlike the
            # harness's own torch_profile (single-process, TP=1 only), this fans
            # out to every TP/DP rank via collective_rpc, so each rank dumps its
            # own trace — needed for TP>1 where the driver-process profiler can't
            # see the worker subprocesses.
            from vllm.config.profiler import ProfilerConfig
            extra_kwargs["profiler_config"] = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=worker_profiler_dir,
            )
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
            max_num_batched_tokens=max_num_batched_tokens,
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

        if inject_scopes:
            # Fan out to every rank (TP=1 UniProc live call; TP>1 cloudpickle).
            # Forces the profiler-func freeze and adds record_function scopes to
            # the V2 runner, in-process. Must run after model load (model_runner
            # exists) and before any profiled steps.
            self.executor.collective_rpc(
                _install_scope_patches, args=(scope_forward,)
            )

    def shutdown(self) -> None:
        """Shut down EngineCore and its workers exactly once."""
        if self._shutdown:
            return
        self._shutdown = True
        self.llm.llm_engine.engine_core.shutdown()

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

    def assert_batch(self, stat: StepStat, expected_bs: int) -> None:
        """Assert schedule() batched all expected_bs requests in this step.

        assert_phase only checks the phase of the *scheduled subset*, so a
        partially-scheduled batch (e.g. allocate_slots returns None on KV-cache
        OOM → scheduler.py breaks early, leaving requests in waiting) still
        looks like a clean all-prefill/all-decode step and passes phase checks.
        The timing is then measured on a smaller batch than requested, silently
        corrupting the result. This guards against that: a benchmark that can't
        fit the requested batch must fail loudly, not report wrong-batch data.
        """
        if stat.num_reqs != expected_bs:
            raise AssertionError(
                f"Expected {expected_bs} reqs scheduled together, got "
                f"{stat.num_reqs} — batch was split (KV-cache OOM or "
                f"scheduler budget clip). Lower --batch-sizes/--seq-lens or "
                f"raise --gpu-memory-utilization."
            )

    def assert_prefill_tokens(
        self,
        stat: StepStat,
        expected_bs: int,
        expected_seq_len: int,
    ) -> None:
        """Assert prefill completed the entire batch in one scheduler step."""
        expected_tokens = expected_bs * expected_seq_len
        if stat.num_scheduled_tokens != expected_tokens:
            raise AssertionError(
                f"Expected {expected_tokens} prefill tokens in one step, got "
                f"{stat.num_scheduled_tokens}; prefill was chunked or clipped."
            )

    def mark_batch_start(self) -> None:
        """Record batch start after cuda sync, aligned with RTP resetBeginTime."""
        torch.cuda.synchronize()
        self._batch_start = time.perf_counter()

    def mark_batch_end(self) -> float:
        """Record batch end and return ms, aligned with RTP cost_time_us."""
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

    def start_worker_profile(self) -> None:
        """Start vLLM's built-in WorkerProfiler on every TP/DP rank."""
        self.llm.start_profile()

    def stop_worker_profile(self) -> None:
        """Stop WorkerProfiler; each rank dumps its trace to torch_profiler_dir."""
        self.llm.stop_profile()

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
        if not hasattr(self.executor, "driver_worker"):
            # MultiprocExecutor (TP>1) runs workers in subprocesses; the driver
            # has no in-process worker to probe.
            print("[scope] runner probe skipped (TP>1, no driver_worker)")
            return
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
        every rank's model_runner via collective_rpc, but skips the actual
        prefill forward. KV cache content is zeroed (not computed), matching
        RTP-LLM's setIsContextStream(false) behavior. Works for TP=1 and
        TP>1 (the registration fans out to all worker processes).
        """
        req_ids = self.submit(
            batch_size, seq_len,
            # registration may take up to batch_size scheduler rounds; every
            # round fake-advances the already-registered requests by one
            # token, so pad the allowance to keep the measured decode steps
            # from finishing anyone early (ignore_eos + explicit drain).
            max_tokens=num_decode_steps + batch_size + 1,
            ignore_eos=True,
        )

        # With the fake-KV token budget capped, one schedule() round may only
        # admit part of the batch; loop until every request is registered.
        pending = set(req_ids)
        stalled_rounds = 0
        while pending:
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.num_scheduled_tokens:
                self._register_without_forward(scheduler_output)
                self._fake_update_from_output(scheduler_output)
            newly_registered = pending & set(
                scheduler_output.num_scheduled_tokens
            )
            if newly_registered:
                pending -= newly_registered
                stalled_rounds = 0
                continue
            stalled_rounds += 1
            if stalled_rounds > 16:
                raise AssertionError(
                    f"Expected {batch_size} reqs registered for fake-KV "
                    f"decode, {len(pending)} still waiting after "
                    f"{stalled_rounds} stalled rounds — batch was split "
                    f"(KV-cache OOM). Lower --batch-sizes/--seq-lens or "
                    f"raise --gpu-memory-utilization."
                )

        return req_ids

    def _register_without_forward(self, scheduler_output):
        """Register requests in every rank's model_runner without forward.

        Fans out to all workers via collective_rpc — TP=1 (UniProc, live
        call) and TP>1 (MultiprocExecutor, cloudpickle over the broadcast
        MQ) both work; each TP rank must register the requests so its
        per-rank runner state (block tables, seq lens) matches the
        scheduler before the first decode step.
        """
        self.executor.collective_rpc(
            _register_requests_no_forward, args=(scheduler_output,)
        )

    def _fake_update_from_output(self, scheduler_output):
        """Fabricate ModelRunnerOutput so scheduler advances past prefill."""
        from vllm.v1.outputs import ModelRunnerOutput

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        # One list per request — [[x]] * n would alias a single inner list
        # across all requests, a hazard if downstream ever mutates in place.
        fake_token_ids = [[_FAKE_TOKEN_ID] for _ in req_ids]

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
