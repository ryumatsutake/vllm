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
    ):
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
        )

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
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as prof:
            yield prof
        trace_path = os.path.join(output_dir, f"{trace_name}.json")
        prof.export_chrome_trace(trace_path)

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
