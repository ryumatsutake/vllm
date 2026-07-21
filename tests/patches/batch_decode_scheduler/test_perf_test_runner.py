from types import SimpleNamespace

import pytest

from vllm.patches.batch_decode_scheduler import perf_test_runner as runner
from vllm.patches.batch_decode_scheduler.perf_test_harness import (
    BenchHarness,
    StepStat,
)


class _Core:
    def __init__(self, error: Exception | None = None):
        self.calls = 0
        self.error = error

    def shutdown(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


def test_harness_shutdown_is_idempotent():
    core = _Core()
    harness = BenchHarness.__new__(BenchHarness)
    harness._shutdown = False
    harness.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=core),
    )

    harness.shutdown()
    harness.shutdown()

    assert core.calls == 1


class _FakeHarness:
    instances = []
    run_error: Exception | None = None
    shutdown_error: Exception | None = None

    def __init__(self, **kwargs):
        self.shutdown_calls = 0
        self.instances.append(self)

    def submit(self, *args, **kwargs):
        pass

    def run_step_no_timing(self):
        return StepStat(0.0, 1, 1, "decode")

    def run_step(self):
        if self.run_error is not None:
            raise self.run_error
        return StepStat(1.0, 128, 1, "prefill")

    def assert_phase(self, stat, expected):
        assert stat.phase == expected

    def assert_batch(self, stat, expected_bs):
        assert stat.num_reqs == expected_bs

    def assert_prefill_tokens(self, stat, expected_bs, expected_seq_len):
        assert stat.num_scheduled_tokens == expected_bs * expected_seq_len

    def mark_batch_start(self):
        pass

    def mark_batch_end(self):
        return 1.0

    def drain(self):
        pass

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _args():
    return SimpleNamespace(
        model="test-model",
        max_model_len=256,
        gpu_memory_utilization=0.1,
        dtype="auto",
        enforce_eager=True,
        tp_size=1,
        pp_size=1,
        enable_expert_parallel=False,
        disable_mm=False,
        worker_profile_dir=None,
        vllm_scopes=False,
        vllm_scopes_v1=False,
        scope_forward=False,
        profile=False,
        profile_output=None,
        partial=2,
        num_iters=1,
        num_warmup_iters=0,
        num_decode_steps=1,
        analyze=False,
        rtp_trace=None,
    )


@pytest.fixture(autouse=True)
def _reset_fake_harness(monkeypatch):
    _FakeHarness.instances = []
    _FakeHarness.run_error = None
    _FakeHarness.shutdown_error = None
    monkeypatch.setattr(runner, "BenchHarness", _FakeHarness)


def test_grid_shuts_down_after_success():
    results = runner._run_bench_grid(_args(), [1], [128])

    assert len(results) == 1
    assert _FakeHarness.instances[0].shutdown_calls == 1


def test_prefill_token_assertion_rejects_chunking():
    harness = BenchHarness.__new__(BenchHarness)
    full = StepStat(1.0, 256, 2, "prefill")
    chunked = StepStat(1.0, 128, 2, "prefill")

    harness.assert_prefill_tokens(full, 2, 128)
    with pytest.raises(AssertionError, match="prefill was chunked or clipped"):
        harness.assert_prefill_tokens(chunked, 2, 128)


def test_grid_shuts_down_and_preserves_benchmark_error(capsys):
    benchmark_error = RuntimeError("benchmark failed")
    _FakeHarness.run_error = benchmark_error
    _FakeHarness.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="benchmark failed") as exc_info:
        runner._run_bench_grid(_args(), [1], [128])

    assert exc_info.value is benchmark_error
    assert _FakeHarness.instances[0].shutdown_calls == 1
    stderr = capsys.readouterr().err
    assert "Harness shutdown also failed" in stderr
    assert "RuntimeError: shutdown failed" in stderr


def test_grid_fails_when_shutdown_fails():
    _FakeHarness.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        runner._run_bench_grid(_args(), [1], [128])


@pytest.mark.parametrize(
    ("rank_results", "rank_errors", "exitcodes", "forced", "match"),
    [
        ({0: []}, {}, {0: 0, 1: 0}, set(), "missing results from ranks [1]"),
        (
            {0: []},
            {1: RuntimeError("rank failed")},
            {0: 0, 1: 0},
            set(),
            "rank 1 failed: RuntimeError: rank failed",
        ),
        ({0: [], 1: []}, {}, {0: 0, 1: 7}, set(), "exit code is 7"),
        (
            {0: [], 1: []},
            {},
            {0: 0, 1: -9},
            {1},
            "forced process-group cleanup was required for ranks [1]",
        ),
    ],
)
def test_dp_failure_reasons(
    rank_results,
    rank_errors,
    exitcodes,
    forced,
    match,
):
    reasons = runner._dp_failure_reasons(
        2, rank_results, rank_errors, exitcodes, forced
    )

    assert any(match in reason for reason in reasons)


def test_dp_failure_reasons_accept_clean_completion():
    assert not runner._dp_failure_reasons(
        2,
        {0: [], 1: []},
        {},
        {0: 0, 1: 0},
        set(),
    )
