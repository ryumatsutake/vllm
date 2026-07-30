# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def test_fake_balance_template_matches_rtp_ep_sequence():
    template = moe_runner._make_fake_balance_template(
        capacity=10,
        num_experts=128,
        ep_size=4,
        dp_rank=0,
        dp_size=1,
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    assert template.tolist() == [0, 32, 64, 96, 1, 33, 65, 97, 2, 34]


def test_fake_balance_template_matches_rtp_dp_offset():
    template = moe_runner._make_fake_balance_template(
        capacity=9,
        num_experts=16,
        ep_size=4,
        dp_rank=1,
        dp_size=2,
        dtype=torch.int64,
        device=torch.device("cpu"),
    )

    assert template.dtype == torch.int64
    assert template.tolist() == [10, 14, 2, 6, 11, 15, 3, 7, 8]


def test_independent_dp_uses_benchmark_rank(monkeypatch):
    monkeypatch.setenv("FAKE_BALANCE_DP_RANK", "2")
    monkeypatch.setenv("FAKE_BALANCE_DP_SIZE", "4")

    assert moe_runner._read_fake_balance_dp_config(0, 1) == (2, 4)
    assert moe_runner._read_fake_balance_dp_config(1, 2) == (1, 2)


@pytest.mark.parametrize(
    ("supports_internal_mk", "is_sp", "dp_size", "ep_size", "pcp_size", "factor"),
    [
        (True, False, 2, 4, 1, 1),
        (False, False, 2, 4, 1, 2),
        (False, True, 2, 4, 1, 4),
        (False, False, 2, 4, 3, 6),
    ],
)
def test_fake_balance_capacity_accounts_for_dispatch(
    supports_internal_mk,
    is_sp,
    dp_size,
    ep_size,
    pcp_size,
    factor,
):
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(supports_internal_mk=supports_internal_mk)
    )
    runner.moe_config = SimpleNamespace(
        max_num_tokens=64,
        experts_per_token=8,
        dp_size=dp_size,
        moe_parallel_config=SimpleNamespace(
            dp_size=dp_size,
            ep_size=ep_size,
            pcp_size=pcp_size,
            is_sequence_parallel=is_sp,
        ),
    )

    assert runner._fake_balance_capacity() == 64 * 8 * factor


class _Router:
    def __init__(self):
        self.calls = 0

    def select_experts(self, hidden_states, router_logits, **kwargs):
        self.calls += 1
        shape = (hidden_states.shape[0], 2)
        return torch.zeros(shape), torch.full(shape, 7, dtype=torch.int64)


def test_constructor_only_marks_fake_balance_and_disables_output_scale(monkeypatch):
    monkeypatch.setenv("FAKE_BALANCE_EXPERT", "1")
    with set_current_vllm_config(VllmConfig()):
        runner = MoERunner(
            layer_name="test.fake_balance",
            moe_config=SimpleNamespace(),
            router=_Router(),
            routed_experts=SimpleNamespace(),
            routed_scaling_factor=2.5,
        )

    assert runner._fake_balance_enabled
    assert not runner._fake_balance_finalized
    assert runner._fake_balance_template is None
    assert runner.routed_scaling_factor == 1.0


def _make_runner(monolithic: bool = False) -> MoERunner:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    parallel_config = SimpleNamespace(
        dp_rank=0,
        dp_size=1,
        ep_size=2,
        pcp_size=1,
        is_sequence_parallel=False,
        enable_eplb=False,
    )
    quant_method = SimpleNamespace(
        is_monolithic=monolithic,
        method_name="TestMoEMethod",
        topk_indices_dtype=torch.int32,
        supports_internal_mk=True,
    )
    runner.moe_config = SimpleNamespace(
        max_num_tokens=4,
        experts_per_token=2,
        num_experts=8,
        num_logical_experts=8,
        dp_size=1,
        device="cuda",
        moe_parallel_config=parallel_config,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method,
        expert_map_manager=SimpleNamespace(placement_strategy="linear"),
    )
    runner.router = _Router()
    runner.layer_name = "model.layers.0.mlp.experts"
    runner._fake_balance_enabled = True
    runner._fake_balance_finalized = False
    runner._fake_balance_template = None
    runner._fake_balance_template_key = None
    runner._fake_balance_select_experts = None
    runner._fake_balance_original_select_experts = None
    return runner


@pytest.fixture
def fake_cuda(monkeypatch):
    original_make_template = moe_runner._make_fake_balance_template
    monkeypatch.setattr(moe_runner.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = {"make_template": 0}

    def make_cpu_template(**kwargs):
        calls["make_template"] += 1
        kwargs["device"] = torch.device("cpu")
        return original_make_template(**kwargs)

    monkeypatch.setattr(moe_runner, "_make_fake_balance_template", make_cpu_template)
    return calls


def test_finalize_installs_idempotent_post_select_wrapper(fake_cuda):
    runner = _make_runner()
    moe_runner._FAKE_BALANCE_TEMPLATE_CACHE.clear()

    runner.finalize_fake_balance()
    wrapper = runner.router.select_experts
    runner.finalize_fake_balance()

    assert fake_cuda["make_template"] == 1
    assert runner.router.select_experts is wrapper
    weights, ids = runner.router.select_experts(
        torch.zeros(3, 4), torch.zeros(3, 8)
    )
    assert runner.router.calls == 1
    assert ids.dtype == torch.int64
    assert ids.tolist() == [[0, 4], [1, 5], [2, 6]]
    assert torch.equal(weights, torch.ones_like(weights))


def test_finalize_shares_template_across_layers(fake_cuda):
    first = _make_runner()
    second = _make_runner()
    second.layer_name = "model.layers.1.mlp.experts"
    moe_runner._FAKE_BALANCE_TEMPLATE_CACHE.clear()

    first.finalize_fake_balance()
    second.finalize_fake_balance()

    assert fake_cuda["make_template"] == 1
    assert first._fake_balance_template is second._fake_balance_template


def test_runtime_rejects_shape_larger_than_template(fake_cuda):
    runner = _make_runner()
    moe_runner._FAKE_BALANCE_TEMPLATE_CACHE.clear()
    runner.finalize_fake_balance()

    with pytest.raises(RuntimeError, match="template is too small"):
        runner.router.select_experts(torch.zeros(5, 4), torch.zeros(5, 8))


def test_finalize_rejects_monolithic_kernel(monkeypatch):
    runner = _make_runner(monolithic=True)
    monkeypatch.setattr(moe_runner.current_platform, "is_cuda", lambda: True)

    with pytest.raises(RuntimeError, match="requires a modular MoE kernel"):
        runner.finalize_fake_balance()


def test_finalize_rejects_round_robin_expert_placement(monkeypatch):
    runner = _make_runner()
    runner.routed_experts.expert_map_manager.placement_strategy = "round_robin"
    monkeypatch.setattr(moe_runner.current_platform, "is_cuda", lambda: True)

    with pytest.raises(RuntimeError, match="requires linear expert placement"):
        runner.finalize_fake_balance()


def test_finalize_rejects_eplb(monkeypatch):
    runner = _make_runner()
    runner.moe_config.moe_parallel_config.enable_eplb = True
    monkeypatch.setattr(moe_runner.current_platform, "is_cuda", lambda: True)

    with pytest.raises(RuntimeError, match="does not support EPLB"):
        runner.finalize_fake_balance()


def test_finalize_rejects_initialization_during_capture(monkeypatch):
    runner = _make_runner()
    moe_runner._FAKE_BALANCE_TEMPLATE_CACHE.clear()
    monkeypatch.setattr(moe_runner.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        runner.finalize_fake_balance()
