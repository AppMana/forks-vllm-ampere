# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import kernel_warmup as kernel_warmup_module


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class _Runner:
    is_pooling_model = False

    def __init__(self, backend_name: str) -> None:
        self.attn_groups = [[SimpleNamespace(backend=_Backend(backend_name))]]
        self.calls: list[dict[str, object]] = []

    def _dummy_run(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _Worker:
    def __init__(self, runner: _Runner) -> None:
        self.model_runner = runner
        self.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)
        self.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 16, 128]),
            kernel_config=SimpleNamespace(enable_flashinfer_autotune=False),
        )

    def get_model(self) -> object:
        return object()


def test_kernel_warmup_runs_deepseek_v4_sparse_mla_dummy_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_warmup_module.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(
        kernel_warmup_module.envs,
        "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "deepseek_v4_mhc_warmup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(kernel_warmup_module, "has_flashinfer", lambda: False)

    runner = _Runner("V4_FLASHMLA_SPARSE")
    kernel_warmup_module.kernel_warmup(_Worker(runner))

    assert runner.calls == [
        {
            "num_tokens": 16,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        },
        {
            "num_tokens": 1024,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
    ]


def test_kernel_warmup_skips_deepseek_v4_sparse_mla_dummy_attention_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel_warmup_module.envs, "VLLM_USE_DEEP_GEMM", False)
    monkeypatch.setattr(
        kernel_warmup_module.envs,
        "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "deepseek_v4_mhc_warmup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(kernel_warmup_module, "has_flashinfer", lambda: False)

    runner = _Runner("V4_FLASHMLA_SPARSE")
    kernel_warmup_module.kernel_warmup(_Worker(runner))

    assert runner.calls == []
