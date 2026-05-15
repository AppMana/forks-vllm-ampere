# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

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
        self.device = torch.device("cpu")
        self.input_batch = SimpleNamespace(block_table=None)
        self.is_last_pp_rank = True
        self.max_num_tokens = 1
        self.model_config = SimpleNamespace(
            get_vocab_size=lambda: 0,
            dtype=torch.float32,
        )

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
    monkeypatch.setattr(
        kernel_warmup_module,
        "_finalize_triton_async_compiles",
        lambda: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module.torch.accelerator,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "_deepseek_v4_request_prep_warmup",
        lambda *args, **kwargs: None,
    )

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
            "num_tokens": 52,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        },
        {
            "num_tokens": 64,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_mixed_batch": True,
        },
        {
            "num_tokens": 52,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
        {
            "num_tokens": 64,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
        {
            "num_tokens": 1024,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
    ]


def test_kernel_warmup_clamps_deepseek_v4_sparse_mla_prefill_to_max_model_len(
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
    monkeypatch.setattr(
        kernel_warmup_module,
        "_finalize_triton_async_compiles",
        lambda: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module.torch.accelerator,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module,
        "_deepseek_v4_request_prep_warmup",
        lambda *args, **kwargs: None,
    )

    runner = _Runner("V4_FLASHMLA_SPARSE")
    runner.max_model_len = 512
    worker = _Worker(runner)
    worker.scheduler_config.max_num_batched_tokens = 8192

    kernel_warmup_module.kernel_warmup(worker)

    assert runner.calls[-3:] == [
        {
            "num_tokens": 52,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
        {
            "num_tokens": 64,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "create_single_prefill": True,
        },
        {
        "num_tokens": 512,
        "skip_eplb": True,
        "is_profile": True,
        "force_attention": True,
        "create_single_prefill": True,
        },
    ]


def test_kernel_warmup_runs_deepseek_v4_request_prep_warmup(
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
    monkeypatch.setattr(
        kernel_warmup_module,
        "_finalize_triton_async_compiles",
        lambda: None,
    )
    monkeypatch.setattr(
        kernel_warmup_module.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    warmup_calls: list[_Worker] = []
    monkeypatch.setattr(
        kernel_warmup_module,
        "_deepseek_v4_request_prep_warmup",
        warmup_calls.append,
        raising=False,
    )

    worker = _Worker(_Runner("V4_FLASHMLA_SPARSE"))
    kernel_warmup_module.kernel_warmup(worker)

    assert warmup_calls == [worker]


class _BlockTable:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def commit_block_table(self, num_reqs: int) -> None:
        self.calls.append(("commit", num_reqs, torch.is_inference_mode_enabled()))

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self.calls.append(
            (
                "compute",
                (
                    num_reqs,
                    tuple(query_start_loc.tolist()),
                    tuple(positions.tolist()),
                    torch.is_inference_mode_enabled(),
                ),
            )
        )


def test_deepseek_v4_request_prep_warmup_triggers_slot_mapping_and_bitmask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_prep_warmup = getattr(
        kernel_warmup_module,
        "_deepseek_v4_request_prep_warmup",
        None,
    )
    assert request_prep_warmup is not None

    monkeypatch.setattr(
        kernel_warmup_module.envs,
        "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        kernel_warmup_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    synchronize_calls = []
    monkeypatch.setattr(
        kernel_warmup_module.torch.accelerator,
        "synchronize",
        lambda: synchronize_calls.append(True),
    )

    bitmask_calls = []

    def _record_bitmask_warmup(
        scheduler_output, grammar_output, input_batch, logits
    ) -> None:
        bitmask_calls.append(
            (
                scheduler_output.scheduled_spec_decode_tokens,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask.shape,
                input_batch.req_ids,
                logits.shape,
            )
        )

    monkeypatch.setattr(
        kernel_warmup_module,
        "apply_grammar_bitmask",
        _record_bitmask_warmup,
    )
    prefill_metadata_calls = []
    monkeypatch.setattr(
        kernel_warmup_module,
        "_deepseek_v4_prefill_metadata_warmup",
        prefill_metadata_calls.append,
    )

    block_table = _BlockTable()
    runner = _Runner("V4_FLASHMLA_SPARSE")
    runner.device = torch.device("cpu")
    runner.input_batch = SimpleNamespace(block_table=block_table)
    runner.is_last_pp_rank = True
    runner.max_num_tokens = 512
    runner.model_config = SimpleNamespace(
        get_vocab_size=lambda: 65,
        dtype=torch.bfloat16,
    )
    worker = _Worker(runner)

    request_prep_warmup(worker)

    commit_calls = [call for call in block_table.calls if call[0] == "commit"]
    compute_calls = [call for call in block_table.calls if call[0] == "compute"]
    assert len(commit_calls) == len(
        kernel_warmup_module._DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS
    )
    assert all(call == ("commit", 1, True) for call in commit_calls)
    assert [call[1][1][1] for call in compute_calls] == list(
        kernel_warmup_module._DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS
    )
    assert all(call[1][3] is True for call in compute_calls)
    assert prefill_metadata_calls == [runner]
    assert bitmask_calls == [
        (
            {},
            ["_deepseek_v4_warmup_"],
            (1, 3),
            ["_deepseek_v4_warmup_"],
            torch.Size([1, 65]),
        ),
        (
            {},
            ["_deepseek_v4_warmup_"],
            (1, 3),
            ["_deepseek_v4_warmup_", "_deepseek_v4_warmup_unmasked_"],
            torch.Size([2, 65]),
        ),
        (
            {},
            ["_deepseek_v4_warmup_"],
            (1, 3),
            ["_deepseek_v4_warmup_"],
            torch.Size([1, 65]),
        ),
        (
            {},
            ["_deepseek_v4_warmup_"],
            (1, 3),
            ["_deepseek_v4_warmup_", "_deepseek_v4_warmup_unmasked_"],
            torch.Size([2, 65]),
        ),
    ]
    assert synchronize_calls == [True]


def test_deepseek_v4_post_capture_warmup_forces_metadata_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kernel_warmup_module.envs,
        "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
        True,
        raising=False,
    )
    calls = []
    monkeypatch.setattr(
        kernel_warmup_module,
        "_deepseek_v4_prefill_metadata_warmup",
        lambda runner, **kwargs: calls.append((runner, kwargs)),
    )
    finalize_calls = []
    monkeypatch.setattr(
        kernel_warmup_module,
        "_finalize_triton_async_compiles",
        lambda: finalize_calls.append(True),
    )

    worker = _Worker(_Runner("V4_FLASHMLA_SPARSE"))

    kernel_warmup_module.deepseek_v4_post_capture_request_prep_warmup(worker)

    assert calls == [(worker.model_runner, {"force_combine": True})]
    assert finalize_calls == [True]


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
