# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from types import SimpleNamespace

from vllm.utils import deep_gemm
from vllm.model_executor.layers.sparse_attn_indexer import (
    SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH,
    SM120_SHORT_ROW_TOPK_MAX_WIDTH,
    _should_use_sm120_short_row_topk_decode,
)


@pytest.mark.parametrize(
    ("topk_tokens", "logits_width", "num_rows", "is_cuda_sm120", "expected"),
    [
        (512, SM120_SHORT_ROW_TOPK_ALWAYS_WIDTH, 32, True, True),
        (512, 8192, 16, True, True),
        (512, 8192, 32, True, True),
        (512, 12288, 32, True, False),
        (512, SM120_SHORT_ROW_TOPK_MAX_WIDTH, 1, True, False),
        (512, 4096, 1, False, False),
        (2048, 4096, 1, True, False),
    ],
)
def test_sm120_short_row_topk_decode_selector(
    topk_tokens: int,
    logits_width: int,
    num_rows: int,
    is_cuda_sm120: bool,
    expected: bool,
) -> None:
    assert (
        _should_use_sm120_short_row_topk_decode(
            topk_tokens,
            logits_width,
            num_rows,
            is_cuda_sm120,
        )
        is expected
    )


def test_fp8_mqa_direct_topk_is_enabled_on_ampere(monkeypatch) -> None:
    called = False

    def fake_topk(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_topk_torch", fake_topk)
    monkeypatch.setattr(deep_gemm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deep_gemm.current_platform,
        "is_device_capability_family",
        lambda family: family == 80,
    )

    assert deep_gemm.fp8_fp4_mqa_topk_indices(
        (object(), None),
        (object(), object()),
        object(),
        object(),
        object(),
        SimpleNamespace(shape=(1, 512)),
    )
    assert called


def test_fp8_mqa_direct_topk_still_rejects_fp4_q_on_ampere(monkeypatch) -> None:
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deep_gemm.current_platform,
        "is_device_capability_family",
        lambda family: family == 80,
    )

    assert not deep_gemm.fp8_fp4_mqa_topk_indices(
        (object(), object()),
        (object(), object()),
        object(),
        object(),
        object(),
        object(),
    )
