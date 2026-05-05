# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.envs import environment_variables
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _compute_mhc_pre_num_split,
    _select_mhc_warmup_token_sizes,
)


def test_deepseek_v4_mhc_warmup_envs_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP" in environment_variables
    assert "VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES" in environment_variables

    monkeypatch.delenv("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", raising=False)
    monkeypatch.delenv("VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES", raising=False)
    assert environment_variables["VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP"]() is True
    assert environment_variables["VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES"]() is None

    monkeypatch.setenv("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", "0")
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES", "1, 64,256")
    assert environment_variables["VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP"]() is False
    assert environment_variables["VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES"]() == [
        1,
        64,
        256,
    ]

    monkeypatch.setenv("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", "1")
    assert environment_variables["VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP"]() is True


def test_select_mhc_warmup_token_sizes_deduplicates_pre_split_buckets() -> None:
    requested_sizes = [1, 2, 64, 65, 512, 1024]
    selected = _select_mhc_warmup_token_sizes(
        max_tokens=1024,
        hidden_size=1024,
        hc_mult=4,
        num_sms=16,
        requested_token_sizes=requested_sizes,
        cudagraph_capture_sizes=[],
    )

    assert selected == [1, 65, 512, 1024]
    selected_splits = [
        _compute_mhc_pre_num_split(
            num_tokens=size,
            hidden_size=1024,
            hc_mult=4,
            num_sms=16,
        )
        for size in selected
    ]
    assert selected_splits == [16, 8, 2, 1]


def test_select_mhc_warmup_token_sizes_includes_capture_and_max_sizes() -> None:
    selected = _select_mhc_warmup_token_sizes(
        max_tokens=2048,
        hidden_size=1024,
        hc_mult=4,
        num_sms=16,
        requested_token_sizes=None,
        cudagraph_capture_sizes=[3, 129],
    )

    assert 129 in selected
    assert 2048 in selected
    assert selected == sorted(selected)
