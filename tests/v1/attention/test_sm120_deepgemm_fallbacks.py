# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm_utils
from vllm.envs import environment_variables
from vllm.model_executor.layers.sparse_attn_indexer import (
    _decode_logits_width,
    _decode_topk_logits_width,
    _sparse_indexer_requires_deep_gemm,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def test_decode_logits_width_uses_active_context_bound():
    assert _decode_logits_width(262144, 1024) == 1024
    assert _decode_logits_width(4096, 8192) == 4096
    assert _decode_logits_width(4096, 0) == 4096
    assert _decode_logits_width(0, 1024) == 0


def test_decode_topk_logits_width_keeps_topk_kernel_width():
    assert _decode_topk_logits_width(262144, 1024, 512) == 1024
    assert _decode_topk_logits_width(262144, 128, 512) == 512
    assert _decode_topk_logits_width(300, 128, 512) == 300
    assert _decode_topk_logits_width(0, 128, 512) == 0


def test_sm120_sparse_indexer_does_not_require_deep_gemm(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )

    assert _sparse_indexer_requires_deep_gemm() is False


def test_non_sm120_cuda_sparse_indexer_still_requires_deep_gemm(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )

    assert _sparse_indexer_requires_deep_gemm() is True


def test_sm120_deepgemm_kernel_override_env_is_registered(
    monkeypatch: pytest.MonkeyPatch,
):
    env_name = "VLLM_DEEPSEEK_V4_USE_DEEPGEMM_SM12X_KERNELS"
    assert env_name in environment_variables
    monkeypatch.setenv(env_name, "1")
    assert environment_variables[env_name]()
    monkeypatch.setenv(env_name, "0")
    assert not environment_variables[env_name]()


def test_sm120_deepgemm_kernel_override_keeps_fp8_mqa_on_sm12x_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_USE_DEEPGEMM_SM12X_KERNELS", "1")
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )

    calls: list[str] = []
    mqa_result = torch.empty(1)
    paged_result = torch.empty(1)
    hc_result = torch.empty(1)

    def fake_mqa_impl(*args, **kwargs):
        raise AssertionError("SM120 FP8 MQA should stay on vLLM fallback")

    def fake_paged_impl(*args, **kwargs):
        raise AssertionError("SM120 FP8 paged MQA should stay on vLLM fallback")

    def fake_mqa_fallback(*args, **kwargs):
        calls.append("mqa_fallback")
        return mqa_result

    def fake_paged_fallback(*args, **kwargs):
        calls.append("paged_fallback")
        return paged_result

    def fake_hc_impl(*args, **kwargs):
        calls.append("hc")
        return hc_result

    monkeypatch.setattr(deep_gemm_utils, "_fp8_fp4_mqa_logits_impl", fake_mqa_impl)
    monkeypatch.setattr(
        deep_gemm_utils, "_fp8_fp4_paged_mqa_logits_impl", fake_paged_impl
    )
    monkeypatch.setattr(deep_gemm_utils, "_fp8_mqa_logits_sm12x", fake_mqa_fallback)
    monkeypatch.setattr(
        deep_gemm_utils, "_fp8_paged_mqa_logits_sm12x", fake_paged_fallback
    )
    monkeypatch.setattr(deep_gemm_utils, "_tf32_hc_prenorm_gemm_impl", fake_hc_impl)

    q = (torch.empty(1, 1, 1), None)
    kv = (torch.empty(1, 1), torch.empty(1))
    weights = torch.empty(1, 1)
    cu_seqlen = torch.empty(1, dtype=torch.int32)
    assert (
        deep_gemm_utils.fp8_fp4_mqa_logits(
            q, kv, weights, cu_seqlen, cu_seqlen, clean_logits=False
        )
        is mqa_result
    )

    kv_cache = torch.empty(1, 1, 1, 5, dtype=torch.uint8)
    context_lens = torch.empty(1, 1, dtype=torch.int32)
    block_tables = torch.empty(1, 1, dtype=torch.int32)
    schedule_metadata = torch.empty(1, dtype=torch.int32)
    assert (
        deep_gemm_utils.fp8_fp4_paged_mqa_logits(
            (torch.empty(1, 1, 1, 1), None),
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len=1,
            clean_logits=False,
        )
        is paged_result
    )

    assert (
        deep_gemm_utils.tf32_hc_prenorm_gemm(
            torch.empty(1, 1),
            torch.empty(1, 1),
            torch.empty(1, 1),
            torch.empty(1),
            num_split=1,
        )
        is hc_result
    )
    assert calls == ["mqa_fallback", "paged_fallback", "hc"]


def test_sm120_deepgemm_kernel_override_keeps_direct_topk_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_USE_DEEPGEMM_SM12X_KERNELS", "1")
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: capability == 120,
    )

    def fake_topk(*args, **kwargs):
        kwargs["out"].fill_(0)
        return kwargs["out"]

    monkeypatch.setattr(deep_gemm_utils, "_fp8_mqa_logits_topk_torch", fake_topk)

    q = (torch.empty(1, 1, 1), None)
    kv = (torch.empty(1, 1), torch.empty(1))
    weights = torch.empty(1, 1)
    cu_seqlen = torch.empty(1, dtype=torch.int32)
    topk_indices = torch.empty(1, 1, dtype=torch.int32)
    assert deep_gemm_utils.fp8_fp4_mqa_topk_indices(
        q, kv, weights, cu_seqlen, cu_seqlen, topk_indices
    )
    assert topk_indices.item() == 0


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_paged_mqa_direct_topk_matches_truncated_decode_width(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_USE_DEEPGEMM_SM12X_KERNELS", "1")
    torch.manual_seed(7)
    batch_size, next_n, num_heads, head_dim = 2, 2, 8, 32
    block_size, max_model_len, num_blocks = 4, 64, 16
    active_max_len = 13
    topk_tokens = 6
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm_utils, "_SM120_PAGED_MQA_TOPK_CHUNK_SIZE", 7)

    q = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn).contiguous()
    kv = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_scale = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv * kv_scale.reciprocal()).to(torch.float8_e4m3fn)
    fused_kv = torch.empty(
        num_blocks,
        block_size,
        1,
        head_dim + 4,
        device="cuda",
        dtype=torch.uint8,
    )
    fused_kv[..., :head_dim] = kv_fp8.view(torch.uint8)
    fused_kv[..., head_dim:] = kv_scale.contiguous().view(torch.uint8)

    weights = torch.randn(
        batch_size * next_n, num_heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[5, active_max_len], [9, 12]], device="cuda", dtype=torch.int32
    )
    block_tables = (
        torch.arange(
            batch_size * cdiv(max_model_len, block_size),
            device="cuda",
            dtype=torch.int32,
        ).reshape(batch_size, -1)
        % num_blocks
    )

    full_width_topk = torch.empty(
        batch_size * next_n, topk_tokens, device="cuda", dtype=torch.int32
    )
    truncated_width_topk = torch.empty_like(full_width_topk)

    assert deep_gemm_utils.fp8_fp4_paged_mqa_topk_indices(
        (q_fp8, None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        full_width_topk,
    )
    assert deep_gemm_utils.fp8_fp4_paged_mqa_topk_indices(
        (q_fp8, None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        active_max_len,
        truncated_width_topk,
    )

    torch.testing.assert_close(truncated_width_topk, full_width_topk, rtol=0, atol=0)
