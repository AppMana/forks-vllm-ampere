# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

deepseek_kernels = importlib.import_module(
    "vllm.model_executor.layers.deepseek_v4_triton_kernels"
)
mhc_triton = importlib.import_module("vllm.model_executor.kernels.mhc.triton")
deepseek_v4_mhc_warmup = importlib.import_module(
    "vllm.model_executor.warmup.deepseek_v4_mhc_warmup"
)
fp8_einsum = importlib.import_module(
    "vllm.models.deepseek_v4.common.ops.fp8_einsum"
)


def test_mhc_prefill_token_counts_are_not_triton_specialized():
    assert mhc_triton._mhc_pre_fuse_triton_kernel.do_not_specialize == [
        "num_tokens"
    ]
    assert mhc_triton._mhc_post_triton_kernel.do_not_specialize == ["num_tokens"]
    assert mhc_triton._mhc_fused_post_prenorm_gemm_triton_kernel.do_not_specialize == [
        "num_tokens"
    ]


def test_mhc_prefill_num_split_is_bucketed_consistently():
    assert mhc_triton._bucket_mhc_pre_num_split(41) == 32
    assert mhc_triton._bucket_mhc_pre_num_split(27) == 16
    assert mhc_triton._bucket_mhc_pre_num_split(13) == 8
    assert mhc_triton._bucket_mhc_pre_num_split(1) == 1

    for split_k in range(1, 65):
        assert mhc_triton._bucket_mhc_pre_num_split(
            split_k
        ) == deepseek_v4_mhc_warmup._bucket_mhc_pre_num_split(split_k)


def test_deepseek_v4_request_extents_are_not_triton_specialized():
    assert deepseek_kernels._tf32_hc_prenorm_gemm_kernel.do_not_specialize == ["M"]
    assert deepseek_kernels._sparse_attention_bf16_kernel.do_not_specialize == [
        "num_tokens"
    ]
    assert deepseek_kernels._decode_sparse_attention_fp8_kernel.do_not_specialize == [
        "num_tokens"
    ]
    assert deepseek_kernels._deepseek_v4_fp8_einsum_triton_kernel.do_not_specialize == [
        "B"
    ]
    assert deepseek_kernels._fp8_mqa_logits_kernel.do_not_specialize == [
        "num_q",
        "seq_len_kv",
    ]
    assert deepseek_kernels._fp8_paged_mqa_logits_kernel.do_not_specialize == [
        "num_rows",
        "logits_width",
    ]
    assert deepseek_kernels._fp8_paged_mqa_logits_rowwise_kernel.do_not_specialize == [
        "num_rows",
        "logits_width",
    ]
    assert fp8_einsum._deepseek_v4_sm12x_fp8_einsum_kernel.do_not_specialize == [
        "num_tokens"
    ]
