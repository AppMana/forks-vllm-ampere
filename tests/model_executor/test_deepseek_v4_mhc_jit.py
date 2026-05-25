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
kernel_warmup = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
fp8_einsum = importlib.import_module(
    "vllm.models.deepseek_v4.common.ops.fp8_einsum"
)


def test_mhc_prefill_token_counts_are_not_triton_specialized():
    assert mhc_triton._mhc_pre_fuse_triton_kernel.do_not_specialize == [
        "num_tokens",
        "gemm_stride_s",
        "sq_stride_s",
    ]
    assert mhc_triton._mhc_post_triton_kernel.do_not_specialize == ["num_tokens"]
    assert mhc_triton._mhc_fused_post_prenorm_gemm_triton_kernel.do_not_specialize == [
        "num_tokens",
        "gemm_stride_s",
        "sq_stride_s",
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


def test_default_mhc_warmup_sizes_are_bucket_or_branch_representatives():
    token_sizes = deepseek_v4_mhc_warmup._select_mhc_warmup_token_sizes(
        max_tokens=16_384,
        hidden_size=7168,
        hc_mult=4,
        num_sms=82,
        requested_token_sizes=None,
        cudagraph_capture_sizes=[],
    )

    assert len(token_sizes) <= 20
    assert 2 in token_sizes
    assert 8 in token_sizes
    assert 16 in token_sizes

    warmed_splits = {
        deepseek_v4_mhc_warmup._compute_mhc_pre_num_split(
            num_tokens=size,
            hidden_size=7168,
            hc_mult=4,
            num_sms=82,
        )
        for size in token_sizes
    }
    assert warmed_splits == {1, 2, 4, 8, 16, 32}


def test_deepseek_v4_warmup_token_lists_avoid_prompt_exact_shapes():
    prompt_exact_sizes = {7, 27, 31, 34, 52, 60, 68, 132, 136, 142, 188}
    warmup_lists = (
        kernel_warmup._DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS,
        kernel_warmup._DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS,
        kernel_warmup._DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS,
        kernel_warmup._DEEPSEEK_V4_REQUEST_PREP_WARMUP_TOKENS,
        kernel_warmup._DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_QUERY_TOKENS,
    )

    for warmup_list in warmup_lists:
        assert not (set(warmup_list) & prompt_exact_sizes)


def test_deepseek_v4_request_extents_are_not_triton_specialized():
    assert deepseek_kernels._tf32_hc_prenorm_gemm_kernel.do_not_specialize == [
        "M",
        "stride_outs",
        "stride_sqs",
    ]
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
