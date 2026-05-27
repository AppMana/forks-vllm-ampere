# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.attention as deepseek_v4_attention
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_indexed_sparse_mla_attention_chunk,
    accumulate_indexed_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
)

_FP8_DIM = 448
_ROPE_DIM = 64
_SCALE_DIM = 8
_TOKEN_DATA_SIZE = _FP8_DIM + _ROPE_DIM * 2


def _write_fp8ds_mla_token(
    k_cache: torch.Tensor,
    slot: int,
    block_size: int,
) -> None:
    block_idx = slot // block_size
    block_offset = slot % block_size
    values = (
        (torch.arange(_FP8_DIM, device=k_cache.device, dtype=torch.float32) % 17) - 8
    ) / 16.0
    values = values + float(slot) / 32.0
    scale_exponents = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 1],
        device=k_cache.device,
        dtype=torch.float32,
    )
    scale_per_dim = torch.exp2(scale_exponents).repeat_interleave(64)
    fp8_values = (values / scale_per_dim).to(torch.float8_e4m3fn)
    rope = (
        torch.linspace(-1.0, 1.0, _ROPE_DIM, device=k_cache.device) + float(slot) / 16.0
    ).to(torch.bfloat16)

    flat_block = k_cache[block_idx].view(-1)
    token_data_start = block_offset * _TOKEN_DATA_SIZE
    token_scale_start = block_size * _TOKEN_DATA_SIZE + block_offset * _SCALE_DIM
    flat_block[token_data_start : token_data_start + _FP8_DIM] = fp8_values.view(
        torch.uint8
    )
    flat_block[token_data_start + _FP8_DIM : token_data_start + _TOKEN_DATA_SIZE] = (
        rope.view(torch.uint8)
    )
    encoded_scales = (scale_exponents.to(torch.int32) + 127).to(torch.uint8)
    flat_block[token_scale_start : token_scale_start + encoded_scales.numel()] = (
        encoded_scales
    )
    flat_block[
        token_scale_start + encoded_scales.numel() : token_scale_start + _SCALE_DIM
    ] = 127


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_multihead_indexed_sparse_mla_accumulate_matches_scalar_path() -> None:
    torch.manual_seed(0)
    num_tokens = 5
    num_heads = 8
    head_dim = 512
    num_kv_tokens = 257
    num_candidates = 128
    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_flat = torch.randn(
        num_kv_tokens,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.randint(
        -1,
        num_kv_tokens,
        (num_tokens, num_candidates),
        device="cuda",
        dtype=torch.int32,
    )
    lens = torch.tensor([128, 79, 0, 13, 96], device="cuda", dtype=torch.int32)
    sink = torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.01
    scale = head_dim**-0.5

    def run_accumulate(use_multihead: bool) -> torch.Tensor:
        max_score = torch.full(
            (num_tokens, num_heads),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        denom = torch.zeros_like(max_score)
        acc = torch.zeros(
            num_tokens,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        if use_multihead:
            accumulate_indexed_sparse_mla_attention_chunk_multihead(
                q=q,
                kv_flat=kv_flat,
                indices=indices,
                lens=lens,
                candidate_offset=0,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
                head_block_size=4,
            )
        else:
            accumulate_indexed_sparse_mla_attention_chunk(
                q=q,
                kv_flat=kv_flat,
                indices=indices,
                lens=lens,
                candidate_offset=0,
                scale=scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
            )
        output = torch.empty_like(q)
        finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, output)
        return output.float()

    expected = run_accumulate(use_multihead=False)
    actual = run_accumulate(use_multihead=True)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fp8ds_paged_sparse_mla_with_sink_matches_chunked_accumulate() -> None:
    torch.manual_seed(37)
    block_size = 4
    num_heads = 64
    k_cache = torch.zeros(
        4,
        block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-0.5, 0.5, num_heads, device="cuda")
    scale = 0.0625

    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // block_size].item())
            slot = physical_block * block_size + pos % block_size
            _write_fp8ds_mla_token(k_cache, slot, block_size)

    max_score = torch.full((2, num_heads), float("-inf"), device="cuda")
    denom = torch.zeros((2, num_heads), device="cuda")
    acc = torch.zeros((2, num_heads, 512), device="cuda")
    for candidate_offset, num_candidates in ((0, 2), (2, 3)):
        accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            candidate_offset=candidate_offset,
            num_candidates=num_candidates,
            scale=scale,
            max_score=max_score,
            denom=denom,
            acc=acc,
            head_block_size=4,
        )
    expected = torch.empty(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    finish_sparse_mla_attention_with_sink(max_score, denom, acc, sink, expected)

    actual = torch.empty_like(expected)
    fp8ds_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        k_cache=k_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        block_size=block_size,
        candidate_offset=0,
        num_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=4,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fp8ds_global_paged_sparse_mla_with_sink_matches_chunked_accumulate() -> None:
    torch.manual_seed(41)
    num_heads = 64
    compressed_block_size = 4
    swa_block_size = 4
    compressed_cache = torch.zeros(
        4,
        compressed_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    swa_cache = torch.zeros(
        4,
        swa_block_size,
        _TOKEN_DATA_SIZE + _SCALE_DIM,
        dtype=torch.uint8,
        device="cuda",
    )
    slot_ids = torch.tensor(
        [[0, 3, -1, 8, 1], [7, -1, 4, 0, 8]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_lens = torch.tensor([4, 5], dtype=torch.int32, device="cuda")
    block_table = torch.tensor(
        [[1, 0, 2, 3], [2, 3, 1, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32, device="cuda")
    gather_lens = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(2, 1, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-1.0, 1.0, num_heads, device="cuda")
    scale = 0.0625

    for slot in (0, 1, 3, 4, 7, 8):
        _write_fp8ds_mla_token(compressed_cache, slot, compressed_block_size)
    for token_idx in range(seq_lens.shape[0]):
        start_pos = int(seq_lens[token_idx].item() - gather_lens[token_idx].item())
        for gather_idx in range(int(gather_lens[token_idx].item())):
            pos = start_pos + gather_idx
            physical_block = int(block_table[token_idx, pos // swa_block_size].item())
            slot = physical_block * swa_block_size + pos % swa_block_size
            _write_fp8ds_mla_token(swa_cache, slot, swa_block_size)

    comp_max = torch.full((2, num_heads), float("-inf"), device="cuda")
    comp_denom = torch.zeros((2, num_heads), device="cuda")
    comp_acc = torch.zeros((2, num_heads, 512), device="cuda")
    swa_max = torch.full((2, num_heads), float("-inf"), device="cuda")
    swa_denom = torch.zeros((2, num_heads), device="cuda")
    swa_acc = torch.zeros((2, num_heads, 512), device="cuda")
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q=q,
        k_cache=compressed_cache,
        slot_ids=slot_ids,
        lens=topk_lens,
        block_size=compressed_block_size,
        candidate_offset=0,
        scale=scale,
        max_score=comp_max,
        denom=comp_denom,
        acc=comp_acc,
        head_block_size=4,
    )
    for candidate_offset, num_candidates in ((0, 2), (2, 3)):
        accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=swa_block_size,
            candidate_offset=candidate_offset,
            num_candidates=num_candidates,
            scale=scale,
            max_score=swa_max,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=4,
        )
    expected = torch.empty(2, num_heads, 512, device="cuda", dtype=torch.bfloat16)
    finish_two_sparse_mla_attention_states_with_sink(
        comp_max,
        comp_denom,
        comp_acc,
        swa_max,
        swa_denom,
        swa_acc,
        sink,
        expected,
    )

    actual = torch.empty_like(expected)
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
        q=q,
        compressed_k_cache=compressed_cache,
        slot_ids=slot_ids,
        topk_lens=topk_lens,
        compressed_block_size=compressed_block_size,
        swa_k_cache=swa_cache,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        block_table=block_table,
        swa_block_size=swa_block_size,
        num_compressed_candidates=5,
        num_swa_candidates=5,
        scale=scale,
        attn_sink=sink,
        output=actual,
        head_block_size=4,
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_deepseek_v4_sparse_prefill_matmul_path_matches_default(
    monkeypatch,
) -> None:
    torch.manual_seed(1)
    num_tokens = 7
    num_heads = 8
    head_dim = 512
    num_kv_tokens = 333
    num_candidates = 65
    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        1,
        num_kv_tokens,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    combined_indices = torch.randint(
        -1,
        num_kv_tokens,
        (num_tokens, num_candidates),
        device="cuda",
        dtype=torch.int32,
    )
    combined_lens = torch.tensor(
        [65, 11, 0, 64, 23, 49, 4],
        device="cuda",
        dtype=torch.int32,
    )
    attn = SimpleNamespace(
        prefix="test",
        num_heads=num_heads,
        scale=head_dim**-0.5,
        attn_sink=torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.01,
    )

    def run_prefill(use_matmul: bool) -> torch.Tensor:
        monkeypatch.setenv(
            "VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL",
            "1" if use_matmul else "0",
        )
        monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE", "17")
        monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE", "3")
        max_score = torch.empty(3, num_heads, device="cuda", dtype=torch.float32)
        denom = torch.empty_like(max_score)
        acc = torch.empty(3, num_heads, head_dim, device="cuda", dtype=torch.float32)
        output = torch.empty_like(q)
        deepseek_v4_attention.DeepseekV4MLAAttention._forward_sparse_mla_prefill_triton(
            attn,
            q=q,
            kv=kv,
            combined_indices=combined_indices,
            combined_lens=combined_lens,
            output=output,
            state_buffers=(max_score, denom, acc),
        )
        torch.cuda.synchronize()
        return output.float()

    expected = run_prefill(use_matmul=False)
    actual = run_prefill(use_matmul=True)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
