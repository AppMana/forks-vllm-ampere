#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the DeepSeek V4 sparse MLA prefill Triton path.

This compares the default indexed sparse MLA prefill accumulate path against the
gated matmul-style path selected by VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL=1.
It calls DeepseekV4MLAAttention._forward_sparse_mla_prefill_triton directly, so
the query chunk loop, top-k chunk loop, candidate offsets, state buffers, and
finish kernel are included in the timing.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace

import torch

import vllm.models.deepseek_v4.attention as deepseek_v4_attention


@dataclass(frozen=True)
class Case:
    tokens: int
    heads: int
    head_dim: int
    kv_tokens: int
    candidates: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def _make_inputs(case: Case, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = torch.randn(
        case.tokens,
        case.heads,
        case.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        1,
        case.kv_tokens,
        case.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    combined_indices = torch.randint(
        -1,
        case.kv_tokens,
        (case.tokens, case.candidates),
        device="cuda",
        dtype=torch.int32,
    )
    min_lens = max(1, case.candidates // 2)
    combined_lens = torch.randint(
        min_lens,
        case.candidates + 1,
        (case.tokens,),
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty_like(q)
    attn_sink = torch.randn(case.heads, device="cuda", dtype=torch.float32) * 0.01
    return q, kv, combined_indices, combined_lens, output, attn_sink


def _run_prefill(
    case: Case,
    inputs: tuple[torch.Tensor, ...],
    *,
    use_matmul: bool,
    query_chunk_size: int,
    topk_chunk_size: int,
) -> torch.Tensor:
    os.environ["VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL"] = "1" if use_matmul else "0"
    os.environ["VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE"] = str(query_chunk_size)
    os.environ["VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE"] = str(topk_chunk_size)
    q, kv, combined_indices, combined_lens, output, attn_sink = inputs
    attn = SimpleNamespace(
        prefix="bench",
        num_heads=case.heads,
        scale=case.head_dim**-0.5,
        attn_sink=attn_sink,
    )
    max_query_chunk = min(case.tokens, query_chunk_size)
    max_score = torch.empty(
        max_query_chunk,
        case.heads,
        device="cuda",
        dtype=torch.float32,
    )
    denom = torch.empty_like(max_score)
    acc = torch.empty(
        max_query_chunk,
        case.heads,
        case.head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    deepseek_v4_attention.DeepseekV4MLAAttention._forward_sparse_mla_prefill_triton(
        attn,
        q=q,
        kv=kv,
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        output=output,
        state_buffers=(max_score, denom, acc),
    )
    return output


def _time_ms(
    case: Case,
    inputs: tuple[torch.Tensor, ...],
    *,
    use_matmul: bool,
    query_chunk_size: int,
    topk_chunk_size: int,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _run_prefill(
            case,
            inputs,
            use_matmul=use_matmul,
            query_chunk_size=query_chunk_size,
            topk_chunk_size=topk_chunk_size,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _run_prefill(
            case,
            inputs,
            use_matmul=use_matmul,
            query_chunk_size=query_chunk_size,
            topk_chunk_size=topk_chunk_size,
        )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _cases(args: argparse.Namespace) -> Iterable[Case]:
    for tokens in _parse_int_list(args.tokens):
        for candidates in _parse_int_list(args.candidates):
            yield Case(
                tokens=tokens,
                heads=args.heads,
                head_dim=args.head_dim,
                kv_tokens=args.kv_tokens,
                candidates=candidates,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="64,128,256")
    parser.add_argument("--candidates", default="512")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--kv-tokens", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--topk-chunk-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        "tokens,heads,head_dim,kv_tokens,candidates,query_chunk,topk_chunk,"
        "default_ms,matmul_ms,speedup,max_abs,mean_abs"
    )

    for case in _cases(args):
        inputs = _make_inputs(case, args.seed)
        max_abs = float("nan")
        mean_abs = float("nan")
        if not args.skip_correctness:
            expected = _run_prefill(
                case,
                inputs,
                use_matmul=False,
                query_chunk_size=args.query_chunk_size,
                topk_chunk_size=args.topk_chunk_size,
            ).float().clone()
            actual = _run_prefill(
                case,
                inputs,
                use_matmul=True,
                query_chunk_size=args.query_chunk_size,
                topk_chunk_size=args.topk_chunk_size,
            ).float().clone()
            diff = (actual - expected).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        default_ms = _time_ms(
            case,
            inputs,
            use_matmul=False,
            query_chunk_size=args.query_chunk_size,
            topk_chunk_size=args.topk_chunk_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        matmul_ms = _time_ms(
            case,
            inputs,
            use_matmul=True,
            query_chunk_size=args.query_chunk_size,
            topk_chunk_size=args.topk_chunk_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(
            f"{case.tokens},{case.heads},{case.head_dim},{case.kv_tokens},"
            f"{case.candidates},{args.query_chunk_size},{args.topk_chunk_size},"
            f"{default_ms:.3f},{matmul_ms:.3f},{default_ms / matmul_ms:.3f},"
            f"{max_abs:.6g},{mean_abs:.6g}"
        )


if __name__ == "__main__":
    main()
