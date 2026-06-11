# SPDX-License-Identifier: Apache-2.0
"""Decode-shape bench: Triton matmul sparse-MLA decode vs sm_86 flash-MLA.

Compares the live sm_86 decode tail (`matmul_sparse_mla_attention_with_sink`
over the materialized [T, C, 576] bf16 workspace) against the
forks-flash-mla-ampere-dsv4 dense kernel fed the same workspace as one page
per token, plus the lse-based sink rescale flash-MLA would need.

    .venv/bin/python tools/ampere/bench_sparse_mla_decode_flashmla.py \
        --flash-mla-repo ~/Documents/forks-flash-mla-ampere-dsv4
"""

import argparse
import os
import sys
import time

import torch

H, D, DV = 128, 576, 512


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-mla-repo", required=True)
    parser.add_argument("--tokens", default="1,4,12")
    parser.add_argument("--candidates", default="512,1088")
    args = parser.parse_args()

    sys.path.insert(0, os.path.expanduser(args.flash_mla_repo))
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    import vllm.models.deepseek_v4.attention  # noqa: F401  (break circular import)
    from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
        matmul_sparse_mla_attention_with_sink,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    scale = D ** -0.5

    print(f"{'T':>3} {'C':>5} | {'triton matmul+sink us':>22} | "
          f"{'flash-mla (+sink merge) us':>26} | speedup")
    for T in [int(x) for x in args.tokens.split(",")]:
        for C in [int(x) for x in args.candidates.split(",")]:
            q = torch.randn(T, H, D, dtype=torch.bfloat16)
            kv = torch.randn(T, C, D, dtype=torch.bfloat16)
            valid = torch.ones(T, C, dtype=torch.bool)
            sink = torch.randn(H, dtype=torch.float32)
            out_a = torch.empty(T, H, D, dtype=torch.bfloat16)
            score = torch.empty(T, H, C, dtype=torch.bfloat16)

            def path_a():
                matmul_sparse_mla_attention_with_sink(
                    q=q, kv=kv, valid_tokens=valid, scale=scale,
                    attn_sink=sink, output=out_a, num_heads=H,
                    score_buffer=score, value_block_size=512,
                    candidate_block_size=128,
                )

            # flash-MLA: one fused page run per token batch; pages of 64.
            block_size = 32
            assert C % block_size == 0
            blocks_per_tok = C // block_size
            blocked_k = kv.reshape(T * blocks_per_tok, block_size, 1, D)
            block_table = torch.arange(
                T * blocks_per_tok, dtype=torch.int32
            ).view(T, blocks_per_tok)
            cache_seqlens = torch.full((T,), C, dtype=torch.int32)
            q_b = q.view(T, 1, H, D)
            meta, splits = get_mla_metadata(cache_seqlens, H, 1)

            def path_b():
                out, lse = flash_mla_with_kvcache(
                    q_b, blocked_k, block_table, cache_seqlens, DV,
                    meta, splits, causal=False,
                )
                # sink merge: rescale by softmax mass excluding the sink
                # logit; lse is [T, H] base-e.
                lse2 = lse.float().reshape(T, H)  # lse is [b, h_q, s_q]
                w = 1.0 / (1.0 + torch.exp(sink[None, :] - lse2))
                return out[:, 0] * w.unsqueeze(-1).to(out.dtype)

            # correctness spot check vs path A (sink-aware reference)
            path_a()
            out_b = path_b()
            ref = out_a[:, :, :DV].float()
            got = out_b.float()
            cos = torch.nn.functional.cosine_similarity(
                ref.flatten(), got.flatten(), dim=0
            ).item()

            ta = bench(path_a)
            tb = bench(path_b)
            print(f"{T:>3} {C:>5} | {ta:>22.1f} | {tb:>26.1f} | "
                  f"{ta / tb:>5.2f}x  (cos={cos:.5f})")


if __name__ == "__main__":
    main()
