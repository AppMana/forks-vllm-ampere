# SPDX-License-Identifier: Apache-2.0
"""Validate + bench the INT8 IMMA indexer logits path (QK_INT8) on sm_86.

Builds a layout-correct int8 indexer cache (block-major payload + fp32
absmax/127 scales), runs `fp8_paged_mqa_logits_rowwise_triton` with
q_is_int8=True (q symmetric INT8, its scale folded into weights) against
the fp32-reference logits computed from the same dequantized tensors, and
reports top-512 recall, logits SNR, and kernel time vs the existing
tf32-dot path fed the same int8 cache.

    APPMANA_DSV4_INDEXER_CACHE_INT8=1 .venv/bin/python \
        tools/ampere/bench_dsv4_indexer_imma.py
"""

import os
import time

os.environ.setdefault("APPMANA_DSV4_INDEXER_CACHE_INT8", "1")

import torch

H, D, BLOCK_SIZE = 64, 128, 64
TOPK = 512


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    import vllm.models.deepseek_v4.attention  # noqa: F401
    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        fp8_paged_mqa_logits_rowwise_triton,
    )

    torch.set_default_device("cuda")
    torch.manual_seed(11)

    for B, ctx in [(1, 4096), (12, 4096), (12, 1024)]:
        num_blocks = B * (ctx // BLOCK_SIZE)
        S = ctx

        # real-valued K, quantized to per-token symmetric int8
        k_true = torch.randn(B, S, D) * 0.3
        k_scale = k_true.abs().amax(dim=2).clamp(min=1e-4) / 127.0  # [B, S]
        k_i8 = torch.round(k_true / k_scale[..., None]).clamp(-127, 127).to(torch.int8)

        # pack: [num_blocks, block_size*D + block_size*4] block-major
        cache = torch.zeros(num_blocks, BLOCK_SIZE * (D + 4), dtype=torch.uint8)
        payload = k_i8.view(torch.uint8).reshape(B, ctx // BLOCK_SIZE, BLOCK_SIZE, D)
        scales = k_scale.reshape(B, ctx // BLOCK_SIZE, BLOCK_SIZE)
        cache_v = cache.view(num_blocks, -1)
        cache_v[:, : BLOCK_SIZE * D] = payload.reshape(num_blocks, -1)
        cache_v[:, BLOCK_SIZE * D :] = (
            scales.reshape(num_blocks, BLOCK_SIZE).view(torch.uint8).reshape(num_blocks, -1)
            if False else
            scales.reshape(num_blocks, BLOCK_SIZE, 1).contiguous().view(torch.float32).view(torch.uint8).reshape(num_blocks, -1)
        )
        kv_cache = cache.view(num_blocks, BLOCK_SIZE, D + 4)

        block_tables = torch.arange(num_blocks, dtype=torch.int32).view(B, -1)
        context_lens = torch.full((B,), S, dtype=torch.int32)

        # q: real values; fp8 baseline bytes and int8 variant
        q_true = torch.randn(B, 1, H, D) * 0.3
        q_fp8 = q_true.to(torch.float8_e4m3fn)
        q_fp8_bytes = q_fp8.view(torch.uint8)
        q_s = q_true.abs().amax(dim=3).clamp(min=1e-30) / 127.0  # [B, 1, H]
        q_i8 = torch.round(q_true / q_s[..., None]).clamp(-127, 127).to(torch.int8)
        weights = torch.rand(B, H, dtype=torch.float32) * 0.1
        weights_imma = weights * q_s[:, 0, :]

        # fp32 reference logits from the dequantized tensors
        k_deq = k_i8.float() * k_scale[..., None]
        ref = torch.einsum("bhd,bsd->bhs", q_fp8.float()[:, 0], k_deq)
        ref = (torch.relu(ref) * weights[:, :, None]).sum(dim=1)  # [B, S]

        out_tf32 = fp8_paged_mqa_logits_rowwise_triton(
            q_fp8_bytes, kv_cache, weights, context_lens, block_tables, S
        )[:, :S]
        out_imma = fp8_paged_mqa_logits_rowwise_triton(
            q_i8.view(torch.uint8), kv_cache, weights_imma, context_lens,
            block_tables, S, q_is_int8=True,
        )[:, :S]

        def snr(a, b):
            return (10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum())).item()

        recalls = []
        for b in range(B):
            ra = torch.topk(ref[b], TOPK).indices
            rb = torch.topk(out_imma[b].float(), TOPK).indices
            recalls.append(len(set(ra.tolist()) & set(rb.tolist())) / TOPK)

        t_tf32 = bench(lambda: fp8_paged_mqa_logits_rowwise_triton(
            q_fp8_bytes, kv_cache, weights, context_lens, block_tables, S))
        t_imma = bench(lambda: fp8_paged_mqa_logits_rowwise_triton(
            q_i8.view(torch.uint8), kv_cache, weights_imma, context_lens,
            block_tables, S, q_is_int8=True))

        print(
            f"B={B} ctx={ctx}: snr_tf32={snr(ref, out_tf32.float()):.1f}dB "
            f"snr_imma={snr(ref, out_imma.float()):.1f}dB "
            f"recall@512={min(recalls):.4f} | tf32={t_tf32:.0f}us "
            f"imma={t_imma:.0f}us speedup={t_tf32 / t_imma:.2f}x"
        )


if __name__ == "__main__":
    main()
