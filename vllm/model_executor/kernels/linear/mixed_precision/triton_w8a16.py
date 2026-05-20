# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W8A16 kernels for channelwise DeepSeek V4 INT8 linears."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _triton_channel_w8a16_kernel(
    a_ptr,
    w_ptr,
    scales_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=0.0)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w_u8 = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=128,
        )
        w = (w_u8.to(tl.float32) - 128.0) * scales[None, :]
        acc += tl.dot(a, w.to(a.dtype), out_dtype=tl.float32)

    c = acc.to(c_ptr.type.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_channel_w8a16_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Compute ``a @ dequant(weight).T`` for biased-uint8 channel W8A16.

    ``weight`` is stored as ``[N, K]`` uint8 with signed int8 values biased by
    +128. ``scales`` is one fp16/bf16 scale per output channel.
    """
    assert a.is_cuda
    assert a.is_contiguous()
    assert weight.is_contiguous()
    assert scales.is_contiguous()
    assert weight.dtype == torch.uint8
    assert a.dtype in (torch.float16, torch.bfloat16)

    M, K = a.shape
    N = weight.shape[0]
    assert weight.shape == (N, K)
    assert scales.shape == (N,)

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M <= 16:
        block_m = 16
        block_n = 64
        block_k = 64
    elif M <= 64:
        block_m = 32
        block_n = 64
        block_k = 64
    else:
        block_m = 64
        block_n = 64
        block_k = 64

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _triton_channel_w8a16_kernel[grid](
        a,
        weight,
        scales,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c
