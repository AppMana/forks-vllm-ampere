# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import time
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _mhc_debug_timings_enabled() -> bool:
    return os.getenv("VLLM_MHC_DEBUG_TIMINGS", "0") == "1"


def _mhc_torch_fallback_synchronize() -> bool:
    return os.getenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "1") != "0"


def _synchronize_mhc_torch_fallback() -> None:
    if not _mhc_torch_fallback_synchronize():
        return
    mode = os.getenv("VLLM_MHC_TORCH_FALLBACK_SYNC_MODE", "stream").lower()
    if mode == "none":
        return
    if mode == "device":
        torch.cuda.synchronize()
        return
    if mode != "stream":
        logger.warning_once(
            "Unknown VLLM_MHC_TORCH_FALLBACK_SYNC_MODE=%r; using stream sync.",
            mode,
        )
    torch.cuda.current_stream().synchronize()


def _mhc_torch_fallback_chunk_tokens() -> int:
    value = os.getenv("VLLM_MHC_TORCH_FALLBACK_CHUNK_TOKENS")
    if value is not None:
        return int(value or "0")
    if current_platform.is_cuda():
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 8:
            return 64
    return 0


def _mhc_post_triton_enabled() -> bool:
    return os.getenv("VLLM_MHC_POST_TRITON", "1") != "0"


def _mhc_head_triton_enabled() -> bool:
    return os.getenv("VLLM_MHC_HEAD_TRITON", "1") != "0"


def _mhc_pre_triton_enabled() -> bool:
    return os.getenv("VLLM_MHC_PRE_TRITON", "1") != "0"


def _should_use_mhc_torch_fallback() -> bool:
    """Hyperconnections (mhc_pre/post/hc_head) use TileLang JIT on CUDA, but
    TileLang requires sm_89+. On Ampere/Ada (sm_8x) and ROCm we fall back to
    a numerically-equivalent pure-torch implementation. The TileLang path
    is roughly 1.5-2x faster on supported hardware; on Ampere we eat the
    overhead in exchange for portability.
    """
    if current_platform.is_rocm():
        return True
    if current_platform.is_cuda():
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 8:
            return True
    return False


@triton.jit
def _mhc_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _mhc_pre_postprocess_triton_kernel(
    mixes_ptr,
    sqrsum_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    num_tokens: tl.constexpr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    mixes_stride_t: tl.constexpr,
    mixes_stride_m: tl.constexpr,
    sqrsum_stride_t: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_m: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
):
    token = tl.program_id(0)
    hc_dim = hc * hidden
    rms = tl.rsqrt(tl.load(sqrsum_ptr + token * sqrsum_stride_t) / hc_dim + rms_eps)
    scale0 = tl.load(scale_ptr + 0).to(tl.float32)
    scale1 = tl.load(scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(scale_ptr + 2).to(tl.float32)

    for i in tl.static_range(0, 4):
        pre_mix = tl.load(mixes_ptr + token * mixes_stride_t + i * mixes_stride_m)
        pre_base = tl.load(base_ptr + i)
        pre = _mhc_sigmoid(pre_mix * rms * scale0 + pre_base) + hc_pre_eps
        tl.store(pre_ptr + token * pre_stride_t + i * pre_stride_m, pre)

        post_mix = tl.load(
            mixes_ptr + token * mixes_stride_t + (4 + i) * mixes_stride_m
        )
        post_base = tl.load(base_ptr + 4 + i)
        post = _mhc_sigmoid(post_mix * rms * scale1 + post_base) * hc_post_mult_value
        tl.store(post_ptr + token * post_stride_t + i * post_stride_m, post)

    # TileLang reference:
    #   cm = softmax(row_logits) + eps
    #   cm = cm / (cm.sum(dim=-2) + eps)
    for row in tl.static_range(0, 4):
        row_max = tl.full((), -float("inf"), dtype=tl.float32)
        for col in tl.static_range(0, 4):
            mix_id = 8 + row * 4 + col
            mix = tl.load(mixes_ptr + token * mixes_stride_t + mix_id * mixes_stride_m)
            base = tl.load(base_ptr + mix_id)
            logit = mix * rms * scale2 + base
            row_max = tl.maximum(row_max, logit)

        row_sum = tl.full((), 0.0, dtype=tl.float32)
        for col in tl.static_range(0, 4):
            mix_id = 8 + row * 4 + col
            mix = tl.load(mixes_ptr + token * mixes_stride_t + mix_id * mixes_stride_m)
            base = tl.load(base_ptr + mix_id)
            logit = mix * rms * scale2 + base
            row_sum += tl.exp(logit - row_max)

        for col in tl.static_range(0, 4):
            mix_id = 8 + row * 4 + col
            mix = tl.load(mixes_ptr + token * mixes_stride_t + mix_id * mixes_stride_m)
            base = tl.load(base_ptr + mix_id)
            logit = mix * rms * scale2 + base
            cm = tl.exp(logit - row_max) / row_sum + hc_sinkhorn_eps

            col_sum = tl.full((), 0.0, dtype=tl.float32)
            for other_row in tl.static_range(0, 4):
                other_row_max = tl.full((), -float("inf"), dtype=tl.float32)
                for other_col in tl.static_range(0, 4):
                    other_id = 8 + other_row * 4 + other_col
                    other_mix = tl.load(
                        mixes_ptr + token * mixes_stride_t + other_id * mixes_stride_m
                    )
                    other_base = tl.load(base_ptr + other_id)
                    other_logit = other_mix * rms * scale2 + other_base
                    other_row_max = tl.maximum(other_row_max, other_logit)

                other_row_sum = tl.full((), 0.0, dtype=tl.float32)
                for other_col in tl.static_range(0, 4):
                    other_id = 8 + other_row * 4 + other_col
                    other_mix = tl.load(
                        mixes_ptr + token * mixes_stride_t + other_id * mixes_stride_m
                    )
                    other_base = tl.load(base_ptr + other_id)
                    other_logit = other_mix * rms * scale2 + other_base
                    other_row_sum += tl.exp(other_logit - other_row_max)

                col_id = 8 + other_row * 4 + col
                col_mix = tl.load(
                    mixes_ptr + token * mixes_stride_t + col_id * mixes_stride_m
                )
                col_base = tl.load(base_ptr + col_id)
                col_logit = col_mix * rms * scale2 + col_base
                col_sum += tl.exp(col_logit - other_row_max) / other_row_sum
                col_sum += hc_sinkhorn_eps

            out = cm / (col_sum + hc_sinkhorn_eps)
            tl.store(
                comb_ptr
                + token * comb_stride_t
                + row * comb_stride_i
                + col * comb_stride_j,
                out,
            )


@triton.jit
def _mhc_post_triton_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_h: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_j: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_j: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    j = tl.program_id(1)
    hidden_block = tl.program_id(2)
    offs_h = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (token < num_tokens) & (j < hc) & (offs_h < hidden)

    x = tl.load(
        x_ptr + token * x_stride_t + offs_h * x_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    post = tl.load(
        post_ptr + token * post_stride_t + j * post_stride_j,
        mask=(token < num_tokens) & (j < hc),
        other=0.0,
    ).to(tl.float32)
    acc = post * x

    for i in range(0, hc):
        comb = tl.load(
            comb_ptr + token * comb_stride_t + i * comb_stride_i + j * comb_stride_j,
            mask=(token < num_tokens) & (j < hc),
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + token * residual_stride_t
            + i * residual_stride_i
            + offs_h * residual_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += comb * residual

    tl.store(
        out_ptr + token * out_stride_t + j * out_stride_j + offs_h * out_stride_h,
        acc.to(tl.bfloat16),
        mask=mask,
    )


def _mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor,
) -> None:
    num_tokens = residual.numel() // (residual.shape[-2] * residual.shape[-1])
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    x_flat = x.reshape(num_tokens, hidden)
    residual_flat = residual.reshape(num_tokens, hc, hidden)
    post_flat = post_layer_mix.reshape(num_tokens, hc, 1)
    comb_flat = comb_res_mix.reshape(num_tokens, hc, hc)
    out_flat = out.reshape(num_tokens, hc, hidden)

    block_h = 1024
    grid = (num_tokens, hc, triton.cdiv(hidden, block_h))
    _mhc_post_triton_kernel[grid](
        x_flat,
        residual_flat,
        post_flat,
        comb_flat,
        out_flat,
        num_tokens,
        hc,
        hidden,
        x_flat.stride(0),
        x_flat.stride(1),
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        post_flat.stride(0),
        post_flat.stride(1),
        comb_flat.stride(0),
        comb_flat.stride(1),
        comb_flat.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
        BLOCK_H=block_h,
        num_warps=8,
    )


@triton.jit
def _hc_head_pre_mix_triton_kernel(
    residual_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    num_tokens: tl.constexpr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    fn_stride_m: tl.constexpr,
    fn_stride_k: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    mix_id = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)
    hc_dim = hc * hidden
    channel = offs // hidden
    h = offs - channel * hidden
    mask = offs < hc_dim

    x = tl.load(
        residual_ptr
        + token * residual_stride_t
        + channel * residual_stride_i
        + h * residual_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        fn_ptr + mix_id * fn_stride_m + offs * fn_stride_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sqrsum = tl.sum(x * x, axis=0)
    mix = tl.sum(x * w, axis=0)
    rsqrt = tl.rsqrt(sqrsum / hc_dim + rms_eps)
    scale = tl.load(scale_ptr).to(tl.float32)
    base = tl.load(base_ptr + mix_id).to(tl.float32)
    pre = _mhc_sigmoid(mix * rsqrt * scale + base) + hc_eps
    tl.store(pre_ptr + token * pre_stride_t + mix_id * pre_stride_m, pre)


@triton.jit
def _hc_head_apply_triton_kernel(
    residual_ptr,
    pre_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    offs_h = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < hidden
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for i in range(0, hc):
        pre = tl.load(pre_ptr + token * pre_stride_t + i * pre_stride_m).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + token * residual_stride_t
            + i * residual_stride_i
            + offs_h * residual_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * residual
    tl.store(
        out_ptr + token * out_stride_t + offs_h * out_stride_h,
        acc.to(tl.bfloat16),
        mask=mask,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return
    residual_flat = hs_flat.reshape(num_tokens, hc_mult, hidden_size)
    pre = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=hs_flat.device
    )
    block_k = triton.next_power_of_2(hc_mult * hidden_size)
    _hc_head_pre_mix_triton_kernel[(num_tokens, hc_mult)](
        residual_flat,
        fn,
        hc_scale,
        hc_base,
        pre,
        num_tokens,
        hc_mult,
        hidden_size,
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        fn.stride(0),
        fn.stride(1),
        pre.stride(0),
        pre.stride(1),
        rms_eps,
        hc_eps,
        BLOCK_K=block_k,
        num_warps=8,
    )
    block_h = 1024
    _hc_head_apply_triton_kernel[(num_tokens, triton.cdiv(hidden_size, block_h))](
        residual_flat,
        pre,
        out,
        num_tokens,
        hc_mult,
        hidden_size,
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        pre.stride(0),
        pre.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=8,
    )


@triton.jit
def _mhc_pre_mix_triton_kernel(
    residual_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    post_ptr,
    comb_ptr,
    pre_ptr,
    num_tokens: tl.constexpr,
    hc: tl.constexpr,
    hidden: tl.constexpr,
    residual_stride_t: tl.constexpr,
    residual_stride_i: tl.constexpr,
    residual_stride_h: tl.constexpr,
    fn_stride_m: tl.constexpr,
    fn_stride_k: tl.constexpr,
    post_stride_t: tl.constexpr,
    post_stride_m: tl.constexpr,
    comb_stride_t: tl.constexpr,
    comb_stride_i: tl.constexpr,
    comb_stride_j: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    mix_id = tl.program_id(1)
    hc2 = hc * hc
    hc3 = hc * 2 + hc2
    offs = tl.arange(0, BLOCK_K)
    hc_dim = hc * hidden
    channel = offs // hidden
    h = offs - channel * hidden
    mask = offs < hc_dim

    x = tl.load(
        residual_ptr
        + token * residual_stride_t
        + channel * residual_stride_i
        + h * residual_stride_h,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        fn_ptr + mix_id * fn_stride_m + offs * fn_stride_k,
        mask=(mix_id < hc3) & mask,
        other=0.0,
    ).to(tl.float32)
    sqrsum = tl.sum(x * x, axis=0)
    mix = tl.sum(x * w, axis=0)
    rsqrt = tl.rsqrt(sqrsum / hc_dim + rms_eps)
    mix = mix * rsqrt

    scale0 = tl.load(scale_ptr + 0).to(tl.float32)
    scale1 = tl.load(scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(scale_ptr + 2).to(tl.float32)
    base = tl.load(base_ptr + mix_id, mask=mix_id < hc3, other=0.0).to(tl.float32)

    if mix_id < hc:
        value = _mhc_sigmoid(mix * scale0 + base) + hc_pre_eps
        tl.store(pre_ptr + token * pre_stride_t + mix_id * pre_stride_m, value)
    elif mix_id < 2 * hc:
        post_id = mix_id - hc
        value = _mhc_sigmoid(mix * scale1 + base) * hc_post_mult_value
        tl.store(post_ptr + token * post_stride_t + post_id * post_stride_m, value)
    else:
        comb_id = mix_id - 2 * hc
        row = comb_id // hc
        col = comb_id - row * hc
        logits = mix * scale2 + base

        # Compute row softmax for this token/row by recomputing the small
        # hc-wide logits. hc is 4 for DeepSeek V4, so this keeps the kernel
        # simple and avoids a separate global-memory logits tensor.
        row_max = tl.full((), -float("inf"), dtype=tl.float32)
        for c in range(0, hc):
            row_mix_id = 2 * hc
            other_id = row_mix_id + row * hc + c
            w_other = tl.load(
                fn_ptr + other_id * fn_stride_m + offs * fn_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            other_mix = tl.sum(x * w_other, axis=0) * rsqrt
            other_base = tl.load(base_ptr + other_id).to(tl.float32)
            other_logit = other_mix * scale2 + other_base
            row_max = tl.maximum(row_max, other_logit)

        row_sum = tl.full((), 0.0, dtype=tl.float32)
        for c in range(0, hc):
            other_id = 2 * hc + row * hc + c
            w_other = tl.load(
                fn_ptr + other_id * fn_stride_m + offs * fn_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            other_mix = tl.sum(x * w_other, axis=0) * rsqrt
            other_base = tl.load(base_ptr + other_id).to(tl.float32)
            other_logit = other_mix * scale2 + other_base
            row_sum += tl.exp(other_logit - row_max)

        soft = tl.exp(logits - row_max) / row_sum + hc_sinkhorn_eps

        col_sum = tl.full((), 0.0, dtype=tl.float32)
        for r in range(0, hc):
            row_max_r = tl.full((), -float("inf"), dtype=tl.float32)
            for c in range(0, hc):
                other_id = 2 * hc + r * hc + c
                w_other = tl.load(
                    fn_ptr + other_id * fn_stride_m + offs * fn_stride_k,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                other_mix = tl.sum(x * w_other, axis=0) * rsqrt
                other_base = tl.load(base_ptr + other_id).to(tl.float32)
                other_logit = other_mix * scale2 + other_base
                row_max_r = tl.maximum(row_max_r, other_logit)
            row_sum_r = tl.full((), 0.0, dtype=tl.float32)
            for c in range(0, hc):
                other_id = 2 * hc + r * hc + c
                w_other = tl.load(
                    fn_ptr + other_id * fn_stride_m + offs * fn_stride_k,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                other_mix = tl.sum(x * w_other, axis=0) * rsqrt
                other_base = tl.load(base_ptr + other_id).to(tl.float32)
                other_logit = other_mix * scale2 + other_base
                row_sum_r += tl.exp(other_logit - row_max_r)
            this_id = 2 * hc + r * hc + col
            w_this = tl.load(
                fn_ptr + this_id * fn_stride_m + offs * fn_stride_k,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            this_mix = tl.sum(x * w_this, axis=0) * rsqrt
            this_base = tl.load(base_ptr + this_id).to(tl.float32)
            this_logit = this_mix * scale2 + this_base
            col_sum += tl.exp(this_logit - row_max_r) / row_sum_r + hc_sinkhorn_eps

        value = soft / (col_sum + hc_sinkhorn_eps)
        # Only sinkhorn_repeat=1 is used by current DSv4 configs we have been
        # testing. Fall back to Torch if more iterations are requested.
        tl.store(
            comb_ptr + token * comb_stride_t + row * comb_stride_i + col * comb_stride_j,
            value,
        )


def _mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sinkhorn_repeat != 1:
        raise NotImplementedError("Triton mhc_pre currently supports sinkhorn_repeat=1")
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = residual.numel() // (hc * hidden)
    residual_flat = residual.reshape(num_tokens, hc, hidden)
    x = residual_flat.reshape(num_tokens, hc * hidden).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1)
    pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=residual.device)
    post = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=residual.device)
    comb = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(
        num_tokens, hidden, dtype=torch.bfloat16, device=residual.device
    )
    _mhc_pre_postprocess_triton_kernel[(num_tokens,)](
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        num_tokens,
        hc,
        hidden,
        mixes.stride(0),
        mixes.stride(1),
        sqrsum.stride(0),
        pre.stride(0),
        pre.stride(1),
        post.stride(0),
        post.stride(1),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        num_warps=1,
    )
    block_h = 1024
    _hc_head_apply_triton_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        residual_flat,
        pre,
        layer_input,
        num_tokens,
        hc,
        hidden,
        residual_flat.stride(0),
        residual_flat.stride(1),
        residual_flat.stride(2),
        pre.stride(0),
        pre.stride(1),
        layer_input.stride(0),
        layer_input.stride(1),
        BLOCK_H=block_h,
        num_warps=8,
    )
    return (
        post.view(*outer_shape, hc, 1),
        comb.view(*outer_shape, hc, hc),
        layer_input.view(*outer_shape, hidden),
    )

# tilelang is only available on CUDA platforms
if TYPE_CHECKING or current_platform.is_cuda_alike():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])
        T.pdl_trigger()


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    if _should_use_mhc_torch_fallback():
        if (
            residual.is_cuda
            and current_platform.is_cuda()
            and _mhc_pre_triton_enabled()
            and sinkhorn_repeat == 1
        ):
            out = _mhc_pre_triton(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
            return out
        debug_timings = _mhc_debug_timings_enabled()
        started = time.perf_counter()
        chunk_tokens = _mhc_torch_fallback_chunk_tokens()
        if debug_timings:
            logger.warning(
                "MHC_DEBUG_START mhc_pre_torch_fallback tokens=%d hc_mult=%d "
                "hidden=%d chunk_tokens=%d residual_shape=%s fn_shape=%s",
                num_tokens,
                hc_mult,
                hidden_size,
                chunk_tokens,
                tuple(residual.shape),
                tuple(fn.shape),
            )

        def compute_chunk(
            residual_chunk: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            chunk_size = residual_chunk.shape[0]
            x = residual_chunk.view(chunk_size, hc_mult * hidden_size).to(torch.float32)
            mixes = torch.matmul(x, fn_flat.t())
            sqrsum = x.square().sum(dim=-1, keepdim=True)
            mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

            pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
            pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

            post_logits = (
                mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
                + hc_base[hc_mult : 2 * hc_mult]
            )
            post_chunk = torch.sigmoid(post_logits) * hc_post_mult_value

            comb_logits = mixes[:, 2 * hc_mult :].view(
                chunk_size, hc_mult, hc_mult
            ) * hc_scale[2] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
            comb_chunk = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
            comb_chunk = comb_chunk / (
                comb_chunk.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
            )
            for _ in range(sinkhorn_repeat - 1):
                comb_chunk = comb_chunk / (
                    comb_chunk.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps
                )
                comb_chunk = comb_chunk / (
                    comb_chunk.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps
                )

            layer_chunk = torch.sum(
                pre_mix.unsqueeze(-1) * residual_chunk.to(torch.float32), dim=1
            ).to(torch.bfloat16)
            return post_chunk, comb_chunk, layer_chunk

        if chunk_tokens > 0 and num_tokens > chunk_tokens:
            post_chunks: list[torch.Tensor] = []
            comb_chunks: list[torch.Tensor] = []
            layer_chunks: list[torch.Tensor] = []
            for start in range(0, num_tokens, chunk_tokens):
                post_chunk, comb_chunk, layer_chunk = compute_chunk(
                    residual_flat[start : start + chunk_tokens]
                )
                post_chunks.append(post_chunk)
                comb_chunks.append(comb_chunk)
                layer_chunks.append(layer_chunk)
            post_mix = torch.cat(post_chunks, dim=0)
            comb_mix = torch.cat(comb_chunks, dim=0)
            layer_input = torch.cat(layer_chunks, dim=0)
        else:
            post_mix, comb_mix, layer_input = compute_chunk(residual_flat)

        if debug_timings:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            logger.warning(
                "MHC_DEBUG_END mhc_pre_torch_fallback %.6fs tokens=%d "
                "chunk_tokens=%d",
                time.perf_counter() - started,
                num_tokens,
                chunk_tokens,
            )
        else:
            _synchronize_mhc_torch_fallback()
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    # these number are from deepgemm kernel impl
    block_k = 64
    block_m = 64
    n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))

    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )

    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
        T.pdl_trigger()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if _should_use_mhc_torch_fallback():
        if x.is_cuda and current_platform.is_cuda() and _mhc_post_triton_enabled():
            out = torch.empty_like(residual)
            _mhc_post_triton(x, residual, post_layer_mix, comb_res_mix, out)
            return out
        mixed_residual = torch.einsum(
            "...ij,...ih->...jh",
            comb_res_mix.to(torch.float32),
            residual.to(torch.float32),
        )
        post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
        out = (mixed_residual + post_term).to(residual.dtype)
        _synchronize_mhc_torch_fallback()
        return out
    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


direct_register_custom_op(
    op_name="mhc_pre",
    op_func=mhc_pre,
    mutates_args=[],
    fake_impl=_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="mhc_post",
    op_func=mhc_post,
    mutates_args=[],
    fake_impl=_mhc_post_fake,
)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def hc_head_fuse_tilelang(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Two-pass fused kernel for hc_head.

    Pass 1: accumulate per-token squared sum and hc_mult dot-products
            (projections onto fn rows) using cross-thread reducers.
    Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

    Avoids materialising mixes / rsqrt / pre tensors to global memory.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc_mult * hidden_size
    h_block = math.gcd(h_blk, hidden_size)
    n_h = hidden_size // h_block

    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]
    fn: T.Tensor[[hc_mult, hc_dim], T.float32]  # type: ignore[no-redef,valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef,valid-type]
    hc_base: T.Tensor[[hc_mult], T.float32]  # type: ignore[no-redef,valid-type]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        T.pdl_sync()

        # ------------------------------------------------------------------
        # Pass 1 – for each residual channel m_c and h_block:
        #   • accumulate squared sum (for RMS norm denominator)
        #   • accumulate hc_mult dot-products with fn rows
        # ------------------------------------------------------------------
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for m_c in T.serial(hc_mult):
            for i_h in T.serial(n_h):
                x_local = T.alloc_fragment(h_block, T.float32)
                T.copy(residual[i, m_c, i_h * h_block], x_local)

                for k in T.Parallel(h_block):
                    sqrsum_r[0] += x_local[k] * x_local[k]

                for m_m in T.unroll(hc_mult):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # ------------------------------------------------------------------
        # Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
        # ------------------------------------------------------------------
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
        for m in T.Parallel(hc_mult):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # ------------------------------------------------------------------
        # Pass 2 – apply_mix: pipelined weighted sum over residual channels
        # ------------------------------------------------------------------
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc_mult, h_block), T.float32)
            T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        T.pdl_trigger()


def _hc_head_fused_reference(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Pure-PyTorch reference for `hc_head_fuse_tilelang`.

    Used on platforms where the tilelang HIP/CUDA backend is not available
    (e.g. ROCm builds shipping a tilelang wheel without `target.build.tilelang_hip`).
    Mirrors the math of the tilelang kernel exactly:

        x      = hs_flat.flatten(-2, -1)                # (T, hc_mult * H), fp32
        mixes  = x @ fn.T                               # (T, hc_mult)
        rsqrt  = 1 / sqrt(||x||^2 / (hc_mult * H) + rms_eps)
        pre[m] = sigmoid(mixes[m] * rsqrt * hc_scale[0] + hc_base[m]) + hc_eps
        out    = sum_m pre[m] * hs_flat[:, m, :]        # cast back to bf16

    `out` is mutated in place to keep the same op contract
    (`mutates_args=["out"]`).
    """
    num_tokens = hs_flat.shape[0]
    if num_tokens == 0:
        return
    x = hs_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    # fn: (hc_mult, hc_mult * hidden_size) → mixes: (T, hc_mult)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    # hc_scale has shape (1,); hc_base has shape (hc_mult,)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    # weighted sum over the hc_mult channel dim
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(
        out.dtype
    )
    out.copy_(result)


def _hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return
    if _should_use_mhc_torch_fallback():
        # ROCm: tilelang ships only the CUDA codegen in upstream wheels, so
        # the HIP FFI target (`target.build.tilelang_hip`) is missing and the
        # JIT call would raise `ValueError: Cannot find global function ...`.
        # Ampere/Ada (sm_8x): TileLang requires sm_89+; the JIT compile
        # silently miscompiles or refuses on sm_86/sm_80. Use a numerically
        # equivalent torch fallback instead. `mhc_pre` and `mhc_post` already
        # follow this same pattern above.
        if hs_flat.is_cuda and current_platform.is_cuda() and _mhc_head_triton_enabled():
            _hc_head_triton(
                hs_flat,
                fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_eps,
                hc_eps,
                hc_mult,
            )
            return
        else:
            _hc_head_fused_reference(
                hs_flat,
                fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_eps,
                hc_eps,
                hc_mult,
            )
        _synchronize_mhc_torch_fallback()
        return
    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
    )


direct_register_custom_op(
    op_name="hc_head_fused_kernel",
    op_func=_hc_head_fused_kernel,
    mutates_args=["out"],
)
