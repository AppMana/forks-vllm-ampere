# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests."""

import time
from collections.abc import Iterable

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    7,
    8,
    16,
    17,
    18,
    19,
    20,
    24,
    28,
    32,
    33,
    52,
    64,
    128,
    198,
    396,
    624,
    256,
    512,
    1024,
    1992,
    2048,
    4096,
    8192,
    16_384,
)


def _compute_mhc_pre_num_split(
    *,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> int:
    block_k = 64
    block_m = 64
    k = hc_mult * hidden_size
    grid_size = cdiv(num_tokens, block_m)
    split_k = num_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
    requested_token_sizes: list[int] | None,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    if requested_token_sizes is None:
        max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
        candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
        candidates.extend(cudagraph_capture_sizes)
        candidates.append(max_auto_tokens)
        candidates = _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)
    else:
        candidates = _normalize_token_sizes(
            requested_token_sizes,
            max_tokens=max_tokens,
        )

    return candidates


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _get_cuda_num_sms(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    return torch.cuda.get_device_properties(index).multi_processor_count


def _finalize_triton_async_compiles() -> None:
    try:
        from triton.runtime import _async_compile
    except ImportError:
        return

    async_mode = _async_compile.active_mode.get()
    if async_mode is None:
        return

    for future in list(async_mode.raw_futures):
        async_mode.future_kernels[future._key].result(async_mode.ignore_errors)


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    if not hasattr(model, "_mtp_hidden_buffer"):
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        model.hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


def _find_triton_channel_w8a16_linears(
    model: torch.nn.Module,
) -> list[torch.nn.Module]:
    seen_shapes: set[tuple[int, int, torch.device]] = set()
    linears: list[torch.nn.Module] = []
    for module in model.modules():
        if not getattr(module, "_dsv4_int_triton_channel", False):
            continue
        weight = getattr(module, "weight", None)
        scales = getattr(module, "weight_scale_inv", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scales, torch.Tensor):
            continue
        if weight.device.type != "cuda" or weight.dtype != torch.uint8:
            continue
        shape_key = (int(weight.shape[0]), int(weight.shape[1]), weight.device)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        linears.append(module)
    return linears


def _warmup_triton_channel_w8a16(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    linears = _find_triton_channel_w8a16_linears(model)
    if not linears:
        return

    from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
        triton_channel_w8a16_gemm,
    )

    for layer in linears:
        weight = layer.weight
        scales = layer.weight_scale_inv
        hidden_size = int(weight.shape[1])
        max_tokens = max(token_sizes)
        x = torch.zeros(
            max_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=weight.device,
        )
        for size in token_sizes:
            triton_channel_w8a16_gemm(x[:size].contiguous(), weight, scales)


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP:
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    num_sms = _get_cuda_num_sms(device)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        hidden_size=int(layer.hidden_size),
        hc_mult=int(layer.hc_mult),
        num_sms=num_sms,
        requested_token_sizes=envs.VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        _warmup_triton_channel_w8a16(model, token_sizes)
        _finalize_triton_async_compiles()
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
