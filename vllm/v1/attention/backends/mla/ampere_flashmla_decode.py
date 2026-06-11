# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_86 flash-MLA decode tail for the DSV4 sparse path.

Replaces `matmul_sparse_mla_attention_with_sink` over the materialized
[T, C, 576] bf16 decode workspace with the forks-flash-mla-ampere-dsv4
dense kernel (single-buffered kP=1 pipeline on sm_86, dv=576 so the RoPE
tail flows through attention for the inverse-RoPE o-projection).

The workspace holds two prefix-valid sections per token (compressed top-k,
then SWA), so the kernel runs once per section with the section lengths as
``cache_seqlens`` and the results are merged in log-space together with the
attention sink:

    m      = max(lse_topk, lse_swa, sink)
    w_i    = exp(lse_i - m)        (0 where the section is empty)
    out    = (w_topk * out_topk + w_swa * out_swa) / (w_topk + w_swa + exp(sink - m))

Enabled with VLLM_AMPERE_FLASHMLA_DECODE=1 when the ``flash_mla`` package
from ~/Documents/forks-flash-mla-ampere-dsv4 is importable. Measured on an
RTX A5000 against the Triton matmul path at decode shapes: 1.2x (T=1,
C=512) to 19.6x (T=12, C=1088), cos > 0.9999 including the sink merge
(tools/ampere/bench_sparse_mla_decode_flashmla.py).
"""

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_NEG_INF = float("-inf")


@functools.lru_cache(maxsize=64)
def _cached_block_table(
    num_tokens: int,
    pages_per_token: int,
    first_page: int,
    section_pages: int,
    device_index: int,
) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    base = torch.arange(
        num_tokens, device=device, dtype=torch.int32
    ).mul_(pages_per_token).add_(first_page)
    return (
        base.unsqueeze(1)
        + torch.arange(section_pages, device=device, dtype=torch.int32).unsqueeze(0)
    ).contiguous()

_PAGE_SIZE = 32  # kBlockN of the sm80/86 kernel


@functools.cache
def ampere_flashmla_decode_min_tokens() -> int:
    """Decode-batch size below which the Triton paths stay active.

    At T=1 the extra materialization plus two kernel launches and the merge
    cost more than the direct paged Triton multihead kernel (chain C=1 row:
    7.7 vs 8.1 tok/s); the flash kernel pulls ahead from T~4 (3x at the op
    level) and dominates at T=12 (8-20x).
    """
    return int(os.environ.get("VLLM_AMPERE_FLASHMLA_DECODE_MIN_TOKENS", "4"))


@functools.cache
def ampere_flashmla_decode_enabled() -> bool:
    if os.environ.get("VLLM_AMPERE_FLASHMLA_DECODE", "0") != "1":
        return False
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    if capability is None or capability[0] != 8:
        return False
    try:
        import flash_mla  # noqa: F401
    except ImportError:
        logger.warning_once(
            "VLLM_AMPERE_FLASHMLA_DECODE=1 but the flash_mla package "
            "(forks-flash-mla-ampere-dsv4) is not importable; falling back "
            "to the Triton matmul decode path."
        )
        return False
    return True


def ampere_flashmla_supports(
    num_candidates: int, compressed_topk: int, head_dim: int
) -> bool:
    """Both workspace sections must be kernel-page aligned."""
    return (
        head_dim in (512, 576)
        and compressed_topk % _PAGE_SIZE == 0
        and (num_candidates - compressed_topk) % _PAGE_SIZE == 0
    )


def _section_attention(
    q4: torch.Tensor,  # [T, 1, H, D]
    blocked_kv: torch.Tensor,  # [T * C / PAGE, PAGE, 1, D]
    pages_per_token: int,
    first_page: int,
    section_pages: int,
    lens: torch.Tensor,  # [T] int32
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    T = q4.shape[0]
    H = q4.shape[2]
    D = q4.shape[3]
    block_table = _cached_block_table(
        T, pages_per_token, first_page, section_pages,
        q4.device.index if q4.device.index is not None else 0,
    )
    # Per-token empty sections: run over one (garbage) page and let the
    # caller's -inf lse mask zero the weight; cache_seqlens=0 is not
    # supported by the kernel.
    lens = lens.to(torch.int32).clamp(min=1)
    meta, splits = get_mla_metadata(lens, H, 1)
    out, lse = flash_mla_with_kvcache(
        q4,
        blocked_kv,
        block_table,
        lens,
        D,
        meta,
        splits,
        softmax_scale=scale,
        causal=False,
        warp_spec=False,
    )
    # out: [T, 1, H, D]; lse: [T, H, 1] base-e (includes scale)
    return out[:, 0], lse.float().reshape(T, H)


def ampere_flashmla_sparse_decode(
    q: torch.Tensor,  # [T, H, 576] or [T, 1, H, 576]
    combined_kv: torch.Tensor,  # [T, C, 576] bf16, contiguous
    topk_lens: torch.Tensor,  # [T]
    swa_lens: torch.Tensor,  # [T]
    compressed_topk: int,
    scale: float,
    attn_sink: torch.Tensor,  # [H] float32 logits
    output: torch.Tensor,  # [T, H_padded, 576]
    num_heads: int,
) -> None:
    if q.dim() == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    T, H, D = q.shape
    C = combined_kv.shape[1]
    assert combined_kv.is_contiguous()
    assert ampere_flashmla_supports(C, compressed_topk, D)

    q_active = q[:, :num_heads].contiguous() if num_heads != H else q
    q4 = q_active.view(T, 1, num_heads, D)
    pages_per_token = C // _PAGE_SIZE
    blocked_kv = combined_kv.view(T * pages_per_token, _PAGE_SIZE, 1, D)

    if compressed_topk > 0:
        out1, lse1 = _section_attention(
            q4, blocked_kv, pages_per_token, 0,
            compressed_topk // _PAGE_SIZE, topk_lens, scale,
        )
    else:
        out1 = torch.zeros(T, num_heads, D, dtype=q.dtype, device=q.device)
        lse1 = torch.full((T, num_heads), _NEG_INF, device=q.device)
    if C > compressed_topk:
        out2, lse2 = _section_attention(
            q4, blocked_kv, pages_per_token, compressed_topk // _PAGE_SIZE,
            (C - compressed_topk) // _PAGE_SIZE, swa_lens, scale,
        )
    else:
        out2 = torch.zeros(T, num_heads, D, dtype=q.dtype, device=q.device)
        lse2 = torch.full((T, num_heads), _NEG_INF, device=q.device)

    # Empty sections attend over one garbage page (lens clamped to 1 in
    # _section_attention); force their softmax weight to zero. Outputs stay
    # finite, so the fused merge needs no NaN handling.
    lse1 = torch.where((topk_lens > 0).unsqueeze(1), lse1, _NEG_INF)
    lse2 = torch.where((swa_lens > 0).unsqueeze(1), lse2, _NEG_INF)

    from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
        merge_two_sparse_mla_subsets_with_sink,
    )

    merge_two_sparse_mla_subsets_with_sink(
        out1, lse1, out2, lse2,
        attn_sink[:num_heads],
        output[:, :num_heads],
    )
    if output.shape[1] > num_heads:
        output[:, num_heads:].zero_()
