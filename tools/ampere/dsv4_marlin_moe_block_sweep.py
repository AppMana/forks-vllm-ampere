# SPDX-License-Identifier: Apache-2.0
"""Standalone repro for the June-csrc illegal access on small-expert checkpoints.

Sweeps (E, M) over fused_marlin_moe at real DSV4 expert shapes
(hidden 4096, moe_intermediate 2048, top-6, group 32, uint4b8 W4A16) and
synchronizes after every call. Hypothesis: the m-block selector

    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

picks 48/64 only when E is small and M is large (the mini testbed profile
run), a tile config production (E=256 -> block 32) never executes.

    python3 tools/ampere/dsv4_marlin_moe_block_sweep.py
"""

import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.dsv4_int import Dsv4Int4MoEMethod
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.scalar_type import scalar_types

HIDDEN = 4096
INTER = 2048
TOPK = 6
GROUP = 32
DTYPE = torch.bfloat16


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (((high & 0x0F) << 4) | (low & 0x0F)).view(torch.int8)


def selected_block_size_m(M: int, E: int) -> int:
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * TOPK / E / block_size_m < 0.9:
            break
    return block_size_m


def make_expert_weights(E: int, device: torch.device):
    g = torch.Generator(device="cpu").manual_seed(7)

    def quant(n: int, k: int):
        nib = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, generator=g)
        q = pack_nibbles(nib).to(device)
        s = (torch.rand(E, n, k // GROUP, generator=g) * 0.02 + 0.001).to(DTYPE).to(device)
        return q, s

    w13_q, w13_s = quant(2 * INTER, HIDDEN)
    w2_q, w2_s = quant(HIDDEN, INTER)
    w13_marlin = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w13_q, size_n=2 * INTER, size_k=HIDDEN, is_a_8bit=False
    )
    w2_marlin = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        w2_q, size_n=HIDDEN, size_k=INTER, is_a_8bit=False
    )
    w13_scale = marlin_moe_permute_scales(
        w13_s.transpose(1, 2).contiguous(),
        size_k=HIDDEN, size_n=2 * INTER, group_size=GROUP, is_a_8bit=False,
    )
    w2_scale = marlin_moe_permute_scales(
        w2_s.transpose(1, 2).contiguous(),
        size_k=INTER, size_n=HIDDEN, group_size=GROUP, is_a_8bit=False,
    )
    return w13_marlin, w2_marlin, w13_scale, w2_scale


def run_case(E: int, M: int, weights, device: torch.device) -> str:
    w13_marlin, w2_marlin, w13_scale, w2_scale = weights
    x = torch.randn(M, HIDDEN, dtype=DTYPE, device=device) * 0.1
    score = torch.randn(M, E, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(score, dim=-1), k=TOPK)
    topk_ids = topk_ids.to(torch.int32)
    empty_i32 = torch.empty(E, 0, dtype=torch.int32, device=device)
    out = fused_marlin_moe(
        x, w13_marlin, w2_marlin, None, None, w13_scale, w2_scale,
        topk_weights, topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        global_num_experts=E,
        g_idx1=empty_i32, g_idx2=empty_i32,
        sort_indices1=empty_i32, sort_indices2=empty_i32,
        workspace=marlin_make_workspace_new(device, 4),
        is_k_full=True,
        input_dtype=None,
        input_global_scale1=None,
        input_global_scale2=None,
    )
    torch.cuda.synchronize()
    bad = (~torch.isfinite(out)).sum().item()
    return f"OK (nonfinite={bad}, out_absmax={out.abs().max().item():.4f})"


def main() -> None:
    device = torch.device("cuda")
    for E in (32, 64, 256):
        weights = make_expert_weights(E, device)
        for M in (8, 64, 256, 1024):
            blk = selected_block_size_m(M, E)
            tag = f"E={E:<4d} M={M:<5d} block_size_m={blk}"
            try:
                status = run_case(E, M, weights, device)
            except Exception as exc:  # noqa: BLE001
                print(f"{tag}: FAULT {type(exc).__name__}: {exc}", flush=True)
                raise SystemExit(1)
            print(f"{tag}: {status}", flush=True)
    print("ALL CASES PASSED", flush=True)


if __name__ == "__main__":
    main()
