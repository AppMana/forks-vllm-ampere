#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe AllSpark Ampere W8A16 numerics and speed for dsv4_int dense linears."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    dequantize_allspark_uint8_w8a16,
)
from vllm.model_executor.layers.quantization.utils.allspark_utils import (  # noqa: E402
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
)
from vllm.utils.platform_utils import num_compute_units  # noqa: E402


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    noise = (reference.float() - actual.float()).norm()
    if noise == 0:
        return float("inf")
    return (20 * torch.log10(reference.float().norm() / noise)).item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-bf16-baseline", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "allspark_w8a16_gemm"
    ):
        raise RuntimeError("AllSpark W8A16 op is not available in this build")
    if args.k % 16 or args.n % 16:
        raise ValueError("AllSpark requires k and n to be multiples of 16")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(0)

    weight = torch.randn(args.n, args.k, device=device, dtype=torch.float32) * 0.02
    scale = weight.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)
    scale = scale.to(dtype)
    q_reorder, scale_reorder, _ = ops.allspark_repack_weight(
        q_biased.t().contiguous(),
        scale.reshape(1, -1).contiguous(),
        None,
        False,
    )

    x = torch.randn(args.m, args.k, device=device, dtype=dtype) * 0.02
    props = torch.cuda.get_device_properties(device)
    sm_version = props.major * 10 + props.minor
    sm_count = num_compute_units(device.index or torch.cuda.current_device())

    out = ops.allspark_w8a16_gemm(
        a=x,
        b_qweight=q_reorder,
        b_scales=scale_reorder,
        b_qzeros=None,
        n=args.n,
        group_size=-1,
        sm_count=sm_count,
        sm_version=sm_version,
        CUBLAS_M_THRESHOLD=ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
        has_zp=False,
        n32k16_reorder=True,
    )
    torch.cuda.synchronize()

    ref_weight = dequantize_allspark_uint8_w8a16(q_biased, scale).to(dtype)
    ref = torch.nn.functional.linear(x, ref_weight)
    snr_db = _snr_db(ref, out)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        out = ops.allspark_w8a16_gemm(
            a=x,
            b_qweight=q_reorder,
            b_scales=scale_reorder,
            b_qzeros=None,
            n=args.n,
            group_size=-1,
            sm_count=sm_count,
            sm_version=sm_version,
            CUBLAS_M_THRESHOLD=ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
            has_zp=False,
            n32k16_reorder=True,
        )
    end.record()
    torch.cuda.synchronize()
    allspark_ms = start.elapsed_time(end) / args.iters

    bf16_ms = None
    if not args.skip_bf16_baseline:
        start.record()
        for _ in range(args.iters):
            torch.nn.functional.linear(x, ref_weight)
        end.record()
        torch.cuda.synchronize()
        bf16_ms = start.elapsed_time(end) / args.iters

    print(
        json.dumps(
            {
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "dtype": str(dtype),
                "sm_version": sm_version,
                "snr_db_vs_dequant_bf16": snr_db,
                "allspark_mean_ms": allspark_ms,
                "bf16_dequant_linear_mean_ms": bf16_ms,
                "allspark_speedup_vs_bf16_dequant": (
                    bf16_ms / allspark_ms if bf16_ms is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
