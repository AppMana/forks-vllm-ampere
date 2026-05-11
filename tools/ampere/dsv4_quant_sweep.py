#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep integer destination formats for DeepSeek V4 tensors.

The sweep reconstructs original FP4/FP8 tensors into FP32 teacher values and
then compares candidate integer formats. It is a format-selection tool, not a
checkpoint converter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dsv4_checkpoint_audit import audit_checkpoint  # noqa: E402

from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
)


def _load_tensor(checkpoint: Path, shard: str, name: str, device: str) -> torch.Tensor:
    with safe_open(checkpoint / shard, framework="pt", device=device) as handle:
        return handle.get_tensor(name)


def _fp4_dequant(weight_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    nibble = _unpack_int4_pairs(weight_packed)
    fp4 = _e2m1_nibble_to_fp32(nibble)
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    return (fp4.reshape(*fp4.shape[:-1], -1, 32) * scale.unsqueeze(-1)).reshape(
        fp4.shape
    )


def _fp8_dequant(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    bn, bk = block_size
    n, k = weight_fp8.shape
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
    return weight_fp8.to(torch.float32) * scale_full[:n, :k]


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    err = ref - actual.float()
    noise = err.norm()
    snr = (
        math.inf
        if noise.item() == 0
        else (20 * torch.log10(ref.norm() / noise)).item()
    )
    return {
        "snr_db": snr,
        "rmse": torch.sqrt(torch.mean(err * err)).item(),
        "mean_abs": err.abs().mean().item(),
        "max_abs": err.abs().max().item(),
    }


def _sample_rows(x: torch.Tensor, max_rows: int | None) -> torch.Tensor:
    if max_rows is None or x.ndim < 2 or x.shape[0] <= max_rows:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_rows, device=x.device).round().long()
    return x.index_select(0, idx)


def _quant_dequant_sym_last(
    x: torch.Tensor, *, bits: int, group_size: int
) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        return torch.full_like(x, torch.nan)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    scale = grouped.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny) / qmax
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(qmin, qmax)
    return (q * scale.unsqueeze(-1)).reshape_as(x)


def _quant_dequant_asym_last(
    x: torch.Tensor, *, bits: int, group_size: int
) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        return torch.full_like(x, torch.nan)
    qmax = (1 << bits) - 1
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    xmin = torch.minimum(grouped.amin(dim=-1), torch.zeros((), device=x.device))
    xmax = torch.maximum(grouped.amax(dim=-1), torch.zeros((), device=x.device))
    scale = (xmax - xmin).clamp(min=torch.finfo(torch.float32).tiny) / qmax
    zero = torch.round(-xmin / scale).clamp(0, qmax)
    q = torch.round(grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0, qmax)
    return ((q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape_as(x)


def _quant_dequant_sym_block(
    x: torch.Tensor, *, bits: int, block_size: tuple[int, int]
) -> torch.Tensor:
    bn, bk = block_size
    n, k = x.shape
    gn = (n + bn - 1) // bn
    gk = (k + bk - 1) // bk
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    padded = F.pad(x.float(), (0, gk * bk - k, 0, gn * bn - n))
    blocked = padded.reshape(gn, bn, gk, bk).permute(0, 2, 1, 3)
    scale = blocked.abs().amax(dim=(-2, -1)).clamp(
        min=torch.finfo(torch.float32).tiny
    ) / qmax
    q = torch.round(blocked / scale[:, :, None, None]).clamp(qmin, qmax)
    out = (q * scale[:, :, None, None]).permute(0, 2, 1, 3).reshape(gn * bn, gk * bk)
    return out[:n, :k]


def _quant_dequant_asym_block(
    x: torch.Tensor, *, bits: int, block_size: tuple[int, int]
) -> torch.Tensor:
    bn, bk = block_size
    n, k = x.shape
    gn = (n + bn - 1) // bn
    gk = (k + bk - 1) // bk
    qmax = (1 << bits) - 1
    padded = F.pad(x.float(), (0, gk * bk - k, 0, gn * bn - n))
    blocked = padded.reshape(gn, bn, gk, bk).permute(0, 2, 1, 3)
    xmin = torch.minimum(blocked.amin(dim=(-2, -1)), torch.zeros((), device=x.device))
    xmax = torch.maximum(blocked.amax(dim=(-2, -1)), torch.zeros((), device=x.device))
    scale = (xmax - xmin).clamp(min=torch.finfo(torch.float32).tiny) / qmax
    zero = torch.round(-xmin / scale).clamp(0, qmax)
    q = torch.round(blocked / scale[:, :, None, None] + zero[:, :, None, None])
    q = q.clamp(0, qmax)
    out = ((q - zero[:, :, None, None]) * scale[:, :, None, None]).permute(0, 2, 1, 3)
    return out.reshape(gn * bn, gk * bk)[:n, :k]


def _candidate_outputs(
    role: str, target: torch.Tensor
) -> Iterable[tuple[str, torch.Tensor]]:
    if role == "routed_expert_mxfp4_weight":
        for group in (32, 64, 128):
            yield f"sym_int4_g{group}", _quant_dequant_sym_last(
                target, bits=4, group_size=group
            )
            yield f"asym_uint4_g{group}", _quant_dequant_asym_last(
                target, bits=4, group_size=group
            )
        for bits in (5, 6, 8):
            yield f"asym_uint{bits}_g32_bound", _quant_dequant_asym_last(
                target, bits=bits, group_size=32
            )
        return

    for block in ((128, 128), (64, 128), (128, 64)):
        yield f"sym_int8_b{block[0]}x{block[1]}", _quant_dequant_sym_block(
            target, bits=8, block_size=block
        )
        yield f"asym_uint8_b{block[0]}x{block[1]}", _quant_dequant_asym_block(
            target, bits=8, block_size=block
        )
    yield "asym_uint4_g128_bound", _quant_dequant_asym_last(
        target, bits=4, group_size=128
    )


def run_sweep(
    src: Path,
    *,
    roles: set[str],
    max_tensors: int,
    max_rows: int | None,
    device: str,
) -> dict[str, object]:
    manifest = audit_checkpoint(src)
    records = manifest["records"]
    assert isinstance(records, list)
    by_name = {r["name"]: r for r in records}

    results: list[dict[str, object]] = []
    for record in records:
        if len(results) >= max_tensors:
            break
        if record["role"] not in roles or not record["name"].endswith(".weight"):
            continue
        scale_name = record["scale_name"]
        if scale_name not in by_name:
            continue
        scale_record = by_name[scale_name]
        weight = _load_tensor(src, record["shard"], record["name"], device)
        scale = _load_tensor(src, scale_record["shard"], scale_name, device)
        if record["role"] == "routed_expert_mxfp4_weight":
            target = _fp4_dequant(weight, scale)
        else:
            target = _fp8_dequant(weight, scale)
        target = _sample_rows(target, max_rows)

        candidate_metrics = []
        for name, actual in _candidate_outputs(record["role"], target):
            if torch.isnan(actual).any():
                continue
            candidate_metrics.append({"candidate": name, **_metrics(target, actual)})
        candidate_metrics.sort(key=lambda item: item["snr_db"], reverse=True)
        results.append(
            {
                "name": record["name"],
                "role": record["role"],
                "shape": record["shape"],
                "candidates": candidate_metrics,
            }
        )

    return {"checkpoint": str(src), "results": results}


def _print_table(report: dict[str, object]) -> None:
    for row in report["results"]:
        assert isinstance(row, dict)
        print(f"\n{row['name']} ({row['role']}, shape={row['shape']})")
        for candidate in row["candidates"][:8]:
            print(
                "  {candidate:24s} snr={snr_db:7.2f}dB "
                "rmse={rmse:.4g} mean_abs={mean_abs:.4g} max_abs={max_abs:.4g}".format(
                    **candidate
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument(
        "--roles",
        default=(
            "routed_expert_mxfp4_weight,dense_fp8_weight,"
            "indexer_qk_fp8_weight,mtp_fp8_weight"
        ),
    )
    parser.add_argument("--max-tensors", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run_sweep(
        args.src.resolve(),
        roles={r.strip() for r in args.roles.split(",") if r.strip()},
        max_tensors=args.max_tensors,
        max_rows=args.max_rows,
        device=args.device,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_table(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
