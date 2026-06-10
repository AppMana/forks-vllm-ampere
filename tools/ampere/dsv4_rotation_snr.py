# SPDX-License-Identifier: Apache-2.0
"""Measure whether AOT Kronecker-Hadamard rotation improves DSV4 requant SNR.

Gate 1 of the cv-checkpoint workstream: on real DeepSeek-V4-Flash tensors,
compare weight-space SNR of the existing requant formats against the same
formats applied after a group-wise orthogonal rotation of the K dimension.

The rotation uses the regular 4x4 Hadamard-type matrix H4 = (J - 2I)/2
Kronecker-powered to the block size (16/64/256). Unlike Sylvester Hadamard it
has no all-ones column, which otherwise amplifies row-wise outliers (the
ConvRot observation). The deployed scheme folds the rotation into adjacent
weights AOT, so the runtime kernels and graph are unchanged; this study only
needs the rotated-space SNR, which equals the unrotated-space SNR because the
rotation is orthogonal.

Run:
    .venv/bin/python tools/ampere/dsv4_rotation_snr.py \
        --src ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<rev> \
        --layers 5,20,40
"""

import argparse
import glob
import json
import os
import re
import struct

import torch
from safetensors import safe_open

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent.parent))

from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
)

GROUP_SIZE = 32


def kron_hadamard(block: int, device, dtype=torch.float32) -> torch.Tensor:
    """Regular orthogonal Hadamard-type matrix of size 4**k via Kronecker."""
    h4 = (torch.ones(4, 4, device=device, dtype=dtype) - 2 * torch.eye(4, device=device, dtype=dtype)) / 2
    h = h4
    while h.shape[0] < block:
        h = torch.kron(h, h4)
    if h.shape[0] != block:
        raise ValueError(f"block must be a power of 4, got {block}")
    return h


def rotate_k(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Rotate the last (K) dimension group-wise: each block of len(h) mixed."""
    n, k = w.shape
    block = h.shape[0]
    assert k % block == 0
    return (w.reshape(n, k // block, block) @ h).reshape(n, k)


def quant_int4_g32(w: torch.Tensor, scale_mode: str = "absmax7") -> torch.Tensor:
    """Symmetric INT4 group-32 absmax quant + dequant (mirrors requant math)."""
    grouped = w.reshape(*w.shape[:-1], -1, GROUP_SIZE)
    abs_max = grouped.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    div = 7.0 if scale_mode == "absmax7" else 8.0
    scale = abs_max / div
    # scale storage is bf16 in the real pipeline
    scale = scale.to(torch.bfloat16).to(torch.float32)
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7)
    return (q * scale.unsqueeze(-1)).reshape(w.shape)


def quant_int4_g32_mse(w: torch.Tensor) -> torch.Tensor:
    """Symmetric INT4 group-32 with per-group MSE scale search."""
    grouped = w.reshape(*w.shape[:-1], -1, GROUP_SIZE)
    abs_max = grouped.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    best_deq = None
    best_err = None
    for div in torch.linspace(5.0, 9.5, 19):
        scale = (abs_max / div).to(torch.bfloat16).to(torch.float32)
        q = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7)
        deq = q * scale.unsqueeze(-1)
        err = (deq - grouped).pow(2).sum(dim=-1)
        if best_err is None:
            best_err, best_deq = err, deq
        else:
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_deq = torch.where(mask.unsqueeze(-1), deq, best_deq)
    return best_deq.reshape(w.shape)


def quant_uint4_asym_g32(w: torch.Tensor) -> torch.Tensor:
    """Affine UINT4 group-32 (AWQ-style zero point), absmin/absmax range."""
    grouped = w.reshape(*w.shape[:-1], -1, GROUP_SIZE)
    gmin = torch.minimum(grouped.amin(dim=-1), torch.zeros((), device=w.device))
    gmax = torch.maximum(grouped.amax(dim=-1), torch.zeros((), device=w.device))
    scale = ((gmax - gmin).clamp(min=torch.finfo(torch.float32).tiny) / 15.0)
    scale = scale.to(torch.bfloat16).to(torch.float32)
    zp = torch.round(-gmin / scale).clamp(0, 15)
    q = torch.round(grouped / scale.unsqueeze(-1) + zp.unsqueeze(-1)).clamp(0, 15)
    return ((q - zp.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(w.shape)


def quant_uint4_asym_g32_mse(w: torch.Tensor) -> torch.Tensor:
    """Affine UINT4 group-32 with a shrink search on the range (GPTQ-style)."""
    grouped = w.reshape(*w.shape[:-1], -1, GROUP_SIZE)
    gmin0 = torch.minimum(grouped.amin(dim=-1), torch.zeros((), device=w.device))
    gmax0 = torch.maximum(grouped.amax(dim=-1), torch.zeros((), device=w.device))
    best_deq = None
    best_err = None
    for shrink in torch.linspace(0.7, 1.0, 13):
        gmin, gmax = gmin0 * shrink, gmax0 * shrink
        scale = ((gmax - gmin).clamp(min=torch.finfo(torch.float32).tiny) / 15.0)
        scale = scale.to(torch.bfloat16).to(torch.float32)
        zp = torch.round(-gmin / scale).clamp(0, 15)
        q = torch.round(grouped / scale.unsqueeze(-1) + zp.unsqueeze(-1)).clamp(0, 15)
        deq = (q - zp.unsqueeze(-1)) * scale.unsqueeze(-1)
        err = (deq - grouped).pow(2).sum(dim=-1)
        if best_err is None:
            best_err, best_deq = err, deq
        else:
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_deq = torch.where(mask.unsqueeze(-1), deq, best_deq)
    return best_deq.reshape(w.shape)


def quant_int8_channel(w: torch.Tensor) -> torch.Tensor:
    """Symmetric INT8 per-output-channel quant + dequant (AllSpark format)."""
    scale = w.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    scale = scale.to(torch.bfloat16).to(torch.float32)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-128, 127)
    return q * scale.unsqueeze(1)


def snr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    err = (ref - approx).float()
    return float(10 * torch.log10(ref.float().pow(2).sum() / err.pow(2).sum().clamp(min=1e-30)))


def shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def load_dequant(snap: str, weight_map: dict, name: str, device) -> torch.Tensor:
    """Load a tensor + its scale and dequantize to FP32 [N, K]."""
    shard = os.path.join(snap, weight_map[name])
    scale_name = name.replace(".weight", ".scale")
    with safe_open(shard, framework="pt", device=str(device)) as h:
        w = h.get_tensor(name)
        s = h.get_tensor(scale_name) if scale_name in weight_map and weight_map[scale_name] == weight_map[name] else None
    if s is None:
        with safe_open(os.path.join(snap, weight_map[scale_name]), framework="pt", device=str(device)) as h:
            s = h.get_tensor(scale_name)
    if w.dtype == torch.int8:  # packed MXFP4 nibbles
        nib = _unpack_int4_pairs(w)
        fp4 = _e2m1_nibble_to_fp32(nib)
        scale = _e8m0_to_fp32_scale(s)
        return (fp4.reshape(*fp4.shape[:-1], -1, GROUP_SIZE) * scale.unsqueeze(-1)).reshape(fp4.shape)
    if w.dtype == torch.float8_e4m3fn:
        deq = w.to(torch.float32)
        scale = _e8m0_to_fp32_scale(s)
        bn = (w.shape[0] + scale.shape[0] - 1) // scale.shape[0]
        bk = (w.shape[1] + scale.shape[1] - 1) // scale.shape[1]
        full = scale.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)
        return deq * full[: w.shape[0], : w.shape[1]]
    raise TypeError(f"{name}: unsupported dtype {w.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=None)
    parser.add_argument("--layers", default="5,20,40")
    parser.add_argument("--blocks", default="16,64,256")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    snap = args.src or os.path.dirname(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"
                "/snapshots/*/model.safetensors.index.json"
            )
        )[0]
    )
    weight_map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))[
        "weight_map"
    ]
    device = torch.device(args.device)
    blocks = [int(b) for b in args.blocks.split(",")]
    layers = [int(x) for x in args.layers.split(",")]

    expert_formats = {
        "int4_sym": quant_int4_g32,
        "int4_sym_mse": quant_int4_g32_mse,
        "uint4_asym": quant_uint4_asym_g32,
        "uint4_asym_mse": quant_uint4_asym_g32_mse,
    }
    dense_formats = {"int8_channel": quant_int8_channel}

    targets = []
    for layer in layers:
        targets += [
            (f"layers.{layer}.ffn.experts.0.w1.weight", expert_formats),
            (f"layers.{layer}.ffn.experts.0.w2.weight", expert_formats),
            (f"layers.{layer}.attn.wq_b.weight", dense_formats),
            (f"layers.{layer}.attn.wo_b.weight", dense_formats),
            (f"layers.{layer}.ffn.shared_experts.w1.weight", dense_formats),
        ]

    print(f"{'tensor':<44} {'fmt':<15} {'plain':>7}" + "".join(f" {'H'+str(b):>7}" for b in blocks) + "  (SNR dB)")
    for name, formats in targets:
        if name not in weight_map:
            print(f"{name:<44} missing")
            continue
        w = load_dequant(snap, weight_map, name, device)
        for fmt, quant in formats.items():
            base = snr_db(w, quant(w))
            row = f"{name:<44} {fmt:<15} {base:>7.2f}"
            for b in blocks:
                if w.shape[1] % b:
                    row += f" {'n/a':>7}"
                    continue
                h = kron_hadamard(b, device)
                wr = rotate_k(w, h)
                row += f" {snr_db(wr, quant(wr)):>7.2f}"
            print(row)
        del w


if __name__ == "__main__":
    main()
