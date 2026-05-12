#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DeepSeek V4 source FP4/FP8 tensors to a dsv4_int checkpoint.

This is a correctness diagnostic for generated-text failures. It compares the
real upstream checkpoint math against the converted checkpoint at boundaries
that matter for logits:

* tensor dequantization parity
* dense linear output parity
* full routed-expert output parity: w1/w3 -> silu(gate) * up -> w2

It is intentionally independent of vLLM model loading so it can catch bad
checkpoint conversion, layer remapping, tensor orientation, and scale handling
before scheduler/runtime effects enter the picture.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dsv4_checkpoint_audit import classify_tensor, matched_scale_name  # noqa: E402
from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
    dequantize_allspark_uint8_w8a16,
    dequantize_int4_w4a16,
    dequantize_int8_w8a16,
    dequantize_uint4_asym_w4a16,
)


@dataclass(frozen=True)
class Metric:
    snr_db: float
    rel_l2: float
    rmse: float
    mean_abs: float
    max_abs: float


@dataclass(frozen=True)
class CheckResult:
    check: str
    name: str
    shape: tuple[int, ...]
    metric: Metric
    details: dict[str, object]


class Checkpoint:
    def __init__(self, path: Path, device: str) -> None:
        self.path = path
        self.device = device
        index_path = path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        self.weight_map: dict[str, str] = json.loads(index_path.read_text())[
            "weight_map"
        ]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def shard(self, name: str) -> str:
        try:
            return self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} is not in {self.path}") from exc

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(
            self.path / self.shard(name), framework="pt", device=self.device
        ) as handle:
            return handle.get_tensor(name)

    def dtype_name(self, name: str) -> str:
        with safe_open(self.path / self.shard(name), framework="pt", device="cpu") as h:
            return h.get_slice(name).get_dtype()


def _metric(reference: torch.Tensor, actual: torch.Tensor) -> Metric:
    ref = reference.float()
    got = actual.float()
    err = ref - got
    ref_norm = ref.norm()
    err_norm = err.norm()
    if err_norm.item() == 0:
        snr = math.inf
        rel = 0.0
    elif ref_norm.item() == 0:
        snr = -math.inf
        rel = math.inf
    else:
        rel = (err_norm / ref_norm).item()
        snr = (20 * torch.log10(ref_norm / err_norm)).item()
    return Metric(
        snr_db=snr,
        rel_l2=rel,
        rmse=torch.sqrt(torch.mean(err * err)).item(),
        mean_abs=err.abs().mean().item(),
        max_abs=err.abs().max().item(),
    )


def _fp4_dequant(weight_packed: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    nibble = _unpack_int4_pairs(weight_packed)
    fp4 = _e2m1_nibble_to_fp32(nibble)
    scale = _e8m0_to_fp32_scale(scale_e8m0)
    if fp4.shape[-1] != scale.shape[-1] * 32:
        raise ValueError(f"{fp4.shape=} does not match {scale.shape=}")
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


def _source_dequant(src: Checkpoint, name: str) -> torch.Tensor:
    dtype = src.dtype_name(name)
    role, action = classify_tensor(name, dtype)
    if action == "preserve":
        return src.tensor(name)
    scale_name = matched_scale_name(name)
    if scale_name is None:
        raise ValueError(f"{name} has no matched scale")
    weight = src.tensor(name)
    scale = src.tensor(scale_name)
    if role == "routed_expert_mxfp4_weight":
        return _fp4_dequant(weight, scale)
    if role in {"dense_fp8_weight", "indexer_qk_fp8_weight", "mtp_fp8_weight"}:
        return _fp8_dequant(weight, scale)
    raise ValueError(f"unsupported source role for {name}: {role} ({dtype})")


def _converted_dequant(dst: Checkpoint, name: str, dense_strategy: str) -> torch.Tensor:
    dtype = dst.dtype_name(name)
    role, action = classify_tensor(name, dtype)
    if role == "unknown" and name.endswith(".weight"):
        role, action = classify_tensor(name, "F8_E4M3")
    if action == "preserve":
        return dst.tensor(name)
    scale_name = matched_scale_name(name)
    if scale_name is None:
        raise ValueError(f"{name} has no matched scale")
    weight = dst.tensor(name)
    scale = dst.tensor(scale_name)
    zero_name = f"{name[:-len('.weight')]}.zero_points"
    if role == "routed_expert_mxfp4_weight":
        if dst.has(zero_name):
            return dequantize_uint4_asym_w4a16(weight, scale, dst.tensor(zero_name))
        return dequantize_int4_w4a16(weight, scale)
    if role in {"dense_fp8_weight", "indexer_qk_fp8_weight", "mtp_fp8_weight"}:
        if dense_strategy == "channel":
            return dequantize_allspark_uint8_w8a16(weight, scale)
        return dequantize_int8_w8a16(weight, scale)
    raise ValueError(f"unsupported converted role for {name}: {role} ({dtype})")


def _to_compute_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype=dtype).contiguous()


def _linear_check(
    src: Checkpoint,
    dst: Checkpoint,
    name: str,
    *,
    dense_strategy: str,
    tokens: int,
    compute_dtype: torch.dtype,
    seed: int,
) -> CheckResult:
    torch.manual_seed(seed)
    src_w = _to_compute_dtype(_source_dequant(src, name), compute_dtype)
    dst_w = _to_compute_dtype(_converted_dequant(dst, name, dense_strategy), compute_dtype)
    x = torch.randn(tokens, src_w.shape[1], device=src_w.device, dtype=compute_dtype)
    src_out = F.linear(x, src_w)
    dst_out = F.linear(x, dst_w)
    return CheckResult(
        check="linear",
        name=name,
        shape=tuple(src_w.shape),
        metric=_metric(src_out, dst_out),
        details={"tokens": tokens, "compute_dtype": str(compute_dtype)},
    )


def _tensor_check(
    src: Checkpoint,
    dst: Checkpoint,
    name: str,
    *,
    dense_strategy: str,
    max_rows: int | None,
) -> CheckResult:
    src_w = _source_dequant(src, name)
    dst_w = _converted_dequant(dst, name, dense_strategy)
    if max_rows is not None and src_w.ndim >= 2 and src_w.shape[0] > max_rows:
        idx = torch.linspace(
            0, src_w.shape[0] - 1, max_rows, device=src_w.device
        ).round().long()
        src_w = src_w.index_select(0, idx)
        dst_w = dst_w.index_select(0, idx)
    return CheckResult(
        check="tensor",
        name=name,
        shape=tuple(src_w.shape),
        metric=_metric(src_w, dst_w),
        details={"max_rows": max_rows},
    )


def _expert_check(
    src: Checkpoint,
    dst: Checkpoint,
    layer: int,
    expert: int,
    *,
    dense_strategy: str,
    tokens: int,
    compute_dtype: torch.dtype,
    seed: int,
) -> CheckResult:
    names = {
        part: f"layers.{layer}.ffn.experts.{expert}.{part}.weight"
        for part in ("w1", "w2", "w3")
    }
    torch.manual_seed(seed)
    src_w1 = _to_compute_dtype(_source_dequant(src, names["w1"]), compute_dtype)
    src_w2 = _to_compute_dtype(_source_dequant(src, names["w2"]), compute_dtype)
    src_w3 = _to_compute_dtype(_source_dequant(src, names["w3"]), compute_dtype)
    dst_w1 = _to_compute_dtype(
        _converted_dequant(dst, names["w1"], dense_strategy), compute_dtype
    )
    dst_w2 = _to_compute_dtype(
        _converted_dequant(dst, names["w2"], dense_strategy), compute_dtype
    )
    dst_w3 = _to_compute_dtype(
        _converted_dequant(dst, names["w3"], dense_strategy), compute_dtype
    )
    x = torch.randn(tokens, src_w1.shape[1], device=src_w1.device, dtype=compute_dtype)
    src_hidden = F.silu(F.linear(x, src_w1)) * F.linear(x, src_w3)
    dst_hidden = F.silu(F.linear(x, dst_w1)) * F.linear(x, dst_w3)
    src_out = F.linear(src_hidden, src_w2)
    dst_out = F.linear(dst_hidden, dst_w2)
    return CheckResult(
        check="expert_mlp",
        name=f"layers.{layer}.ffn.experts.{expert}",
        shape=tuple(src_out.shape),
        metric=_metric(src_out, dst_out),
        details={
            "tokens": tokens,
            "w1_shape": tuple(src_w1.shape),
            "w2_shape": tuple(src_w2.shape),
            "w3_shape": tuple(src_w3.shape),
            "compute_dtype": str(compute_dtype),
        },
    )


def _quant_dequant_sym_last(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    grouped = weight.float().reshape(*weight.shape[:-1], -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = grouped.abs().amax(dim=-1).clamp(
        min=torch.finfo(torch.float32).tiny
    ) / qmax
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(qmin, qmax)
    return (q * scale.unsqueeze(-1)).reshape_as(weight)


def _quant_dequant_asym_last(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    grouped = weight.float().reshape(*weight.shape[:-1], -1, group_size)
    qmax = (1 << bits) - 1
    group_min = torch.minimum(
        grouped.amin(dim=-1), torch.zeros((), device=grouped.device)
    )
    group_max = torch.maximum(
        grouped.amax(dim=-1), torch.zeros((), device=grouped.device)
    )
    scale = (group_max - group_min).clamp(min=torch.finfo(torch.float32).tiny) / qmax
    zero = torch.round(-group_min / scale).clamp(0, qmax)
    q = torch.round(grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1)).clamp(0, qmax)
    return ((q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape_as(weight)


def _candidate_expert_check(
    src: Checkpoint,
    layer: int,
    expert: int,
    *,
    candidate: str,
    tokens: int,
    compute_dtype: torch.dtype,
    seed: int,
) -> CheckResult:
    if candidate == "sym_int4_g32":
        quant = lambda w: _quant_dequant_sym_last(w, bits=4, group_size=32)
    elif candidate.startswith("asym_uint") and candidate.endswith("_g32"):
        bits = int(candidate.removeprefix("asym_uint").removesuffix("_g32"))
        quant = lambda w: _quant_dequant_asym_last(w, bits=bits, group_size=32)
    else:
        raise ValueError(f"unsupported candidate {candidate}")

    names = {
        part: f"layers.{layer}.ffn.experts.{expert}.{part}.weight"
        for part in ("w1", "w2", "w3")
    }
    torch.manual_seed(seed)
    src_w1_f32 = _source_dequant(src, names["w1"])
    src_w2_f32 = _source_dequant(src, names["w2"])
    src_w3_f32 = _source_dequant(src, names["w3"])
    src_w1 = _to_compute_dtype(src_w1_f32, compute_dtype)
    src_w2 = _to_compute_dtype(src_w2_f32, compute_dtype)
    src_w3 = _to_compute_dtype(src_w3_f32, compute_dtype)
    dst_w1 = _to_compute_dtype(quant(src_w1_f32), compute_dtype)
    dst_w2 = _to_compute_dtype(quant(src_w2_f32), compute_dtype)
    dst_w3 = _to_compute_dtype(quant(src_w3_f32), compute_dtype)
    x = torch.randn(tokens, src_w1.shape[1], device=src_w1.device, dtype=compute_dtype)
    src_hidden = F.silu(F.linear(x, src_w1)) * F.linear(x, src_w3)
    dst_hidden = F.silu(F.linear(x, dst_w1)) * F.linear(x, dst_w3)
    src_out = F.linear(src_hidden, src_w2)
    dst_out = F.linear(dst_hidden, dst_w2)
    return CheckResult(
        check="expert_candidate",
        name=f"layers.{layer}.ffn.experts.{expert}:{candidate}",
        shape=tuple(src_out.shape),
        metric=_metric(src_out, dst_out),
        details={"tokens": tokens, "candidate": candidate},
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _default_tensor_names(layers: list[int], experts: list[int]) -> list[str]:
    names: list[str] = []
    for layer in layers:
        names.extend(
            [
                f"layers.{layer}.attn.wq_a.weight",
                f"layers.{layer}.attn.wq_b.weight",
                f"layers.{layer}.attn.wkv.weight",
                f"layers.{layer}.attn.wo_a.weight",
                f"layers.{layer}.attn.wo_b.weight",
                f"layers.{layer}.attn.indexer.wq_b.weight",
                f"layers.{layer}.attn.indexer.compressor.wkv.weight",
                f"layers.{layer}.ffn.shared_experts.w1.weight",
                f"layers.{layer}.ffn.shared_experts.w2.weight",
                f"layers.{layer}.ffn.shared_experts.w3.weight",
            ]
        )
        for expert in experts:
            names.extend(
                [
                    f"layers.{layer}.ffn.experts.{expert}.w1.weight",
                    f"layers.{layer}.ffn.experts.{expert}.w2.weight",
                    f"layers.{layer}.ffn.experts.{expert}.w3.weight",
                ]
            )
    return names


def _print_result(result: CheckResult) -> None:
    metric = result.metric
    print(
        f"{result.check:10s} {result.name:68s} "
        f"snr={metric.snr_db:7.2f}dB rel_l2={metric.rel_l2:.4g} "
        f"rmse={metric.rmse:.4g} max_abs={metric.max_abs:.4g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--layers", default="1")
    parser.add_argument("--experts", default="0,1,2")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dense-strategy", choices=("block", "channel"), default="channel")
    parser.add_argument(
        "--preserved-names",
        default="embed.weight,head.weight",
        help="Comma-separated preserved tensors to compare exactly or near-exactly.",
    )
    parser.add_argument(
        "--expert-candidates",
        default="",
        help=(
            "Comma-separated hypothetical source-side expert formats to test, "
            "for example sym_int4_g32,asym_uint4_g32,asym_uint5_g32."
        ),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.compute_dtype]
    src = Checkpoint(args.src.resolve(), args.device)
    dst = Checkpoint(args.dst.resolve(), args.device)
    layers = _parse_int_list(args.layers)
    experts = _parse_int_list(args.experts)
    expert_candidates = [item for item in args.expert_candidates.split(",") if item]
    preserved_names = [item for item in args.preserved_names.split(",") if item]

    results: list[CheckResult] = []
    missing: list[str] = []
    for name in preserved_names:
        if not src.has(name) or not dst.has(name):
            missing.append(name)
            continue
        results.append(
            _tensor_check(
                src,
                dst,
                name,
                dense_strategy=args.dense_strategy,
                max_rows=args.max_rows,
            )
        )
        if name.endswith("head.weight"):
            results.append(
                _linear_check(
                    src,
                    dst,
                    name,
                    dense_strategy=args.dense_strategy,
                    tokens=args.tokens,
                    compute_dtype=dtype,
                    seed=args.seed,
                )
            )

    for name in _default_tensor_names(layers, experts):
        if not src.has(name) or not dst.has(name):
            missing.append(name)
            continue
        results.append(
            _tensor_check(
                src,
                dst,
                name,
                dense_strategy=args.dense_strategy,
                max_rows=args.max_rows,
            )
        )
        results.append(
            _linear_check(
                src,
                dst,
                name,
                dense_strategy=args.dense_strategy,
                tokens=args.tokens,
                compute_dtype=dtype,
                seed=args.seed,
            )
        )

    for layer in layers:
        for expert in experts:
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            if not all(src.has(f"{prefix}.{part}.weight") for part in ("w1", "w2", "w3")):
                missing.append(prefix)
                continue
            results.append(
                _expert_check(
                    src,
                    dst,
                    layer,
                    expert,
                    dense_strategy=args.dense_strategy,
                    tokens=args.tokens,
                    compute_dtype=dtype,
                    seed=args.seed,
                )
            )
            for candidate in expert_candidates:
                results.append(
                    _candidate_expert_check(
                        src,
                        layer,
                        expert,
                        candidate=candidate,
                        tokens=args.tokens,
                        compute_dtype=dtype,
                        seed=args.seed,
                    )
                )

    for result in results:
        _print_result(result)
    if missing:
        print("\nmissing:")
        for name in missing:
            print(f"  {name}")

    report = {
        "src": str(src.path),
        "dst": str(dst.path),
        "device": args.device,
        "dense_strategy": args.dense_strategy,
        "compute_dtype": args.compute_dtype,
        "missing": missing,
        "results": [asdict(result) for result in results],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    bad = [
        result
        for result in results
        if result.check in {"linear", "expert_mlp"} and result.metric.snr_db < 15.0
    ]
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
