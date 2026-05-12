#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert DeepSeek V4 FP4/FP8 checkpoint shards to dsv4_int.

This is the conservative Ampere baseline converter:

* routed expert MXFP4 weights -> symmetric INT4 W4A16, group size 32
* FP8 linears -> symmetric INT8 W8A16, 128x128 blocks by default, or
  channelwise biased UINT8 for the AllSpark Ampere W8A16 kernel
* BF16/F32/etc. tensors -> passthrough

The converter preserves tensor names and shard names so the original
``model.safetensors.index.json`` remains valid.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dsv4_checkpoint_audit import classify_tensor, matched_scale_name  # noqa: E402

from vllm.model_executor.layers.quantization.dsv4_int import (  # noqa: E402
    requantize_fp8_to_allspark_uint8_w8a16,
    requantize_fp8_to_int8_w8a16,
    requantize_mxfp4_to_int4_w4a16,
)

_FP8_WEIGHT_ROLES = {
    "dense_fp8_weight",
    "indexer_qk_fp8_weight",
    "mtp_fp8_weight",
}
_LAYER_NAME_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def _log(message: str) -> None:
    print(f"[dsv4-requant] {message}", flush=True)


def _remap_tensor_name(name: str, layer_remap: dict[int, int] | None) -> str | None:
    if layer_remap is None:
        return name
    match = _LAYER_NAME_RE.match(name)
    if match is None:
        return name
    source_idx = int(match.group(1))
    if source_idx not in layer_remap:
        return None
    return f"layers.{layer_remap[source_idx]}.{match.group(2)}"


def _discover_layer_remap(src: Path) -> dict[int, int] | None:
    cfg = json.loads((src / "config.json").read_text())
    expected_layers = int(cfg.get("num_hidden_layers", 0))
    if expected_layers <= 0:
        return None

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    weight_map = json.loads(index_path.read_text())["weight_map"]
    layer_ids = sorted(
        {
            int(match.group(1))
            for name in weight_map
            if (match := _LAYER_NAME_RE.match(name)) is not None
        }
    )
    if layer_ids == list(range(expected_layers)):
        return None
    if len(layer_ids) != expected_layers:
        raise ValueError(
            f"cannot auto-remap {layer_ids=} to {expected_layers=} layers"
        )
    return {source: target for target, source in enumerate(layer_ids)}


def _copy_metadata(src: Path, dst: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        src_path = src / name
        if src_path.exists():
            shutil.copy(src_path.resolve(), dst / name)


def _write_index(src: Path, dst: Path, layer_remap: dict[int, int] | None) -> None:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    remapped_weight_map = {}
    for name, shard in index["weight_map"].items():
        remapped = _remap_tensor_name(name, layer_remap)
        if remapped is not None:
            remapped_weight_map[remapped] = shard
    index["weight_map"] = remapped_weight_map
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def _write_config(
    src: Path,
    dst: Path,
    layer_remap: dict[int, int] | None,
    *,
    dense_int8_strategy: str,
    expert_int4_scale_mode: str,
) -> None:
    cfg = json.loads((src / "config.json").read_text())
    if layer_remap is not None:
        cfg["num_hidden_layers"] = len(layer_remap)
    dense_weights_cfg: dict[str, object] = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": dense_int8_strategy,
    }
    if dense_int8_strategy == "block":
        dense_weights_cfg["block_size"] = [128, 128]
    cfg["expert_dtype"] = "int4"
    cfg["quantization_config"] = {
        "quant_method": "dsv4_int",
        "format": "int_packed",
        "config_groups": {
            "experts_w4a16": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 32,
                    "strategy": "group",
                    "scale_mode": expert_int4_scale_mode,
                },
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.ffn.experts.*.w1",
                    "*.ffn.experts.*.w2",
                    "*.ffn.experts.*.w3",
                ],
            },
            "linears_w8a16": {
                "weights": dense_weights_cfg,
                "input_activations": {"num_bits": 16, "type": "float"},
                "targets": [
                    "*.attn.wq_a",
                    "*.attn.wq_b",
                    "*.attn.wkv",
                    "*.attn.wo_a",
                    "*.attn.wo_b",
                    "*.attn.indexer.wq_b",
                    "*.attn.indexer.compressor.wkv",
                    "*.attn.indexer.compressor.wgate",
                    "*.ffn.shared_experts.w1",
                    "*.ffn.shared_experts.w2",
                    "*.ffn.shared_experts.w3",
                    "mtp.*.e_proj",
                    "mtp.*.h_proj",
                ],
            },
        },
        "ignore": [
            "embed",
            "head",
            "norm",
            "lm_head",
            "*norm.weight",
            "attn.attn_sink",
            "*.gate.*",
            "hc_*",
            "*.hc_attn_*",
            "*.hc_ffn_*",
        ],
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def _classify_shard(
    src_shard: Path,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    roles: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    missing_scales: set[str] = set()
    with safe_open(src_shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name in sorted(keys):
            tensor_slice = handle.get_slice(name)
            dtype = tensor_slice.get_dtype()
            role, action = classify_tensor(name, dtype)
            if role == "unknown":
                raise ValueError(f"unknown tensor in {src_shard.name}: {name} {dtype}")
            roles[name] = role
            dtypes[name] = dtype
            scale_name = matched_scale_name(name)
            if (
                scale_name is not None
                and action != "preserve"
                and scale_name not in keys
            ):
                missing_scales.add(name)
    return roles, dtypes, missing_scales


def convert_shard(
    src_shard: Path,
    dst_shard: Path,
    *,
    device: str,
    out_scale_dtype: torch.dtype,
    layer_remap: dict[int, int] | None,
    dense_int8_strategy: str,
    expert_int4_scale_mode: str,
) -> dict[str, int]:
    roles, _dtypes, missing_scales = _classify_shard(src_shard)
    if missing_scales:
        sample = ", ".join(sorted(missing_scales)[:8])
        raise ValueError(f"{src_shard.name} missing scales for: {sample}")

    out: dict[str, torch.Tensor] = {}
    counts = {"int4": 0, "int8": 0, "preserve": 0}
    paired_scales = {
        matched_scale_name(name)
        for name, role in roles.items()
        if role == "routed_expert_mxfp4_weight" or role in _FP8_WEIGHT_ROLES
    }
    paired_scales.discard(None)

    with safe_open(src_shard, framework="pt", device=device) as handle:
        for name in sorted(handle.keys()):
            if name in paired_scales:
                continue
            role = roles[name]
            out_name = _remap_tensor_name(name, layer_remap)
            if out_name is None:
                continue
            if role == "routed_expert_mxfp4_weight":
                scale_name = matched_scale_name(name)
                assert scale_name is not None
                out_scale_name = _remap_tensor_name(scale_name, layer_remap)
                assert out_scale_name is not None
                converted = requantize_mxfp4_to_int4_w4a16(
                    handle.get_tensor(name),
                    handle.get_tensor(scale_name),
                    scale_mode=expert_int4_scale_mode,
                    out_scale_dtype=out_scale_dtype,
                )
                out[out_name] = converted["qweight_packed"].cpu()
                out[out_scale_name] = converted["scales"].cpu()
                counts["int4"] += 1
            elif role in _FP8_WEIGHT_ROLES:
                scale_name = matched_scale_name(name)
                assert scale_name is not None
                out_scale_name = _remap_tensor_name(scale_name, layer_remap)
                assert out_scale_name is not None
                if dense_int8_strategy == "channel":
                    converted = requantize_fp8_to_allspark_uint8_w8a16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                        out_scale_dtype=out_scale_dtype,
                    )
                else:
                    converted = requantize_fp8_to_int8_w8a16(
                        handle.get_tensor(name),
                        handle.get_tensor(scale_name),
                        out_scale_dtype=out_scale_dtype,
                    )
                out[out_name] = converted["qweight"].cpu()
                out[out_scale_name] = converted["scales"].cpu()
                counts["int8"] += 1
            elif role.endswith("_scale"):
                raise ValueError(f"unpaired scale tensor in {src_shard.name}: {name}")
            else:
                out[out_name] = handle.get_tensor(name).cpu()
                counts["preserve"] += 1

    save_file(out, str(dst_shard))
    return counts


def convert_checkpoint(
    src: Path,
    dst: Path,
    *,
    device: str,
    out_scale_dtype: torch.dtype,
    overwrite: bool,
    layer_remap: dict[int, int] | None,
    dense_int8_strategy: str = "block",
    expert_int4_scale_mode: str = "absmax7",
) -> None:
    if dense_int8_strategy not in ("block", "channel"):
        raise ValueError(
            f"dense_int8_strategy must be 'block' or 'channel', got "
            f"{dense_int8_strategy!r}"
        )
    if expert_int4_scale_mode not in ("absmax7", "absmax8"):
        raise ValueError(
            "expert_int4_scale_mode must be 'absmax7' or 'absmax8', got "
            f"{expert_int4_scale_mode!r}"
        )
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{dst} exists and is not empty; pass --overwrite")
        for child in dst.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {src}")
    if layer_remap is None:
        layer_remap = _discover_layer_remap(src)
    if layer_remap is not None:
        _log(f"layer_remap={layer_remap}")

    totals = {"int4": 0, "int8": 0, "preserve": 0}
    _log(f"converting {len(shards)} shards from {src} to {dst}")
    for shard in shards:
        _log(f"-> {shard.name}")
        counts = convert_shard(
            shard,
            dst / shard.name,
            device=device,
            out_scale_dtype=out_scale_dtype,
            layer_remap=layer_remap,
            dense_int8_strategy=dense_int8_strategy,
            expert_int4_scale_mode=expert_int4_scale_mode,
        )
        for key, value in counts.items():
            totals[key] += value
        _log(
            f"{shard.name}: int4={counts['int4']} int8={counts['int8']} "
            f"preserve={counts['preserve']}"
        )

    _copy_metadata(src, dst)
    _write_index(src, dst, layer_remap)
    _write_config(
        src,
        dst,
        layer_remap,
        dense_int8_strategy=dense_int8_strategy,
        expert_int4_scale_mode=expert_int4_scale_mode,
    )
    _log(
        f"done: int4={totals['int4']} int8={totals['int8']} "
        f"preserve={totals['preserve']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scale-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--dense-int8-strategy",
        choices=("block", "channel"),
        default="block",
        help="Use 128x128 block INT8 fallback format or channelwise AllSpark "
        "biased UINT8 format for FP8 dense linears.",
    )
    parser.add_argument(
        "--expert-int4-scale-mode",
        choices=("absmax7", "absmax8"),
        default="absmax7",
        help="Scale selection for MXFP4 routed experts converted to signed INT4.",
    )
    parser.add_argument(
        "--layer-remap",
        help="JSON mapping of source layer id to destination id. If omitted, "
        "truncated checkpoints are auto-remapped when possible.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    layer_remap = None
    if args.layer_remap:
        layer_remap = {int(k): int(v) for k, v in json.loads(args.layer_remap).items()}

    convert_checkpoint(
        args.src.resolve(),
        args.dst.resolve(),
        device=args.device,
        out_scale_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[
            args.scale_dtype
        ],
        overwrite=args.overwrite,
        layer_remap=layer_remap,
        dense_int8_strategy=args.dense_int8_strategy,
        expert_int4_scale_mode=args.expert_int4_scale_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
