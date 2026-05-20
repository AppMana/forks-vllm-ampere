#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize DeepSeek V4 MTP shared embed/head aliases in a checkpoint.

DeepSeek V4 stores MTP's token embedding and LM head as shared modules in its
reference inference code, not as separate ``mtp.*`` tensors. That is fine for a
single in-process model, but PP deployments that load the MTP draft separately
need explicit checkpoint names for the draft loader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _find_required_tensor(
    weight_map: dict[str, str], candidates: tuple[str, ...]
) -> str:
    for name in candidates:
        if name in weight_map:
            return name
    raise KeyError(f"none of {candidates!r} found in checkpoint")


def _copy_tensor(
    checkpoint: Path, weight_map: dict[str, str], name: str
) -> torch.Tensor:
    with safe_open(
        checkpoint / weight_map[name], framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(name)


def _load_index(checkpoint: Path) -> dict[str, object]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    return json.loads(index_path.read_text())


def materialize(checkpoint: Path, shard_name: str) -> list[str]:
    cfg = json.loads((checkpoint / "config.json").read_text())
    num_mtp_layers = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    if num_mtp_layers <= 0:
        return []

    index = _load_index(checkpoint)
    weight_map = dict(index["weight_map"])
    if not any(name.startswith("mtp.") for name in weight_map):
        return []

    embed_name = _find_required_tensor(
        weight_map,
        ("embed.weight", "model.embed.weight", "model.embed_tokens.weight"),
    )
    head_name = _find_required_tensor(
        weight_map,
        ("head.weight", "lm_head.weight", "model.head.weight"),
    )

    additions: dict[str, torch.Tensor] = {}
    for mtp_idx in range(num_mtp_layers):
        embed_alias = f"mtp.{mtp_idx}.emb.tok_emb.weight"
        head_alias = f"mtp.{mtp_idx}.head.weight"
        if embed_alias not in weight_map:
            additions[embed_alias] = _copy_tensor(checkpoint, weight_map, embed_name)
        if head_alias not in weight_map:
            additions[head_alias] = _copy_tensor(checkpoint, weight_map, head_name)

    if not additions:
        return []

    save_file(additions, str(checkpoint / shard_name))
    for name in additions:
        weight_map[name] = shard_name

    total_size = sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = str(total_size)
    index["metadata"]["mtp_shared_tensors"] = "materialized"
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return sorted(additions)


def upload(repo_id: str, checkpoint: Path, shard_name: str, *, revision: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(checkpoint / shard_name),
        path_in_repo=shard_name,
        repo_id=repo_id,
        revision=revision,
    )
    api.upload_file(
        path_or_fileobj=str(checkpoint / "model.safetensors.index.json"),
        path_in_repo="model.safetensors.index.json",
        repo_id=repo_id,
        revision=revision,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shard-name", default="model-mtp-shared.safetensors")
    parser.add_argument("--upload-repo")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    added = materialize(checkpoint, args.shard_name)
    if added:
        print("materialized: " + ", ".join(added), flush=True)
    else:
        print("nothing to materialize", flush=True)
    if args.upload_repo:
        upload(args.upload_repo, checkpoint, args.shard_name, revision=args.revision)
        print(
            f"uploaded {args.shard_name} and model.safetensors.index.json "
            f"to {args.upload_repo}@{args.revision}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
