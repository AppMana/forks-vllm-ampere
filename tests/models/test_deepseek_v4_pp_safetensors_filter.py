# SPDX-License-Identifier: Apache-2.0

import json

from vllm.models.deepseek_v4.nvidia import model as deepseek_v4_model
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model


class _FakePPGroup:
    def __init__(self, rank: int, world_size: int) -> None:
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1


def _model_for_rank(monkeypatch, rank: int) -> DeepseekV4Model:
    monkeypatch.setattr(
        deepseek_v4_model,
        "get_pp_group",
        lambda: _FakePPGroup(rank, 3),
    )
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    model.start_layer = [0, 2, 4][rank]
    model.end_layer = [2, 4, 6][rank]
    return model


def test_deepseek_v4_pp_safetensors_filter_uses_index(tmp_path, monkeypatch):
    weight_map = {
        "embed.weight": "model-00001-of-00005.safetensors",
        "layers.0.attn.wq.weight": "model-00001-of-00005.safetensors",
        "layers.1.ffn.w1.weight": "model-00002-of-00005.safetensors",
        "layers.2.attn.wq.weight": "model-00002-of-00005.safetensors",
        "layers.3.ffn.w1.weight": "model-00003-of-00005.safetensors",
        "layers.4.attn.wq.weight": "model-00004-of-00005.safetensors",
        "norm.weight": "model-00005-of-00005.safetensors",
        "head.weight": "model-00005-of-00005.safetensors",
        "mtp.0.ffn.w1.weight": "model-00005-of-00005.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    files = [
        str(tmp_path / f"model-0000{i}-of-00005.safetensors") for i in range(1, 6)
    ]

    first_rank = _model_for_rank(monkeypatch, 0)
    assert [
        p.rsplit("/", 1)[-1]
        for p in first_rank.filter_safetensors_files_for_current_rank(
            str(tmp_path), files
        )
    ] == [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
    ]

    middle_rank = _model_for_rank(monkeypatch, 1)
    assert [
        p.rsplit("/", 1)[-1]
        for p in middle_rank.filter_safetensors_files_for_current_rank(
            str(tmp_path), files
        )
    ] == [
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
    ]

    last_rank = _model_for_rank(monkeypatch, 2)
    assert [
        p.rsplit("/", 1)[-1]
        for p in last_rank.filter_safetensors_files_for_current_rank(str(tmp_path), files)
    ] == [
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]


def test_deepseek_v4_pp_safetensors_filter_falls_back_without_index(
    tmp_path, monkeypatch
):
    model = _model_for_rank(monkeypatch, 1)
    files = [str(tmp_path / "model-00001-of-00001.safetensors")]

    assert model.filter_safetensors_files_for_current_rank(str(tmp_path), files) is files
