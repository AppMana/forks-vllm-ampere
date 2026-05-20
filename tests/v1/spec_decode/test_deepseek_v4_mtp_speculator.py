# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as speculator_mod
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


class _DummyBlockTables:
    def __init__(self) -> None:
        self.slot_mappings = torch.zeros((1, 8), dtype=torch.int64)
        self.computed = 0
        self.gathered = 0

    def compute_slot_mappings(self, *args, **kwargs):
        self.computed += 1
        return self.slot_mappings[:, : args[3]]

    def gather_block_tables(self, *args, **kwargs):
        self.gathered += 1
        return (torch.zeros((1, 1), dtype=torch.int32),)


def _make_speculator(with_block_tables: bool) -> EagleSpeculator:
    speculator = EagleSpeculator.__new__(EagleSpeculator)
    speculator.deepseek_v4_mtp_positions_follow_shift = True
    speculator.hidden_states = torch.zeros((8, 6), dtype=torch.float32)
    speculator.temperature = torch.zeros(2, dtype=torch.float32)
    speculator.seeds = torch.zeros(2, dtype=torch.int64)
    speculator.idx_mapping = torch.zeros(2, dtype=torch.int32)
    speculator.last_token_indices = torch.zeros(2, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0, dtype=torch.int64)
    speculator.input_buffers = InputBuffers(2, 8, torch.device("cpu"))
    speculator.max_num_reqs = 2
    speculator.max_num_tokens = 8
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.prefill_cudagraph_manager = None
    speculator.num_speculative_steps = 1
    speculator.draft_tokens = torch.zeros((2, 1), dtype=torch.int64)
    speculator.prefill = lambda *args, **kwargs: None
    speculator._build_draft_prefill_attn_metadata = lambda **kwargs: {"rebuilt": True}
    if with_block_tables:
        speculator.block_tables = _DummyBlockTables()
        speculator.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    return speculator


def _make_input_batch():
    return SimpleNamespace(
        num_tokens_after_padding=4,
        num_tokens=4,
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        idx_mapping=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_start_loc_np=np.array([0, 4], dtype=np.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([4], dtype=torch.int32),
        dcp_local_seq_lens=None,
    )


def _propose_once(speculator: EagleSpeculator, input_batch) -> None:
    speculator.propose(
        input_batch,
        {"target": "metadata"},
        {"target": torch.zeros(4)},
        torch.ones((4, 6)),
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.int64),
    )


def test_dsv4_mtp_speculator_skips_shifted_metadata_during_profile():
    calls: list[bool | None] = []
    input_batch = _make_input_batch()

    def fake_prepare(*args, **kwargs):
        calls.append(kwargs.get("shift_positions"))
        return args[0]

    with (
        mock.patch.object(speculator_mod, "prepare_eagle_inputs", fake_prepare),
        mock.patch.object(speculator_mod, "get_uniform_token_count", lambda *a, **k: None),
        mock.patch.object(
            speculator_mod,
            "dispatch_cg_and_sync_dp",
            lambda *a, **k: (BatchExecutionDescriptor(CUDAGraphMode.NONE, 4, None), None),
        ),
    ):
        _propose_once(_make_speculator(with_block_tables=False), input_batch)

    assert calls == [False]


def test_dsv4_mtp_speculator_rebuilds_shifted_draft_metadata():
    calls: list[bool | None] = []
    input_batch = _make_input_batch()
    speculator = _make_speculator(with_block_tables=True)

    def fake_prepare(*args, **kwargs):
        calls.append(kwargs.get("shift_positions"))
        return args[0]

    with (
        mock.patch.object(speculator_mod, "prepare_eagle_inputs", fake_prepare),
        mock.patch.object(speculator_mod, "get_uniform_token_count", lambda *a, **k: None),
        mock.patch.object(
            speculator_mod,
            "dispatch_cg_and_sync_dp",
            lambda *a, **k: (BatchExecutionDescriptor(CUDAGraphMode.NONE, 4, None), None),
        ),
        mock.patch.object(
            speculator_mod,
            "build_slot_mappings_by_layer",
            lambda slot_mappings, kv_cache_config: {"layer": slot_mappings},
        ),
    ):
        _propose_once(speculator, input_batch)

    assert calls == [True]
    assert speculator.block_tables.computed == 1
