# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as speculator_mod
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
    EagleSpeculator,
    prepare_eagle_inputs,
)


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


def test_dsv4_mtp_speculator_enforce_eager_disables_cudagraph_managers(monkeypatch):
    modes = []

    class FakeCudaGraphManager:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            del vllm_config, device, decode_query_len
            self.pool = None
            modes.append(cudagraph_mode)

    speculator = EagleSpeculator.__new__(EagleSpeculator)
    speculator.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    )
    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 1

    monkeypatch.setattr(
        speculator_mod, "PrefillEagleCudaGraphManager", FakeCudaGraphManager
    )
    monkeypatch.setattr(
        speculator_mod, "DecodeEagleCudaGraphManager", FakeCudaGraphManager
    )

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)

    assert modes == [CUDAGraphMode.NONE, CUDAGraphMode.NONE]


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


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_prepare_eagle_inputs_clamps_all_rejected_query_len():
    device = torch.device("cuda")
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_tokens_after_padding=1,
        num_scheduled_tokens=np.array([1], dtype=np.int32),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        query_start_loc_np=np.array([0, 1], dtype=np.int32),
        seq_lens=torch.ones(1, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.ones(1, dtype=torch.int32, device=device),
        dcp_local_seq_lens=None,
        input_ids=torch.tensor([100], dtype=torch.int32, device=device),
        positions=torch.zeros(1, dtype=torch.int64, device=device),
    )
    buffers = InputBuffers(4, 1, device)
    last_token_indices = torch.full((4,), -777, dtype=torch.int64, device=device)
    current_draft_step = torch.tensor(1, dtype=torch.int64, device=device)
    last_sampled = torch.full((4, 1), 4242, dtype=torch.int64, device=device)
    next_prefill_tokens = torch.full((4,), 3131, dtype=torch.int32, device=device)

    prepare_eagle_inputs(
        last_token_indices,
        current_draft_step,
        buffers,
        input_batch,
        torch.ones(1, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        4,
        shift_positions=True,
    )
    torch.cuda.synchronize()

    assert last_token_indices[0].item() == 0
    assert buffers.input_ids[0].item() == 4242
    assert buffers.positions[0].item() == 1
    assert current_draft_step.item() == 0
