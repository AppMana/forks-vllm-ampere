# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.mtp import (
    DeepSeekV4MTP,
    DeepSeekV4MultiTokenPredictor,
)
from vllm.model_executor.models.interfaces import supports_pp
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


def test_deepseek_v4_declares_pipeline_parallel_support():
    assert supports_pp(DeepseekV4ForCausalLM)


def test_deepseek_v4_mtp_declares_pipeline_parallel_support():
    assert supports_pp(DeepSeekV4MTP)


def test_deepseek_v4_mtp_bind_lm_head_replaces_layer_heads():
    predictor = DeepSeekV4MultiTokenPredictor.__new__(DeepSeekV4MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)

    old_head = torch.nn.Linear(4, 4, bias=False)
    new_head = torch.nn.Linear(4, 4, bias=False)
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = old_head
    predictor.layers = torch.nn.ModuleDict({"43": layer})
    predictor._target_lm_head = None

    predictor.bind_lm_head(new_head)

    assert predictor._target_lm_head is new_head
    assert predictor.layers["43"].shared_head.head is new_head


def test_deepseek_v4_mtp_first_pass_positions_follow_shift():
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    proposer.needs_extra_input_slots = False
    proposer.deepseek_v4_mtp_positions_follow_shift = True
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.input_ids = torch.zeros(4, dtype=torch.int32)
    proposer.hidden_states = torch.zeros(4, 3)
    proposer.positions = torch.zeros(4, dtype=torch.int64)
    proposer.vllm_config = type(
        "FakeVllmConfig",
        (),
        {"model_config": type("FakeModelConfig", (), {"uses_mrope": False})()},
    )()

    class Cad:
        query_start_loc = torch.tensor([0, 4], dtype=torch.int32)

    target_hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    num_tokens, token_indices, _ = proposer.set_inputs_first_pass(
        target_token_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        next_token_ids=torch.tensor([14], dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=Cad(),
        num_rejected_tokens_gpu=None,
    )

    assert num_tokens == 4
    assert token_indices.tolist() == [3]
    assert proposer.input_ids.tolist() == [11, 12, 13, 14]
    assert proposer.positions.tolist() == [1, 2, 3, 4]
    assert torch.equal(proposer.hidden_states, target_hidden_states)
