# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.mtp import (
    DeepSeekV4MTP,
    DeepSeekV4MultiTokenPredictor,
)
from vllm.model_executor.models.interfaces import supports_pp


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
