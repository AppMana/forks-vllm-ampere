# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker import gpu_worker


def test_deepseek_v4_sparse_mla_warmup_predicate(monkeypatch):
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v4")
        )
    )

    monkeypatch.setattr(
        gpu_worker.envs, "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", True
    )
    assert gpu_worker._uses_deepseek_v4_sparse_mla_warmup(worker)

    monkeypatch.setattr(
        gpu_worker.envs, "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", False
    )
    assert not gpu_worker._uses_deepseek_v4_sparse_mla_warmup(worker)

    worker.model_config.hf_config.model_type = "llama"
    monkeypatch.setattr(
        gpu_worker.envs, "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", True
    )
    assert not gpu_worker._uses_deepseek_v4_sparse_mla_warmup(worker)
