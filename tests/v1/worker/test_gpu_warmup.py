# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu.warmup import warmup_kernels


class _KVConnector:

    def __init__(self):
        self.disabled_states = []

    def set_disabled(self, disabled: bool) -> None:
        self.disabled_states.append(disabled)


def test_warmup_kernels_defaults_missing_spec_decode_attr(monkeypatch):
    connector = _KVConnector()
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=32,
            max_num_batched_tokens=1024,
        ),
        is_pooling_model=False,
        is_last_pp_rank=False,
        kv_connector=connector,
    )
    executed = []
    sampled = []

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.warmup.torch.accelerator.synchronize",
        lambda: None,
    )

    warmup_kernels(
        runner,
        executed.append,
        sampled.append,
        prompt_len=52,
        num_reqs_limit=12,
    )

    assert connector.disabled_states == [True, False]
    assert len(executed) == 3
    assert executed[0].total_num_scheduled_tokens == 624
    assert len(executed[0].scheduled_new_reqs) == 12
    assert executed[1].total_num_scheduled_tokens == 12
    assert executed[2].finished_req_ids == {f"_warmup_{i}_" for i in range(12)}
    assert sampled == [None, None]
