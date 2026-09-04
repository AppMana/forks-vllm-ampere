# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifetime contract for tensors handed between the main and aux CUDA streams.

Events and ``wait_stream`` order two streams; they do not tell the caching
allocator that a second stream consumes the tensor. A tensor allocated on
stream A and used on stream B needs ``record_stream(B)`` or its block can be
handed to another allocation while B still reads it. The corruption is a few
rows, only under concurrency, and presents as run-to-run nondeterminism rather
than a fault -- which is why it is pinned by contract here instead of by
trying to race an allocator in a test.

CPU-only: ``record_stream`` is patched out, so no CUDA context is needed.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.utils import multi_stream_utils

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def recorded(monkeypatch):
    """Collect (tensor, stream) pairs passed to Tensor.record_stream."""
    calls: list[tuple[torch.Tensor, object]] = []

    def _record(self, stream):
        calls.append((self, stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", _record, raising=False)
    return calls


def test_record_stream_walks_nested_results(recorded):
    """Aux callables return tuples and dicts, not bare tensors."""
    stream = object()
    a, b, c = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)

    multi_stream_utils._record_stream((a, [b, None], {"k": c}), stream)

    assert [t for t, _ in recorded] == [a, b, c]
    assert {s for _, s in recorded} == {stream}
    # Non-tensors must not raise.
    multi_stream_utils._record_stream(None, stream)
    multi_stream_utils._record_stream("not a tensor", stream)


def _patch_cuda(monkeypatch, main_stream):
    """Make the aux-stream paths runnable without a CUDA context."""
    import contextlib

    monkeypatch.setattr(
        torch.cuda, "stream", lambda s: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: main_stream, raising=False
    )


def test_maybe_execute_in_parallel_records_aux_results(monkeypatch, recorded):
    main_stream = object()
    _patch_cuda(monkeypatch, main_stream)
    aux_out = torch.zeros(4)

    _, result1 = multi_stream_utils.maybe_execute_in_parallel(
        fn0=lambda: torch.ones(4),
        fn1=lambda: aux_out,
        event0=MagicMock(),
        event1=MagicMock(),
        aux_stream=MagicMock(),
    )

    assert result1 is aux_out
    assert (aux_out, main_stream) in recorded


def test_execute_in_parallel_records_every_aux_result(monkeypatch, recorded):
    main_stream = object()
    _patch_cuda(monkeypatch, main_stream)
    first, second = torch.zeros(2), torch.ones(2)

    _, aux_results = multi_stream_utils.execute_in_parallel(
        default_fn=lambda: None,
        aux_fns=[lambda: first, None, lambda: second],
        start_event=MagicMock(),
        done_events=[MagicMock(), MagicMock(), MagicMock()],
        aux_streams=[MagicMock(), MagicMock(), MagicMock()],
        enable=True,
    )

    assert aux_results == [first, None, second]
    # Identity, not ``in``: ``==`` on tensors yields a tensor.
    recorded_ids = {id(t) for t, _ in recorded}
    assert id(first) in recorded_ids and id(second) in recorded_ids


def test_sequential_fallback_does_not_record(monkeypatch, recorded):
    """With no aux stream nothing crosses streams, so nothing is recorded."""
    _patch_cuda(monkeypatch, object())
    multi_stream_utils.maybe_execute_in_parallel(
        fn0=lambda: torch.zeros(1),
        fn1=lambda: torch.zeros(1),
        event0=MagicMock(),
        event1=MagicMock(),
        aux_stream=None,
    )
    assert recorded == []


def _patch_shared_experts_stream(monkeypatch, main_stream):
    """shared_experts imports current_stream from vllm.utils.torch_utils, so
    patching torch.cuda.current_stream would miss it."""
    import contextlib

    from vllm.model_executor.layers.fused_moe.runner import shared_experts as mod

    monkeypatch.setattr(mod, "current_stream", lambda: main_stream, raising=False)
    monkeypatch.setattr(
        torch.cuda, "stream", lambda s: contextlib.nullcontext(), raising=False
    )


def _bare_shared_experts(aux_stream, layer):
    """A SharedExperts without running __init__, which needs a full MoE config."""
    runner = object.__new__(SharedExperts)
    runner._stream = aux_stream
    runner._layer = layer
    runner._synced_input = None
    return runner


def test_run_in_aux_stream_requires_the_sync_that_records_the_input(
    monkeypatch, recorded
):
    """Skipping maybe_sync_shared_experts_stream must fail loudly.

    That call both record_streams the input against the aux stream and orders
    the aux stream after the producer; without it the aux stream can read the
    input before it is written, and the allocator can reuse the input's block
    underneath it.
    """
    # MagicMock, not a sentinel: the code calls main_stream.wait_stream().
    main_stream = MagicMock()
    _patch_shared_experts_stream(monkeypatch, main_stream)
    aux_stream = MagicMock()
    hidden = torch.zeros(8)
    runner = _bare_shared_experts(aux_stream, lambda x: torch.ones(8))

    with pytest.raises(AssertionError, match="maybe_sync_shared_experts_stream"):
        runner._run_in_aux_stream(hidden)

    # Synced with a different tensor is still a violation.
    runner._synced_input = torch.zeros(8)
    with pytest.raises(AssertionError, match="maybe_sync_shared_experts_stream"):
        runner._run_in_aux_stream(hidden)


def test_run_in_aux_stream_records_its_output_on_the_main_stream(
    monkeypatch, recorded
):
    # MagicMock, not a sentinel: the code calls main_stream.wait_stream().
    main_stream = MagicMock()
    _patch_shared_experts_stream(monkeypatch, main_stream)
    aux_stream = MagicMock()
    hidden = torch.zeros(8)
    produced = torch.ones(8)
    runner = _bare_shared_experts(aux_stream, lambda x: produced)
    runner._synced_input = hidden

    output = runner._run_in_aux_stream(hidden)

    assert output is produced
    assert any(t is produced and s is main_stream for t, s in recorded)
    # The aux stream must also be joined before the output is used.
    main_stream.wait_stream.assert_called_once_with(aux_stream)
    # The guard is one-shot: the next call must sync again.
    assert runner._synced_input is None
