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
    """A SharedExperts without running __init__, which needs a full MoE config.

    The order decision depends on the MoE config and platform; pin it to the
    overlapped path so the test exercises the aux-stream code, not the gate.
    """
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )

    runner = object.__new__(SharedExperts)
    runner._stream = aux_stream
    runner._layer = layer
    runner.enable_dbo = False
    runner._output = [None, None]
    runner._input_ready_event = [MagicMock(), MagicMock()]
    runner._output_ready_event = [MagicMock(), MagicMock()]
    runner._determine_shared_experts_order = (
        lambda hidden: SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    return runner


def test_maybe_forward_async_records_the_input_on_the_aux_stream(
    monkeypatch, recorded
):
    """The input is allocated on the main stream and read on the aux one.

    The input-ready event orders the aux stream after the producer; only
    record_stream stops the allocator handing the input's block to another
    tensor while the aux stream is still reading it.
    """
    main_stream = MagicMock()
    _patch_shared_experts_stream(monkeypatch, main_stream)
    aux_stream = MagicMock()
    hidden = torch.zeros(8)
    runner = _bare_shared_experts(aux_stream, lambda x: torch.ones(8))

    assert runner.maybe_forward_async(hidden)

    assert any(t is hidden and s is aux_stream for t, s in recorded)
    # Ordering must still be there: the aux stream waits on the input event.
    runner._input_ready_event[0].record.assert_called_once_with(main_stream)
    runner._input_ready_event[0].wait.assert_called_once_with(aux_stream)
    runner._output_ready_event[0].record.assert_called_once_with(aux_stream)


def test_wait_records_the_output_on_the_main_stream(monkeypatch, recorded):
    """Mirror of the input case: the output is allocated on the aux stream and
    consumed, then freed, on the main one."""
    main_stream = MagicMock()
    _patch_shared_experts_stream(monkeypatch, main_stream)
    aux_stream = MagicMock()
    hidden = torch.zeros(8)
    produced = torch.ones(8)
    runner = _bare_shared_experts(aux_stream, lambda x: produced)

    assert runner.maybe_forward_async(hidden)
    runner.wait()

    assert any(t is produced and s is main_stream for t, s in recorded)
    # The main stream must join the aux stream before the output is used.
    runner._output_ready_event[0].wait.assert_called_once_with(main_stream)
    assert runner.output is produced
    # The slot is one-shot: taking the output clears it for the next layer.
    assert runner._output[0] is None
