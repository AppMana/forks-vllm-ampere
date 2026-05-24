# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


def _dsv4_mtp_trace_enabled() -> bool:
    return os.getenv("VLLM_DSV4_MTP_TRACE", "0") != "0"


def _dsv4_mtp_trace_rows() -> int:
    try:
        return max(0, int(os.getenv("VLLM_DSV4_MTP_TRACE_ROWS", "8")))
    except ValueError:
        return 8


class DraftTokensHandler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.copy_event = torch.cuda.Event()

        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0

    def set_draft_tokens(
        self,
        input_batch: InputBatch,
        draft_tokens: torch.Tensor,
        *,
        force_copy_to_cpu: bool = False,
    ) -> None:
        # InputBatch.req_ids is mutated/reused by the model runner across
        # iterations. Keep a snapshot so PP draft handoff cannot observe a
        # later batch's request ordering.
        self.req_ids = list(input_batch.req_ids)
        self.num_draft_tokens = draft_tokens.shape[1]
        if _dsv4_mtp_trace_enabled():
            rows = _dsv4_mtp_trace_rows()
            logger.warning(
                "DSV4_MTP_TRACE draft_handler_set req_ids=%s "
                "num_draft_tokens=%s force_copy_to_cpu=%s structured=%s",
                self.req_ids[:rows],
                self.num_draft_tokens,
                force_copy_to_cpu,
                input_batch.has_structured_output_reqs,
            )
        if not force_copy_to_cpu and not input_batch.has_structured_output_reqs:
            # No draft token validation needs to be performed by
            # the scheduler for this batch.
            self.draft_tokens_np = None
            return

        # For spec decoding + structured outputs, we must transfer the draft
        # tokens back to the scheduler for grammar validation. Pipeline
        # parallelism also needs the real draft IDs on CPU so the next scheduler
        # output can propagate them to earlier PP ranks, which do not run the
        # drafter locally.
        current_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.copy_stream):
            self.draft_tokens_np = async_copy_to_np(draft_tokens)
            self.copy_event.record()

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            draft_token_ids = self.draft_tokens_np.tolist()
        else:
            # This case only happens when async scheduling is disabled.
            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        if _dsv4_mtp_trace_enabled():
            rows = _dsv4_mtp_trace_rows()
            logger.warning(
                "DSV4_MTP_TRACE draft_handler_get req_ids=%s "
                "draft_token_ids=%s copied_to_cpu=%s",
                self.req_ids[:rows],
                draft_token_ids[:rows],
                self.draft_tokens_np is not None,
            )
        return DraftTokenIds(self.req_ids, draft_token_ids)
