#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproducer for the Eagle/MTP prepare-inputs Triton kernel.

This avoids model loading and exercises the kernel that prepares draft-model
input ids, positions, query_start_loc, and last_token_indices from the target
batch. It is useful for DeepSeek V4 MTP failures where the live stack points at
``_prepare_eagle_inputs_kernel`` before the draft model runs.

Examples:
  CUDA_LAUNCH_BLOCKING=1 .venv/bin/python scripts/repro_eagle_prepare_inputs.py
  CUDA_LAUNCH_BLOCKING=1 .venv/bin/python scripts/repro_eagle_prepare_inputs.py --case all-rejected
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import prepare_eagle_inputs


def _batch(
    *,
    device: torch.device,
    query_lens: list[int],
    seq_lens: list[int] | None = None,
) -> SimpleNamespace:
    if seq_lens is None:
        seq_lens = query_lens
    num_reqs = len(query_lens)
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc_np[1:] = np.cumsum(query_lens, dtype=np.int32)
    num_tokens = int(query_start_loc_np[-1])
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor(query_start_loc_np, dtype=torch.int32, device=device),
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        dcp_local_seq_lens=None,
        input_ids=torch.arange(100, 100 + num_tokens, dtype=torch.int32, device=device),
        positions=torch.arange(num_tokens, dtype=torch.int64, device=device),
    )


def run_case(case: str, shift_positions: bool) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this Triton reproducer")
    device = torch.device("cuda")
    max_num_reqs = 4

    if case == "normal":
        input_batch = _batch(device=device, query_lens=[4], seq_lens=[4])
        num_sampled = torch.tensor([1], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([0], dtype=torch.int32, device=device)
    elif case == "one-rejected":
        input_batch = _batch(device=device, query_lens=[2], seq_lens=[2])
        num_sampled = torch.tensor([1], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([1], dtype=torch.int32, device=device)
    elif case == "all-rejected":
        input_batch = _batch(device=device, query_lens=[1], seq_lens=[1])
        num_sampled = torch.tensor([1], dtype=torch.int32, device=device)
        num_rejected = torch.tensor([1], dtype=torch.int32, device=device)
    else:
        raise ValueError(f"unknown case: {case}")

    buffers = InputBuffers(max_num_reqs, max(input_batch.num_tokens_after_padding, 1), device)
    last_token_indices = torch.full((max_num_reqs,), -777, dtype=torch.int64, device=device)
    current_draft_step = torch.tensor(123, dtype=torch.int64, device=device)
    last_sampled = torch.full((max_num_reqs, 1), 4242, dtype=torch.int64, device=device)
    next_prefill_tokens = torch.full(
        (max_num_reqs,), 3131, dtype=torch.int32, device=device
    )

    print(
        {
            "case": case,
            "query_start_loc": input_batch.query_start_loc.cpu().tolist(),
            "seq_lens": input_batch.seq_lens.cpu().tolist(),
            "num_sampled": num_sampled.cpu().tolist(),
            "num_rejected": num_rejected.cpu().tolist(),
            "shift_positions": shift_positions,
        }
    )
    prepare_eagle_inputs(
        last_token_indices,
        current_draft_step,
        buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        max_num_reqs,
        shift_positions=shift_positions,
    )
    torch.cuda.synchronize()
    print(
        {
            "last_token_indices": last_token_indices.cpu().tolist(),
            "input_ids": buffers.input_ids.cpu().tolist(),
            "positions": buffers.positions.cpu().tolist(),
            "query_start_loc": buffers.query_start_loc.cpu().tolist(),
            "seq_lens": buffers.seq_lens.cpu().tolist(),
            "current_draft_step": int(current_draft_step.cpu()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("normal", "one-rejected", "all-rejected"),
        default="normal",
    )
    parser.add_argument("--shift-positions", action="store_true")
    args = parser.parse_args()
    run_case(args.case, args.shift_positions)


if __name__ == "__main__":
    main()
