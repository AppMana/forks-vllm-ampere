# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group


def _get_pp_pynccl_comm():
    device_communicator = getattr(get_pp_group(), "device_communicator", None)
    pynccl_comm = getattr(device_communicator, "pynccl_comm", None)
    if pynccl_comm is None or pynccl_comm.disabled:
        return None
    return pynccl_comm


def _send_to_non_last_pp_ranks(*tensors: torch.Tensor) -> bool:
    pp = get_pp_group()
    pynccl_comm = _get_pp_pynccl_comm()
    if pynccl_comm is None:
        return False

    pynccl_comm.group_start()
    try:
        for dst in range(pp.world_size - 1):
            for tensor in tensors:
                if tensor.numel() > 0:
                    pynccl_comm.send(tensor, dst)
    finally:
        pynccl_comm.group_end()
    for tensor in tensors:
        if tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
    return True


def _recv_from_last_pp_rank(*tensors: torch.Tensor) -> bool:
    pp = get_pp_group()
    pynccl_comm = _get_pp_pynccl_comm()
    if pynccl_comm is None:
        return False

    pynccl_comm.group_start()
    try:
        for tensor in tensors:
            if tensor.numel() > 0:
                pynccl_comm.recv(tensor, pp.world_size - 1)
    finally:
        pynccl_comm.group_end()
    return True


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    combined = torch.stack((num_sampled, num_rejected), dim=0)
    if envs.VLLM_PP_ASYNC_TOKEN_COMM == "pynccl_fanout" and _send_to_non_last_pp_ranks(
        sampled_token_ids.contiguous(), combined
    ):
        return

    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    if envs.VLLM_PP_ASYNC_TOKEN_COMM == "pynccl_fanout" and _recv_from_last_pp_rank(
        sampled_tokens, combined
    ):
        num_sampled, num_rejected = combined.unbind(dim=0)
        return sampled_tokens, num_sampled, num_rejected

    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
