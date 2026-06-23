# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

from types import SimpleNamespace
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "V4_FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = (
    1,
    2,
    3,
    4,
    8,
    16,
    64,
    192,
)
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = (
    64,
    128,
    256,
    512,
    1024,
    2048,
)
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
)
_DEEPSEEK_V4_REQUEST_PREP_WARMUP_REQUESTS = (1, 2, 4, 8, 12, 16)
_DEEPSEEK_V4_REQUEST_PREP_WARMUP_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    192,
    1024,
    2048,
)
_DEEPSEEK_V4_PREFILL_METADATA_WARMUP_REQUESTS = (1, 2, 4, 8, 12, 16)
_DEEPSEEK_V4_PREFILL_METADATA_WARMUP_DECODES = (0, 1, 4, 8, 12)
_DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_NUM_REQS = (1, 2, 4, 12)
_DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_QUERY_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
)
_DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_SLICE_OFFSETS = (0, 1)


class _CombineTopkSwaWarmupKey(NamedTuple):
    device_index: int
    topk: int
    topk_storage: int
    window_size: int
    compress_ratio: int
    m_bound: int
    n_bound: int
    num_reqs: int
    num_tokens: int
    slice_offset: int


_DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUPS: set[_CombineTopkSwaWarmupKey] = set()


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _clamp_warmup_token_sizes(
    num_tokens: tuple[int, ...] | list[int], max_tokens: int
) -> list[int]:
    return sorted(
        {
            clamped
            for requested in num_tokens
            if (clamped := _clamp_warmup_tokens(requested, max_tokens)) > 0
        }
    )


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    max_tokens = getattr(runner, "max_num_tokens", 1)
    v1_input_batch = getattr(runner, "input_batch", None)
    v1_block_table = getattr(v1_input_batch, "block_table", None)
    v2_block_tables = getattr(runner, "block_tables", None)

    if v1_block_table is None and v2_block_tables is None:
        logger.debug(
            "Skipping DeepSeek V4 slot mapping warmup for %s: no block table.",
            type(runner).__name__,
        )
        return

    for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
        num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
        if num_tokens <= 0:
            continue

        positions_source = torch.arange(
            num_tokens, dtype=torch.int64, device=runner.device
        )
        if hasattr(runner, "query_start_loc"):
            runner.query_start_loc.np[0] = 0
            runner.query_start_loc.np[1] = num_tokens
            runner.query_start_loc.copy_to_gpu(2)
            query_start_loc = runner.query_start_loc.gpu[:2]
        else:
            query_start_loc = torch.tensor(
                [0, num_tokens], dtype=torch.int32, device=runner.device
            )

        if hasattr(runner, "positions"):
            runner.positions[:num_tokens].copy_(positions_source)
            positions = runner.positions[:num_tokens]
        else:
            positions = positions_source

        if v1_block_table is not None:
            v1_block_table.commit_block_table(1)
            v1_block_table.compute_slot_mapping(1, query_start_loc, positions)
            continue

        assert v2_block_tables is not None
        idx_mapping = torch.zeros(1, dtype=torch.int32, device=runner.device)
        block_ids = tuple(
            list(range(cdiv(num_tokens, block_size)))
            for block_size in v2_block_tables.block_sizes
        )
        v2_block_tables.append_block_ids(0, block_ids, overwrite=True)
        v2_block_tables.apply_staged_writes()
        v2_block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            positions,
            num_tokens_padded=num_tokens,
        )


def _deepseek_v4_gpu_worker_kernel_warmup(runner: "GPUModelRunner") -> None:
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda":
        return

    max_tokens = max(1, int(getattr(runner, "max_num_tokens", 1)))
    max_num_reqs = int(getattr(runner, "max_num_reqs", 16))
    warmup_reqs = tuple(
        reqs
        for requested in _DEEPSEEK_V4_REQUEST_PREP_WARMUP_REQUESTS
        if (reqs := min(requested, max_num_reqs)) > 0
    )
    warmup_tokens = tuple(
        tokens
        for requested in _DEEPSEEK_V4_REQUEST_PREP_WARMUP_TOKENS
        if (tokens := _clamp_warmup_tokens(requested, max_tokens)) > 0
    )
    if not warmup_reqs or not warmup_tokens:
        return

    from vllm.v1.worker.gpu.input_batch import (
        _post_update_kernel,
        combine_sampled_and_draft_tokens,
        get_num_sampled_and_rejected,
        post_update,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    logger.info(
        "Warming up DeepSeek V4 GPU request-preparation kernels "
        "for request counts=%s and token counts=%s.",
        sorted(set(warmup_reqs)),
        sorted(set(warmup_tokens)),
    )

    def unaligned_1d(length: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(length + 1, dtype=dtype, device=device)[1:]

    def unaligned_2d(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty((rows + 1, cols), dtype=dtype, device=device)[1:]

    for num_reqs in sorted(set(warmup_reqs)):
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        num_computed_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        write_warmup = StagedWriteTensor((num_reqs, 1), torch.int32, device)
        for req_index in range(num_reqs):
            write_warmup.stage_write_elem(req_index, 0)
        write_warmup.apply_write()

        next_prefill_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        last_sampled_tokens = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        draft_tokens = torch.empty((num_reqs, 0), dtype=torch.int64, device=device)
        cu_num_logits = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
        num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=device)

        for num_tokens in sorted(set(warmup_tokens)):
            query_lens = torch.full(
                (num_reqs,),
                max(1, num_tokens // num_reqs),
                dtype=torch.int32,
                device=device,
            )
            query_lens[-1] += max(0, num_tokens - int(query_lens.sum().item()))
            query_start_loc = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
            query_start_loc[0] = 0
            query_start_loc[1:] = torch.cumsum(query_lens, dim=0)
            total_tokens = int(query_start_loc[-1].item())

            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            positions = torch.zeros(total_tokens, dtype=torch.int64, device=device)
            seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            prefill_len = query_lens + 1
            all_token_ids = torch.zeros(
                (num_reqs, max(2, int(prefill_len.max().item()) + 1)),
                dtype=torch.int32,
                device=device,
            )

            warm_num_computed = torch.zeros_like(num_computed_tokens)
            warm_total_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
            sampled_tokens = torch.zeros(
                (num_reqs, 1), dtype=torch.int64, device=device
            )

            prepare_prefill_inputs(
                input_ids,
                next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                all_token_ids,
                prefill_len,
                warm_num_computed,
            )
            prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                warm_num_computed,
                positions,
                seq_lens,
            )
            logits_indices = combine_sampled_and_draft_tokens(
                input_ids,
                idx_mapping,
                last_sampled_tokens,
                query_start_loc,
                seq_lens,
                prefill_len,
                draft_tokens,
                all_token_ids,
                cu_num_logits,
                num_logits=num_reqs,
            )
            sampled, rejected = get_num_sampled_and_rejected(
                num_sampled.clone(),
                seq_lens,
                cu_num_logits,
                idx_mapping,
                prefill_len,
            )
            post_update(
                idx_mapping,
                warm_num_computed,
                last_sampled_tokens,
                None,
                sampled_tokens,
                sampled,
                rejected,
                query_start_loc,
                all_token_ids,
                warm_total_len,
            )
            post_update(
                idx_mapping,
                warm_num_computed,
                last_sampled_tokens,
                torch.zeros((num_reqs, 1), dtype=torch.int32, device=device),
                sampled_tokens,
                sampled,
                rejected,
                query_start_loc,
                all_token_ids,
                warm_total_len,
            )
            unaligned_post_update_args = (
                unaligned_1d(num_reqs, torch.int32).copy_(idx_mapping),
                unaligned_1d(num_reqs, torch.int32).zero_(),
                unaligned_1d(num_reqs, torch.int64).zero_(),
                unaligned_2d(num_reqs, 1, torch.int32).zero_(),
                unaligned_2d(num_reqs, 1, torch.int64).zero_(),
                unaligned_1d(num_reqs, torch.int32).zero_(),
                unaligned_1d(num_reqs, torch.int32).zero_(),
                unaligned_1d(num_reqs + 1, torch.int32).copy_(query_start_loc),
                unaligned_2d(num_reqs,
                             max(2, int(prefill_len.max().item()) + 1),
                             torch.int32).zero_(),
                unaligned_1d(num_reqs, torch.int32).zero_(),
            )
            post_update(
                *unaligned_post_update_args,
            )
            post_update_kernel = _post_update_kernel.warmup(
                torch.int32,
                torch.int32,
                torch.int64,
                torch.int32,
                1,
                torch.int64,
                1,
                torch.int32,
                torch.int32,
                torch.int32,
                torch.int32,
                all_token_ids.stride(0),
                torch.int32,
                grid=(num_reqs,),
                num_warps=1,
            )
            if hasattr(post_update_kernel, "result"):
                post_update_kernel.result()
            # Keep the tensor live until after the launch is queued.
            _ = logits_indices, unaligned_post_update_args

        v2_block_tables = getattr(runner, "block_tables", None)
        if v2_block_tables is not None and hasattr(v2_block_tables, "gather_block_tables"):
            block_ids = tuple([0] for _ in v2_block_tables.block_sizes)
            for req_index in range(num_reqs):
                v2_block_tables.append_block_ids(req_index, block_ids, overwrite=True)
            v2_block_tables.apply_staged_writes()
            v2_block_tables.gather_block_tables(idx_mapping, num_reqs_padded=num_reqs)


def _deepseek_v4_structured_output_bitmask_warmup(
    runner: "GPUModelRunner",
) -> None:
    vocab_size = runner.model_config.get_vocab_size()
    if vocab_size <= 0:
        return

    dtypes = [torch.float32]
    model_dtype = getattr(runner.model_config, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype not in dtypes:
        dtypes.append(model_dtype)

    bitmask_width = (vocab_size + 31) // 32
    req_id = "_deepseek_v4_warmup_"
    grammar_bitmask = np.full((1, bitmask_width), fill_value=-1, dtype=np.int32)
    grammar_output = GrammarOutput(
        structured_output_request_ids=[req_id], grammar_bitmask=grammar_bitmask
    )

    for dtype in dtypes:
        for req_ids in ([req_id], [req_id, "_deepseek_v4_warmup_unmasked_"]):
            logits = torch.zeros(
                (len(req_ids), vocab_size), dtype=dtype, device=runner.device
            )
            input_batch = SimpleNamespace(req_ids=req_ids)
            apply_grammar_bitmask(
                SchedulerOutput.make_empty(), grammar_output, input_batch, logits
            )


def _deepseek_v4_sparse_config(runner: "GPUModelRunner") -> tuple[int, int, int]:
    hf_config = getattr(getattr(runner, "model_config", None), "hf_config", None)
    index_topk = int(getattr(hf_config, "index_topk", 512))
    window_size = int(getattr(hf_config, "sliding_window", 128) or 128)
    compress_ratios = getattr(hf_config, "compress_ratios", None)
    compress_ratio = 4
    if compress_ratios:
        positive_ratios = [int(ratio) for ratio in compress_ratios if int(ratio) > 1]
        if positive_ratios:
            compress_ratio = min(positive_ratios)
    return index_topk, window_size, compress_ratio


def _deepseek_v4_combine_topk_swa_warmup(
    device: torch.device,
    *,
    topk: int,
    topk_storage: int | None = None,
    window_size: int,
    compress_ratio: int,
    m_bounds: tuple[int, ...],
    n_bounds: tuple[int, ...],
    force: bool = False,
) -> None:
    if device.type != "cuda" or topk < 0 or window_size <= 0 or compress_ratio <= 0:
        return
    if topk_storage is None:
        topk_storage = topk
    topk_storage = max(1, int(topk_storage))

    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        combine_topk_swa_indices,
    )

    for m_bound in m_bounds:
        if m_bound < 0:
            continue
        for n_bound in n_bounds:
            if n_bound < 0:
                continue
            for num_reqs in _DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_NUM_REQS:
                if num_reqs <= 0:
                    continue
                for num_tokens in _DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_QUERY_TOKENS:
                    if num_tokens <= 0:
                        continue
                    for slice_offset in (
                        _DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUP_SLICE_OFFSETS
                    ):
                        if slice_offset < 0:
                            continue
                        warmup_key = _CombineTopkSwaWarmupKey(
                            device_index=device_index,
                            topk=int(topk),
                            topk_storage=int(topk_storage),
                            window_size=int(window_size),
                            compress_ratio=int(compress_ratio),
                            m_bound=int(m_bound),
                            n_bound=int(n_bound),
                            num_reqs=int(num_reqs),
                            num_tokens=int(num_tokens),
                            slice_offset=int(slice_offset),
                        )
                        if (
                            not force
                            and warmup_key in _DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUPS
                        ):
                            continue

                        tokens_per_req = max(
                            1, (num_tokens + num_reqs - 1) // num_reqs
                        )
                        query_start_base = (
                            torch.arange(
                                slice_offset + num_reqs + 1,
                                dtype=torch.int32,
                                device=device,
                            )
                            * tokens_per_req
                        )
                        query_start = query_start_base[
                            slice_offset : slice_offset + num_reqs + 1
                        ]
                        num_query_tokens = int(
                            (query_start[-1] - query_start[0]).item()
                        )
                        topk_indices = torch.zeros(
                            num_query_tokens,
                            topk_storage,
                            dtype=torch.int32,
                            device=device,
                        )
                        seq_lens_base = torch.full(
                            (slice_offset + num_reqs,),
                            tokens_per_req + window_size,
                            dtype=torch.int32,
                            device=device,
                        )
                        seq_lens = seq_lens_base[
                            slice_offset : slice_offset + num_reqs
                        ]
                        gather_lens = torch.full(
                            (num_reqs,),
                            min(tokens_per_req + window_size, window_size),
                            dtype=torch.int32,
                            device=device,
                        )
                        combine_topk_swa_indices(
                            topk_indices,
                            query_start,
                            seq_lens,
                            gather_lens,
                            window_size,
                            compress_ratio,
                            topk,
                            M=m_bound,
                            N=n_bound,
                        )
                        _DEEPSEEK_V4_COMBINE_TOPK_SWA_WARMUPS.add(warmup_key)
    torch.accelerator.synchronize()


def _deepseek_v4_prefill_metadata_warmup(
    runner: "GPUModelRunner",
    *,
    force_combine: bool = False,
) -> None:
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda":
        return

    topk, window_size, compress_ratio = _deepseek_v4_sparse_config(runner)
    from vllm.v1.attention.backends.mla.indexer import (
        warmup_prefill_chunk_metadata_kernel,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        warmup_prefill_metadata_kernel,
    )

    warmup_prefill_chunk_metadata_kernel(device, 1)
    if compress_ratio > 1:
        warmup_prefill_chunk_metadata_kernel(device, compress_ratio)
    warmup_prefill_metadata_kernel(
        device,
        window_size,
        _DEEPSEEK_V4_PREFILL_METADATA_WARMUP_REQUESTS,
        _DEEPSEEK_V4_PREFILL_METADATA_WARMUP_DECODES,
    )
    _deepseek_v4_combine_topk_swa_warmup(
        device,
        topk=0,
        topk_storage=topk,
        window_size=window_size,
        compress_ratio=1,
        m_bounds=(1, window_size),
        n_bounds=(0, 1),
        force=force_combine,
    )
    if compress_ratio <= 1:
        return
    _deepseek_v4_combine_topk_swa_warmup(
        device,
        topk=topk,
        topk_storage=topk,
        window_size=window_size,
        compress_ratio=compress_ratio,
        m_bounds=(1, window_size, topk),
        n_bounds=(1, topk),
        force=force_combine,
    )


def _finalize_triton_async_compiles() -> None:
    try:
        from triton.runtime import _async_compile
    except ImportError:
        return

    async_mode = _async_compile.active_mode.get()
    if async_mode is None:
        return

    # Triton may submit JIT work to an async compile mode during model warmup.
    # Finalizing here keeps those warmup compiles from surfacing as first-request
    # latency after the JIT monitor is activated.
    for future in list(async_mode.raw_futures):
        async_mode.future_kernels[future._key].result(async_mode.ignore_errors)


def _deepseek_v4_prefill_forward_warmup(runner: "GPUModelRunner") -> None:
    """Run a dummy PREFILL forward at chunk-sized token counts so the sparse
    attention, the prefill indexer (mqa_logits) and the kv-insert/compress
    kernels compile on a real long-context shape BEFORE first traffic.

    None of the other DeepSeek V4 warmups exercise the model forward at prefill:
    deepseek_v4_mhc_warmup warms only the mHC/MLP path, the direct sparse-MLA
    warmup uses decode shapes (<=32 query tokens, one block of context), and the
    memory profile_run skips attention. So the entire sparse-attention + indexer
    + compressor path is cold on the first prefill, which on the Thunderbolt
    chain shows up as a multi-minute first-request stall (a rank pinned in the
    indexer kernel's module load). A dummy prefill at max_num_batched_tokens warms
    those kernels; the context-independent kernels (stride params made runtime)
    then reuse the same specialization for longer contexts.

    Opt-in via VLLM_DEEPSEEK_V4_PREFILL_FORWARD_WARMUP=1 and exception-safe so it
    can never wedge startup for the whole fleet.
    """
    import os

    if os.environ.get("VLLM_DEEPSEEK_V4_PREFILL_FORWARD_WARMUP", "0") != "1":
        return
    dummy_run = getattr(runner, "_dummy_run", None)
    if dummy_run is None:
        return
    max_tok = int(getattr(runner, "max_num_tokens", 0))
    if max_tok <= 0:
        return
    sizes = sorted({s for s in (512, 2048, max_tok) if 0 < s <= max_tok})
    logger.info("Warming up DeepSeek V4 prefill forward for token counts: %s", sizes)
    for n in sizes:
        try:
            dummy_run(n, skip_attn=False, uniform_decode=False, skip_eplb=True)
        except Exception as exc:  # never let warmup wedge startup
            logger.warning(
                "DeepSeek V4 prefill forward warmup failed at %d tokens: %s", n, exc
            )
            return
    torch.accelerator.synchronize()


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_REQUEST_PREP_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)
    _deepseek_v4_gpu_worker_kernel_warmup(runner)
    _deepseek_v4_prefill_metadata_warmup(runner)
    # Request-prep warmup runs on every pipeline-parallel Ray worker. Keep the
    # raw sparse-MLA direct warmup here unconditional so all PP ranks compile
    # the same Triton specializations before serving first traffic.
    _deepseek_v4_sparse_mla_direct_kernel_warmup(runner)
    _finalize_triton_async_compiles()

    if getattr(runner, "is_last_pp_rank", True):
        try:
            _deepseek_v4_structured_output_bitmask_warmup(runner)
        except ImportError:
            logger.debug(
                "Skipping DeepSeek V4 structured output bitmask warmup because "
                "xgrammar is unavailable."
            )

        torch.accelerator.synchronize()


@torch.inference_mode()
def deepseek_v4_post_capture_request_prep_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_REQUEST_PREP_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    logger.info("Refreshing DeepSeek V4 request preparation warmup after CUDA graphs.")
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )
    _deepseek_v4_slot_mapping_warmup(runner)
    _deepseek_v4_gpu_worker_kernel_warmup(runner)
    _deepseek_v4_prefill_metadata_warmup(runner, force_combine=True)
    if envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_DIRECT_KERNEL_WARMUP:
        _deepseek_v4_sparse_mla_direct_kernel_warmup(runner)
    # Dummy prefill forward (post-capture, so the KV cache is allocated) to warm
    # the sparse-attention / indexer / compress kernels on a real prefill shape.
    _deepseek_v4_prefill_forward_warmup(runner)
    _finalize_triton_async_compiles()
    torch.accelerator.synchronize()


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = min(
        worker.scheduler_config.max_num_batched_tokens,
        getattr(
            runner,
            "max_model_len",
            worker.scheduler_config.max_num_batched_tokens,
        ),
    )
    mixed_token_sizes = _clamp_warmup_token_sizes(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    requested_prefill_sizes = (
        envs.VLLM_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKEN_SIZES
        or _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS
    )
    prefill_token_sizes = _clamp_warmup_token_sizes(
        requested_prefill_sizes, max_tokens
    )
    if not mixed_token_sizes and not prefill_token_sizes:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed token sizes=%s and prefill token sizes=%s.",
        mixed_token_sizes,
        prefill_token_sizes,
    )
    for mixed_tokens in mixed_token_sizes:
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    for prefill_tokens in prefill_token_sizes:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
    _finalize_triton_async_compiles()
    torch.accelerator.synchronize()
    if envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_DIRECT_KERNEL_WARMUP:
        _deepseek_v4_sparse_mla_direct_kernel_warmup(runner)
        _finalize_triton_async_compiles()
        torch.accelerator.synchronize()
    else:
        logger.info("Skipping DeepSeek V4 direct sparse MLA kernel warmup.")


def _deepseek_v4_sparse_mla_direct_kernel_warmup(runner: "GPUModelRunner") -> None:
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda":
        return
    if not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    hf_config = getattr(getattr(runner, "model_config", None), "hf_config", None)
    num_heads = int(getattr(hf_config, "num_attention_heads", 128))
    max_model_len = int(getattr(runner, "max_model_len", 262144))
    max_compressed = max(1, (max_model_len + 127) // 128)
    max_compressed = ((max_compressed + 127) // 128) * 128
    block_size = int(getattr(getattr(runner, "cache_config", None), "block_size", 256))
    cache_block_size = max(block_size, 256)
    num_blocks = max(2, (max_compressed + cache_block_size - 1) // cache_block_size)

    logger.info(
        "Warming up DeepSeek V4 direct sparse MLA kernels "
        "for heads=%s, block_size=%s, max_compressed=%s.",
        num_heads,
        cache_block_size,
        max_compressed,
    )

    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        fp8_paged_mqa_logits_rowwise_triton,
    )
    from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        build_c128a_topk_metadata,
    )
    from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
        accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead,
        accumulate_indexed_sparse_mla_attention_chunk_multihead,
        finish_two_sparse_mla_attention_states_with_sink,
        fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
        fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    )

    token_counts = (1, 2, 4, 8, 16, 32)
    sparse_mla_scale = 512**-0.5
    for num_tokens in token_counts:
        head_block_size = 1 if num_tokens <= 4 else 2 if num_tokens < 16 else 4
        q = torch.zeros(
            (num_tokens, num_heads, 512),
            dtype=torch.bfloat16,
            device=device,
        )
        output = torch.empty_like(q)
        attn_sink = torch.zeros((num_heads,), dtype=torch.float32, device=device)
        max_score = torch.full(
            (num_tokens, num_heads), -float("inf"), dtype=torch.float32, device=device
        )
        denom = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
        acc = torch.zeros(
            (num_tokens, num_heads, 512), dtype=torch.float32, device=device
        )
        seq_lens = torch.full(
            (num_tokens,), cache_block_size, dtype=torch.int32, device=device
        )
        gather_lens = torch.full(
            (num_tokens,), min(128, cache_block_size), dtype=torch.int32, device=device
        )
        block_table = torch.zeros(
            (num_tokens, num_blocks), dtype=torch.int32, device=device
        )
        slot_ids = torch.zeros((num_tokens, 128), dtype=torch.int32, device=device)
        topk_lens = torch.full((num_tokens,), 128, dtype=torch.int32, device=device)

        token_data_size = 576
        scale_bytes = 8
        fp8ds_cache = torch.zeros(
            (num_blocks, cache_block_size * (token_data_size + scale_bytes)),
            dtype=torch.uint8,
            device=device,
        )

        for num_candidates in (64, 128, 512):
            slot_storage = max(128, num_candidates)
            slot_ids = torch.zeros(
                (num_tokens, slot_storage), dtype=torch.int32, device=device
            )
            topk_lens.fill_(min(num_candidates, 512))
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=fp8ds_cache,
                slot_ids=slot_ids[:, :num_candidates],
                lens=topk_lens,
                block_size=cache_block_size,
                scale=sparse_mla_scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
                candidate_offset=0,
                head_block_size=head_block_size,
            )
            accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=fp8ds_cache,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                block_table=block_table,
                block_size=cache_block_size,
                scale=sparse_mla_scale,
                max_score=max_score,
                denom=denom,
                acc=acc,
                candidate_offset=0,
                num_candidates=num_candidates,
                head_block_size=head_block_size,
            )
            fp8ds_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                k_cache=fp8ds_cache,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                block_table=block_table,
                block_size=cache_block_size,
                candidate_offset=0,
                num_candidates=num_candidates,
                scale=sparse_mla_scale,
                attn_sink=attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=num_heads,
            )
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=fp8ds_cache,
                slot_ids=slot_ids[:, :num_candidates],
                topk_lens=topk_lens,
                compressed_block_size=cache_block_size,
                swa_k_cache=fp8ds_cache,
                seq_lens=seq_lens,
                gather_lens=gather_lens,
                block_table=block_table,
                swa_block_size=cache_block_size,
                num_compressed_candidates=num_candidates,
                num_swa_candidates=num_candidates,
                scale=sparse_mla_scale,
                attn_sink=attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=num_heads,
            )

    # The generic indexed sparse MLA accumulation kernel specializes on
    # indices.stride(0). Long prefill chunks use the full top-k buffer width as
    # that stride, so warm representative widths seen in production instead of
    # only the small 128-column synthetic case above.
    indexed_q = torch.zeros(
        (1, num_heads, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    indexed_kv = torch.zeros(
        (max_compressed, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    indexed_lens = torch.full((1,), 512, dtype=torch.int32, device=device)
    indexed_max_score = torch.full(
        (1, num_heads), -float("inf"), dtype=torch.float32, device=device
    )
    indexed_denom = torch.zeros((1, num_heads), dtype=torch.float32, device=device)
    indexed_acc = torch.zeros((1, num_heads, 512), dtype=torch.float32, device=device)
    for index_stride_width in (128, 512, 640, 768, 2176, 8064):
        index_width = min(index_stride_width, max_compressed)
        indexed_indices = torch.zeros(
            (1, index_stride_width), dtype=torch.int32, device=device
        )
        for head_block_size in (1, 2):
            indexed_lens.fill_(min(index_width, 512))
            indexed_max_score.fill_(-float("inf"))
            indexed_denom.zero_()
            indexed_acc.zero_()
            accumulate_indexed_sparse_mla_attention_chunk_multihead(
                q=indexed_q,
                kv_flat=indexed_kv,
                indices=indexed_indices[:, :index_width],
                lens=indexed_lens,
                scale=sparse_mla_scale,
                max_score=indexed_max_score,
                denom=indexed_denom,
                acc=indexed_acc,
                candidate_offset=0,
                head_block_size=head_block_size,
            )

    # The SWA K cache is stored with 64-token blocks even when the runner KV
    # cache block size is larger. Match the live layout so first traffic does
    # not specialize this kernel.
    swa_cache_block_size = 64
    swa_token_data_size = 576
    swa_scale_bytes = 8
    swa_block_stride = swa_cache_block_size * (
        swa_token_data_size + swa_scale_bytes
    )
    for num_reqs in (1, 2, 4, 12):
        max_blocks_per_seq = 4096
        swa_out_aligned = torch.empty(
            (num_reqs, swa_cache_block_size + 1, 512),
            dtype=torch.bfloat16,
            device=device,
        )
        swa_k_cache_aligned = torch.zeros(
            (2, swa_block_stride), dtype=torch.uint8, device=device
        )
        seq_lens_aligned = torch.full(
            (num_reqs,),
            swa_cache_block_size,
            dtype=torch.int32,
            device=device,
        )
        gather_lens_aligned = torch.ones(
            num_reqs, dtype=torch.int32, device=device
        )
        block_table_aligned = torch.zeros(
            (num_reqs, max_blocks_per_seq), dtype=torch.int32, device=device
        )
        swa_out_unaligned_storage = torch.empty(
            (num_reqs * (swa_cache_block_size + 1) * 512 + 1,),
            dtype=torch.bfloat16,
            device=device,
        )
        swa_out_unaligned = torch.as_strided(
            swa_out_unaligned_storage[1:],
            size=(num_reqs, swa_cache_block_size + 1, 512),
            stride=((swa_cache_block_size + 1) * 512, 512, 1),
        )
        swa_k_cache_unaligned = torch.zeros(
            (3, swa_block_stride + 1), dtype=torch.uint8, device=device
        )[1:, 1:]
        seq_lens_sliced = torch.empty((num_reqs + 1,), dtype=torch.int32, device=device)[
            1:
        ]
        seq_lens_sliced.fill_(swa_cache_block_size)
        gather_lens_sliced = torch.empty(
            (num_reqs + 1,), dtype=torch.int32, device=device
        )[1:]
        gather_lens_sliced.fill_(1)
        block_table_sliced = torch.zeros(
            (num_reqs + 1, max_blocks_per_seq + 1),
            dtype=torch.int32,
            device=device,
        )[1:, 1:]
        for offset in (0, 1):
            for swa_out, swa_k_cache, seq_lens, gather_lens, block_table in (
                (
                    swa_out_aligned,
                    swa_k_cache_aligned,
                    seq_lens_aligned,
                    gather_lens_aligned,
                    block_table_aligned,
                ),
                (
                    swa_out_unaligned,
                    swa_k_cache_unaligned,
                    seq_lens_sliced,
                    gather_lens_sliced,
                    block_table_sliced,
                ),
            ):
                dequantize_and_gather_k_cache(
                    swa_out,
                    swa_k_cache,
                    seq_lens=seq_lens,
                    gather_lens=gather_lens,
                    block_table=block_table,
                    block_size=swa_cache_block_size,
                    offset=offset,
                )
                dequantize_and_gather_k_cache(
                    swa_out,
                    swa_k_cache,
                    seq_lens=seq_lens,
                    gather_lens=None,
                    block_table=block_table,
                    block_size=swa_cache_block_size,
                    offset=offset,
                )

    # Warm the live fp8ds sparse decode layouts. C4A compressed cache uses
    # 64-token compressed blocks with 512 top-k candidates; C128A uses 2-token
    # compressed blocks with the same 512-candidate top-k. The synthetic loop
    # above keys off runner cache block size, so cover these model constants
    # directly.
    live_num_reqs = 12
    live_num_heads = min(num_heads, 64)
    live_q = torch.zeros(
        (live_num_reqs, live_num_heads, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    live_output = torch.empty_like(live_q)
    live_sink = torch.zeros((live_num_heads,), dtype=torch.float32, device=device)
    live_seq_lens = torch.full(
        (live_num_reqs,), swa_cache_block_size, dtype=torch.int32, device=device
    )
    live_gather_lens = torch.full(
        (live_num_reqs,), 1, dtype=torch.int32, device=device
    )
    live_block_table = torch.zeros(
        (live_num_reqs, 4096), dtype=torch.int32, device=device
    )
    live_swa_cache = torch.zeros(
        (2, swa_block_stride), dtype=torch.uint8, device=device
    )
    live_swa_slots = torch.zeros(
        (live_num_reqs, swa_cache_block_size), dtype=torch.int32, device=device
    )
    live_swa_lens = torch.full(
        (live_num_reqs,), swa_cache_block_size, dtype=torch.int32, device=device
    )
    live_swa_max = torch.full(
        (live_num_reqs, live_num_heads),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    live_swa_denom = torch.zeros_like(live_swa_max)
    live_swa_acc = torch.zeros(
        (live_num_reqs, live_num_heads, 512), dtype=torch.float32, device=device
    )
    for compressed_block_size in (2, 64):
        compressed_stride = compressed_block_size * (
            swa_token_data_size + swa_scale_bytes
        )
        compressed_cache = torch.zeros(
            (max(256, live_num_reqs), compressed_stride),
            dtype=torch.uint8,
            device=device,
        )
        compressed_slots = torch.zeros(
            (live_num_reqs, 640), dtype=torch.int32, device=device
        )
        compressed_lens = torch.full(
            (live_num_reqs,), 512, dtype=torch.int32, device=device
        )
        comp_max = torch.full(
            (live_num_reqs, live_num_heads),
            -float("inf"),
            dtype=torch.float32,
            device=device,
        )
        comp_denom = torch.zeros_like(comp_max)
        comp_acc = torch.zeros(
            (live_num_reqs, live_num_heads, 512),
            dtype=torch.float32,
            device=device,
        )
        head_block_size = 1
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=live_q,
            k_cache=compressed_cache,
            slot_ids=compressed_slots[:, :512],
            lens=compressed_lens,
            block_size=compressed_block_size,
            scale=sparse_mla_scale,
            max_score=comp_max,
            denom=comp_denom,
            acc=comp_acc,
            candidate_offset=0,
            head_block_size=head_block_size,
        )
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=live_q,
            k_cache=live_swa_cache,
            slot_ids=live_swa_slots,
            lens=live_swa_lens,
            block_size=swa_cache_block_size,
            scale=sparse_mla_scale,
            max_score=live_swa_max,
            denom=live_swa_denom,
            acc=live_swa_acc,
            candidate_offset=0,
            head_block_size=head_block_size,
        )
        accumulate_fp8ds_paged_sparse_mla_attention_chunk_multihead(
            q=live_q,
            k_cache=live_swa_cache,
            seq_lens=live_seq_lens,
            gather_lens=live_gather_lens,
            block_table=live_block_table,
            block_size=swa_cache_block_size,
            scale=sparse_mla_scale,
            max_score=live_swa_max,
            denom=live_swa_denom,
            acc=live_swa_acc,
            candidate_offset=0,
            num_candidates=128,
            head_block_size=head_block_size,
        )
        fp8ds_paged_sparse_mla_attention_with_sink_multihead(
            q=live_q,
            k_cache=live_swa_cache,
            seq_lens=live_seq_lens,
            gather_lens=live_gather_lens,
            block_table=live_block_table,
            block_size=swa_cache_block_size,
            candidate_offset=0,
            num_candidates=128,
            scale=sparse_mla_scale,
            attn_sink=live_sink,
            output=live_output,
            head_block_size=head_block_size,
            num_heads=live_num_heads,
        )
        fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
            q=live_q,
            compressed_k_cache=compressed_cache,
            slot_ids=compressed_slots[:, :512],
            topk_lens=compressed_lens,
            compressed_block_size=compressed_block_size,
            swa_k_cache=live_swa_cache,
            seq_lens=live_seq_lens,
            gather_lens=live_gather_lens,
            block_table=live_block_table,
            swa_block_size=swa_cache_block_size,
            num_compressed_candidates=512,
            num_swa_candidates=128,
            scale=sparse_mla_scale,
            attn_sink=live_sink,
            output=live_output,
            head_block_size=head_block_size,
            num_heads=live_num_heads,
        )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max,
            comp_denom,
            comp_acc,
            live_swa_max,
            live_swa_denom,
            live_swa_acc,
            live_sink,
            output=live_output,
        )

    fp8_logits_cache = torch.zeros(
        (num_blocks, cache_block_size, 512 + 32), dtype=torch.uint8, device=device
    )
    live_block_table_width = 1024
    # The live C128A metadata path passes block_size as a runtime scalar whose
    # Triton specialization does not carry a divisibility assumption. Use a
    # non-16-divisible dummy scalar so warmup compiles that same config.
    c128a_warmup_block_size = cache_block_size + 1
    for num_tokens in token_counts:
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        token_to_req_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        c128a_block_table = torch.zeros(
            (num_tokens, live_block_table_width), dtype=torch.int32, device=device
        )
        global_decode_buffer = torch.empty(
            (num_tokens, max_compressed), dtype=torch.int32, device=device
        )
        decode_lens_buffer = torch.empty(num_tokens, dtype=torch.int32, device=device)
        prefill_buffer = torch.empty(
            (num_tokens, max_compressed), dtype=torch.int32, device=device
        )
        build_c128a_topk_metadata(
            positions=positions,
            compress_ratio=128,
            num_decode_tokens=num_tokens,
            token_to_req_indices=token_to_req_indices,
            block_table=c128a_block_table,
            block_size=c128a_warmup_block_size,
            slot_mapping=slot_mapping,
            global_decode_buffer=global_decode_buffer,
            decode_lens_buffer=decode_lens_buffer,
            prefill_buffer=prefill_buffer,
            max_compressed_tokens=max_compressed,
        )

        for batch_size, next_n in ((num_tokens, 1), (1, num_tokens)):
            fp8_logits_q = torch.zeros(
                (batch_size, next_n, num_heads, 512),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            weights = torch.ones(
                (batch_size * next_n, num_heads),
                dtype=torch.float32,
                device=device,
            )
            context_lens = torch.full(
                (batch_size, next_n),
                cache_block_size,
                dtype=torch.int32,
                device=device,
            )
            # Clamp the fixed warmup widths: any max_model_len <= 16384 gives
            # max_compressed < 256 and the kernel asserts
            # token_start + token_count <= max_model_len.
            warmup_token_counts = sorted(
                {min(128, max_compressed), min(256, max_compressed), max_compressed}
            )
            for token_count in warmup_token_counts:
                fp8_paged_mqa_logits_rowwise_triton(
                    fp8_logits_q,
                    fp8_logits_cache,
                    weights,
                    context_lens,
                    c128a_block_table,
                    max_model_len=max_compressed,
                    token_start=0,
                    token_count=token_count,
                )

    ampere_q_heads = 64
    ampere_q_dim = 128
    configured_gpu_blocks = getattr(
        getattr(runner, "cache_config", None), "num_gpu_blocks", None
    )
    ampere_cache_block_counts = [max(2, int(num_blocks))]
    if configured_gpu_blocks:
        ampere_cache_block_counts.append(max(2, int(configured_gpu_blocks)))
    ampere_cache_block_counts.append(max(19652, num_blocks, 2))
    ampere_cache_block_counts = sorted(set(ampere_cache_block_counts))
    ampere_cache_block_size = 64
    ampere_cache_stride = 8640
    ampere_cache_last_dim = 132
    ampere_cache_storage_tail = max(
        (ampere_cache_block_size - 1) * ampere_q_dim + ampere_cache_last_dim,
        ampere_cache_block_size * ampere_q_dim
        + (ampere_cache_block_size - 1) * (ampere_cache_last_dim - ampere_q_dim)
        + (ampere_cache_last_dim - ampere_q_dim),
    )
    for ampere_cache_blocks in ampere_cache_block_counts:
        ampere_cache_storage = torch.empty(
            (
                (ampere_cache_blocks - 1) * ampere_cache_stride
                + ampere_cache_storage_tail,
            ),
            dtype=torch.uint8,
            device=device,
        )
        ampere_cache = torch.as_strided(
            ampere_cache_storage,
            size=(
                ampere_cache_blocks,
                ampere_cache_block_size,
                1,
                ampere_cache_last_dim,
            ),
            stride=(ampere_cache_stride, ampere_q_dim, ampere_q_dim, 1),
        )
        for batch_size, next_n in ((1, 1), (1, 4), (4, 1)):
            ampere_q = torch.zeros(
                (batch_size, next_n, ampere_q_heads, ampere_q_dim),
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            ampere_weights = torch.ones(
                (batch_size * next_n, ampere_q_heads),
                dtype=torch.float32,
                device=device,
            )
            ampere_context_lens = torch.full(
                (batch_size, next_n),
                ampere_cache_block_size,
                dtype=torch.int32,
                device=device,
            )
            for block_table_width in (320, 1024):
                ampere_block_tables = torch.zeros(
                    (batch_size, block_table_width), dtype=torch.int32, device=device
                )
                for max_logits_width in (512, 1152, 1280, 8064):
                    ampere_context_lens.fill_(
                        min(ampere_cache_block_size, max_logits_width)
                    )
                    for token_count in (
                        128,
                        min(512, max_logits_width),
                        max_logits_width,
                    ):
                        fp8_paged_mqa_logits_rowwise_triton(
                            ampere_q,
                            ampere_cache,
                            ampere_weights,
                            ampere_context_lens,
                            ampere_block_tables,
                            max_model_len=max_logits_width,
                            token_start=0,
                            token_count=token_count,
                        )


def _flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def _resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / _flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_request_prep_warmup(worker)
    if envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_DIRECT_KERNEL_WARMUP:
        _deepseek_v4_sparse_mla_direct_kernel_warmup(worker.model_runner)
        _finalize_triton_async_compiles()
        torch.accelerator.synchronize()

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = _resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(tune_mode=True, cache=str(cache_path)):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        if not is_leader and world.local_rank == 0:
            with open(cache_path, "wb") as f:
                f.write(tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
