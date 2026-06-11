# SPDX-License-Identifier: Apache-2.0
"""Trace where mini-checkpoint routing garbage appears: custom op vs router
output vs marlin input.

    python3 tools/ampere/dsv4_router_trace.py /models/v4-flash-mini4-int4mse
"""

import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", "0")
os.environ.setdefault("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", "0")

import torch

import vllm._custom_ops as ops_mod
from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router as router_mod
from vllm.model_executor.layers.fused_moe.experts import marlin_moe as marlin_moe_mod
from vllm.model_executor.layers.quantization import dsv4_int as dsv4_int_mod


def ids_ok(ids: torch.Tensor, e: int) -> str:
    torch.cuda.synchronize()
    mn, mx = ids.min().item(), ids.max().item()
    return f"range=({mn},{mx}) ok={0 <= mn and mx < e}"


_real_op = ops_mod.topk_hash_softplus_sqrt
_op_calls = 0


def traced_op(tw, ti, tei, gating, renorm, rsf, bias, input_tokens, hash_table):
    global _op_calls
    _op_calls += 1
    if _op_calls <= 3:
        print(f"[op call {_op_calls}] gating {tuple(gating.shape)} {gating.dtype} "
              f"finite={torch.isfinite(gating).all().item()} "
              f"bias={'None' if bias is None else tuple(bias.shape)} "
              f"hash={'None' if hash_table is None else tuple(hash_table.shape)} "
              f"input_tokens={'None' if input_tokens is None else tuple(input_tokens.shape)} "
              f"ti_dtype={ti.dtype}", flush=True)
    _real_op(tw, ti, tei, gating, renorm, rsf, bias, input_tokens, hash_table)
    if _op_calls <= 3:
        print(f"[op call {_op_calls}] after: ids {ids_ok(ti, gating.shape[-1])} "
              f"w_row0_sum={tw[0].sum().item():.4f}", flush=True)


ops_mod.topk_hash_softplus_sqrt = traced_op

_real_router = router_mod.fused_topk_bias
_router_calls = 0


def traced_router(*args, **kwargs):
    global _router_calls
    _router_calls += 1
    tw, ti = _real_router(*args, **kwargs)
    if _router_calls <= 3:
        gating = kwargs.get("gating_output", args[1] if len(args) > 1 else None)
        e = gating.shape[-1]
        print(f"[router call {_router_calls}] returned ids {ids_ok(ti, e)} "
              f"w_row0_sum={tw[0].sum().item():.4f}", flush=True)
    return tw, ti


router_mod.fused_topk_bias = traced_router

_real_marlin = marlin_moe_mod.fused_marlin_moe
_marlin_calls = 0


def traced_marlin(hidden_states, w1, w2, b1, b2, w1s, w2s, topk_weights, topk_ids, **kwargs):
    global _marlin_calls
    _marlin_calls += 1
    e = kwargs.get("global_num_experts", w1.shape[0])
    print(f"[marlin call {_marlin_calls}] ids {ids_ok(topk_ids, e)} "
          f"w_row0_sum={topk_weights[0].sum().item():.4f}", flush=True)
    return _real_marlin(hidden_states, w1, w2, b1, b2, w1s, w2s,
                        topk_weights, topk_ids, **kwargs)


marlin_moe_mod.fused_marlin_moe = traced_marlin
dsv4_int_mod.fused_marlin_moe = traced_marlin

from vllm import LLM, SamplingParams

llm = LLM(model=sys.argv[1], trust_remote_code=True, dtype="bfloat16",
          max_model_len=4096, kv_cache_dtype="fp8", enforce_eager=True,
          gpu_memory_utilization=0.6, max_num_batched_tokens=1024, max_num_seqs=1)
out = llm.generate(["The quick brown fox jumps over the lazy dog. " * 200],
                   SamplingParams(max_tokens=4, temperature=0))
print("TRACE-RESULT", list(out[0].outputs[0].token_ids))
print("op_calls", _op_calls, "router_calls", _router_calls, "marlin_calls", _marlin_calls)
