# SPDX-License-Identifier: Apache-2.0
"""Capture the exact fused_marlin_moe call that faults on the mini checkpoints.

Runs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) and wraps
fused_marlin_moe at every import site. Before each call the lightweight args
(activations, routing, shapes, flags) are saved to CAPTURE_DIR/call_args.pt so
the file left behind after the crash describes the faulting call. Weight
tensors are described by shape/dtype only; pass --save-weights to also dump
them (large).

    python3 tools/ampere/dsv4_marlin_moe_capture.py /models/v4-flash-mini4-int4mse /tmp/capture
"""

import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", "0")
os.environ.setdefault("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", "0")

import torch

model_path = sys.argv[1]
capture_dir = sys.argv[2]
save_weights = "--save-weights" in sys.argv
os.makedirs(capture_dir, exist_ok=True)

from vllm.model_executor.layers.fused_moe.experts import marlin_moe as marlin_moe_mod
from vllm.model_executor.layers.quantization import dsv4_int as dsv4_int_mod

_real = marlin_moe_mod.fused_marlin_moe
_call_idx = 0


def _describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "stride": tuple(value.stride()),
            "device": str(value.device),
        }
    return value


def wrapped(hidden_states, w1, w2, *args, **kwargs):
    global _call_idx
    _call_idx += 1
    meta = {
        "call_idx": _call_idx,
        "kwargs_meta": {k: _describe(v) for k, v in kwargs.items()},
        "x": hidden_states.detach().clone().cpu(),
        "w1_meta": _describe(w1),
        "w2_meta": _describe(w2),
        "args_meta": [_describe(a) for a in args],
    }
    # positional args after w2 in this tree: bias1, bias2, w1_scale, w2_scale,
    # topk_weights, topk_ids -- capture routing tensors fully.
    for key in ("topk_weights", "topk_ids", "expert_map", "w1_scale", "w2_scale",
                "global_scale1", "global_scale2",
                "input_global_scale1", "input_global_scale2"):
        v = kwargs.get(key)
        if isinstance(v, torch.Tensor):
            meta[key] = v.detach().clone().cpu()
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor) and a.dtype in (torch.float32, torch.int32) \
                and a.dim() == 2 and a.shape[0] == hidden_states.shape[0]:
            meta[f"pos_arg_{i}"] = a.detach().clone().cpu()
    if save_weights:
        meta["w1"] = w1.detach().clone().cpu()
        meta["w2"] = w2.detach().clone().cpu()
        # scales arrive positionally (bias1, bias2, w1_scale, w2_scale, ...)
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                meta[f"pos_full_{i}"] = a.detach().clone().cpu()
    torch.cuda.synchronize()
    torch.save(meta, f"{capture_dir}/call_args.pt")
    out = _real(hidden_states, w1, w2, *args, **kwargs)
    torch.cuda.synchronize()
    return out


marlin_moe_mod.fused_marlin_moe = wrapped
dsv4_int_mod.fused_marlin_moe = wrapped

from vllm import LLM, SamplingParams

llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16",
          max_model_len=4096, kv_cache_dtype="fp8", enforce_eager=True,
          gpu_memory_utilization=0.6, max_num_batched_tokens=1024, max_num_seqs=1)
out = llm.generate(["The quick brown fox jumps over the lazy dog. " * 200],
                   SamplingParams(max_tokens=4, temperature=0))
print("CAPTURE-RESULT", list(out[0].outputs[0].token_ids))
print("total calls:", _call_idx)
