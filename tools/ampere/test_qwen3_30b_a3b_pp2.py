#!/usr/bin/env python3
"""Rung 1 PP=2: Qwen3-30B-A3B-Instruct-2507-AWQ across 2x A5000 with PP=2.

This reproduces the kv_cache_config-vs-model PP rank assignment mismatch
that broke Qwen3-235B at PP=12 on the chain (patch-6 dump showed: rank
holding model layers 15-22 received kv_spec for layers 55-62).

If this fails identically locally, it confirms the bug is in vllm core
PP slicing not chain-specific. Also produces a much faster iteration
loop than 12-node cluster cycle.

Run with:
  cd forks-vllm-ampere && source .venv/bin/activate
  python tools/ampere/test_qwen3_30b_a3b_pp2.py
"""

from __future__ import annotations

import os
import sys


HF_CACHE = "/home/administrator/inference/.cache/huggingface"
MODEL = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"


def main() -> int:
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
    # Triton sparse MLA path is irrelevant for Qwen3 (no MLA); explicit off
    os.environ.setdefault("VLLM_TRITON_MLA_SPARSE", "0")

    from vllm import LLM, SamplingParams

    print(f"[pp2] booting {MODEL} at TP=1 PP=2 across 2x A5000", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        dtype="auto",
        gpu_memory_utilization=0.78,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
        # 30B / 2 ranks ~ 8 GiB weights per rank, plenty headroom
        # Mp executor (no Ray) fits a single host with 2 GPUs
        # Switch to "ray" to reproduce the cluster bug; "mp" is the
        # single-host fast path. PP=12 cluster used ray; PP=2 here mirrors.
        distributed_executor_backend=os.environ.get("VLLM_EXECUTOR_BACKEND", "mp"),
    )
    print("[pp2] LLM ready, running smoke prompt", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["hi"], sp)
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print(f"[pp2] prompt='hi' -> {text!r}")
    print(f"[pp2] token_ids={list(token_ids)[:10]}...")

    if len(text.strip()) == 0:
        print("[pp2] FAIL: empty output (engine likely crashed)")
        return 2
    print("[pp2] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
