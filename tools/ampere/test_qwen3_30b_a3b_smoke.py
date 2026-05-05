#!/usr/bin/env python3
"""Rung 1 smoke test: Qwen3-30B-A3B-Instruct-2507-AWQ at TP=1 PP=1 on 1 A5000.

Boots vLLM, runs greedy completion of "hi", verifies the output is non-empty
English text (rough check). The PP=1 control answers: does our build run a
known-good Qwen3-MoE checkpoint correctly on sm_86? Without this, any PP>1
failure could be confused with build / quant / runtime issues.

Run with:
  cd forks-vllm-ampere && source .venv/bin/activate
  python tools/ampere/test_qwen3_30b_a3b_smoke.py
"""

from __future__ import annotations

import os
import sys


HF_CACHE = "/home/administrator/inference/.cache/huggingface"
MODEL = "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"


def main() -> int:
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
    os.environ.setdefault("VLLM_USE_V1", "1")
    # Triton sparse MLA path is irrelevant for Qwen3 (no MLA); explicit off
    # to make sure we're not hitting an unrelated kernel selector.
    os.environ.setdefault("VLLM_TRITON_MLA_SPARSE", "0")

    from vllm import LLM, SamplingParams

    print(f"[smoke] booting {MODEL} at TP=1 PP=1", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        # let vllm auto-detect from config.json (compressed-tensors)
        dtype="auto",
        gpu_memory_utilization=0.78,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("[smoke] LLM ready, running smoke prompt", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["hi"], sp)
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print(f"[smoke] prompt='hi' -> {text!r}")
    print(f"[smoke] token_ids={list(token_ids)[:10]}...")

    if len(text.strip()) == 0:
        print("[smoke] FAIL: empty output")
        return 2
    if all(b > 127 for b in text.encode("utf-8", errors="replace")[:8]):
        print(f"[smoke] FAIL: output looks non-ASCII: {text!r}")
        return 2
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
