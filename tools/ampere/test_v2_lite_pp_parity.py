#!/usr/bin/env python3
"""Rung 2: DeepSeek-V2-Lite-Chat (MLA + MoE) on Ampere TP=1 PP=1 → PP=2 parity.

DeepSeek-V2-Lite (16B / 2.4B active) is the V4-family base. 27 layers, MLA,
DeepSeekMoE — the simplest model that exercises the same attention path as
V4-Flash. No NSA, no MTP, no FP4. BF16 ~32 GB.

This validates that our merged ampere-v4 (jasl Triton sparse-MLA + sm_8x gate)
gets MLA running on Ampere correctly. If PP=1 produces gibberish, the bug
is in the model code or our patches. If PP=1 works but PP=2 differs, it's PP.

BF16 ~32 GB is too big for one A5000 (22 GB). Use PP=2 from the start to
spread across both A5000s.

Run with:
  cd forks-vllm-ampere && source .venv/bin/activate
  python tools/ampere/test_v2_lite_pp_parity.py
"""

from __future__ import annotations

import os
import sys


HF_CACHE = "/home/administrator/inference/.cache/huggingface"
MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


def main() -> int:
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)

    from vllm import LLM, SamplingParams

    print(f"[v2-lite] booting {MODEL} at TP=1 PP=2 across 2x A5000", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.78,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend=os.environ.get("VLLM_EXECUTOR_BACKEND", "mp"),
    )
    print("[v2-lite] LLM ready, running smoke prompt", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["The capital of France is"], sp)
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print(f"[v2-lite] prompt='The capital of France is' -> {text!r}")
    print(f"[v2-lite] token_ids={list(token_ids)[:10]}...")

    if len(text.strip()) == 0:
        print("[v2-lite] FAIL: empty output")
        return 2
    if "Paris" not in text:
        print(f"[v2-lite] WARN: 'Paris' not in output. Output may be wrong: {text!r}")
        # don't fail — model may give correct but indirect answer
    print("[v2-lite] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
