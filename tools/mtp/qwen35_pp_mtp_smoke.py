#!/usr/bin/env python3
"""Local PP+MTP correctness smoke for Qwen3.5.

This is intentionally small enough to run on a two-GPU workstation. It uses
Qwen3.5's model-specific MTP implementation with pipeline parallelism and
prints speculative decode counters plus sentinel-contamination checks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def _default_qwen35_path() -> str:
    root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"
    snapshots = sorted(root.glob("*"))
    if not snapshots:
        return "Qwen/Qwen3.5-2B"
    return str(snapshots[-1])


def _metric_value(metrics, name: str) -> float:
    total = 0.0
    for metric in metrics:
        if metric.name == name:
            if isinstance(metric, Counter):
                total += metric.value
            elif isinstance(metric, Vector):
                total += sum(metric.values)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=_default_qwen35_path())
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_PP_MAX_CONCURRENT_BATCHES", str(args.batch))

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )

    prompts = []
    sentinels = []
    for idx in range(args.batch):
        sentinel = f"PPMTP_SENTINEL_{idx:02d}_ONLY"
        sentinels.append(sentinel)
        prompts.append(
            "Repeat exactly this sentinel once and nothing else: "
            f"{sentinel}\nAnswer:"
        )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    outputs = llm.generate(prompts, sampling)

    contamination = 0
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text
        own = sentinels[idx] in text
        foreign = [s for j, s in enumerate(sentinels) if j != idx and s in text]
        contamination += int(bool(foreign))
        token_count = len(output.outputs[0].token_ids)
        print(
            f"row={idx} own={own} foreign={foreign} "
            f"tokens={token_count} text={text[:180]!r}"
        )

    metrics = llm.get_metrics()
    drafts = _metric_value(metrics, "vllm:spec_decode_num_drafts")
    draft_tokens = _metric_value(metrics, "vllm:spec_decode_num_draft_tokens")
    accepted = _metric_value(metrics, "vllm:spec_decode_num_accepted_tokens")
    mean_acceptance = 1.0 + accepted / drafts if drafts else 1.0
    print(
        "metrics "
        f"drafts={drafts:.0f} draft_tokens={draft_tokens:.0f} "
        f"accepted={accepted:.0f} mean_acceptance={mean_acceptance:.3f} "
        f"contaminated_rows={contamination}"
    )


if __name__ == "__main__":
    main()
