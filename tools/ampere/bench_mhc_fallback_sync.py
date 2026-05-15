#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark DeepSeek V4 mHC Torch fallback synchronization modes.

This is intentionally a local kernel/op benchmark, not a serving benchmark.
The ``none`` mode is useful here as a raw overhead baseline even when it is
known-incorrect for PP serving, where tensors cross rank/stage boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers import mhc


@dataclass(frozen=True)
class BenchResult:
    op: str
    mode: str
    backend: str
    tokens: int
    hidden_size: int
    warmup_iters: int
    bench_iters: int
    cuda_mean_ms: float
    cuda_median_ms: float
    cuda_p90_ms: float
    cuda_min_ms: float
    cuda_max_ms: float
    wall_mean_ms: float
    wall_median_ms: float
    wall_p90_ms: float
    wall_min_ms: float
    wall_max_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _summarize(
    op: str,
    mode: str,
    backend: str,
    tokens: int,
    hidden_size: int,
    warmup_iters: int,
    bench_iters: int,
    cuda_values_ms: list[float],
    wall_values_ms: list[float],
) -> BenchResult:
    return BenchResult(
        op=op,
        mode=mode,
        backend=backend,
        tokens=tokens,
        hidden_size=hidden_size,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
        cuda_mean_ms=statistics.fmean(cuda_values_ms),
        cuda_median_ms=statistics.median(cuda_values_ms),
        cuda_p90_ms=_percentile(cuda_values_ms, 0.90),
        cuda_min_ms=min(cuda_values_ms),
        cuda_max_ms=max(cuda_values_ms),
        wall_mean_ms=statistics.fmean(wall_values_ms),
        wall_median_ms=statistics.median(wall_values_ms),
        wall_p90_ms=_percentile(wall_values_ms, 0.90),
        wall_min_ms=min(wall_values_ms),
        wall_max_ms=max(wall_values_ms),
    )


def _time_cuda(
    fn: Callable[[], object],
    op: str,
    mode: str,
    backend: str,
    tokens: int,
    hidden_size: int,
    warmup_iters: int,
    bench_iters: int,
) -> BenchResult:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    cuda_values_ms: list[float] = []
    wall_values_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_values_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_values_ms.append(start.elapsed_time(end))

    return _summarize(
        op,
        mode,
        backend,
        tokens,
        hidden_size,
        warmup_iters,
        bench_iters,
        cuda_values_ms,
        wall_values_ms,
    )


def _make_inputs(tokens: int, hidden_size: int, hc_mult: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    return {
        "residual": torch.randn(
            tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16
        ),
        "fn": torch.randn(
            hc_mult3,
            hc_mult * hidden_size,
            device=device,
            dtype=torch.float32,
        ),
        "hc_scale3": torch.randn(3, device=device, dtype=torch.float32),
        "hc_base3": torch.randn(hc_mult3, device=device, dtype=torch.float32),
        "x": torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16),
        "post": torch.randn(tokens, hc_mult, 1, device=device, dtype=torch.float32),
        "comb": torch.randn(tokens, hc_mult, hc_mult, device=device, dtype=torch.float32),
        "head_fn": torch.randn(
            hc_mult,
            hc_mult * hidden_size,
            device=device,
            dtype=torch.float32,
        ),
        "head_scale": torch.randn(1, device=device, dtype=torch.float32),
        "head_base": torch.randn(hc_mult, device=device, dtype=torch.float32),
        "head_out": torch.empty(tokens, hidden_size, device=device, dtype=torch.bfloat16),
    }


def _bench_one_size(
    tokens: int,
    hidden_size: int,
    hc_mult: int,
    mode: str,
    warmup_iters: int,
    bench_iters: int,
) -> list[BenchResult]:
    tensors = _make_inputs(tokens, hidden_size, hc_mult)

    def run_pre() -> object:
        return mhc.mhc_pre(
            tensors["residual"],
            tensors["fn"],
            tensors["hc_scale3"],
            tensors["hc_base3"],
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=1,
        )

    def run_post_torch() -> object:
        os.environ["VLLM_MHC_POST_TRITON"] = "0"
        return mhc.mhc_post(
            tensors["x"],
            tensors["residual"],
            tensors["post"],
            tensors["comb"],
        )

    def run_post_triton() -> object:
        os.environ["VLLM_MHC_POST_TRITON"] = "1"
        return mhc.mhc_post(
            tensors["x"],
            tensors["residual"],
            tensors["post"],
            tensors["comb"],
        )

    def run_head() -> object:
        return mhc._hc_head_fused_kernel(
            tensors["residual"],
            tensors["head_fn"],
            tensors["head_scale"],
            tensors["head_base"],
            tensors["head_out"],
            hidden_size,
            rms_eps=1e-6,
            hc_eps=1e-6,
            hc_mult=hc_mult,
        )

    return [
        _time_cuda(
            run_pre,
            "mhc_pre",
            mode,
            "torch",
            tokens,
            hidden_size,
            warmup_iters,
            bench_iters,
        ),
        _time_cuda(
            run_post_torch,
            "mhc_post",
            mode,
            "torch",
            tokens,
            hidden_size,
            warmup_iters,
            bench_iters,
        ),
        _time_cuda(
            run_post_triton,
            "mhc_post",
            mode,
            "triton",
            tokens,
            hidden_size,
            warmup_iters,
            bench_iters,
        ),
        _time_cuda(
            run_head,
            "hc_head",
            mode,
            "torch",
            tokens,
            hidden_size,
            warmup_iters,
            bench_iters,
        ),
    ]


def _print_table(results: list[BenchResult]) -> None:
    print(
        f"{'op':<10} {'backend':<8} {'mode':<8} {'tokens':>7} "
        f"{'cuda_mean':>10} {'cuda_p90':>10} {'wall_mean':>10} {'wall_p90':>10}"
    )
    for result in results:
        print(
            f"{result.op:<10} {result.backend:<8} {result.mode:<8} "
            f"{result.tokens:>7} "
            f"{result.cuda_mean_ms:>10.3f} {result.cuda_p90_ms:>10.3f} "
            f"{result.wall_mean_ms:>10.3f} {result.wall_p90_ms:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 64])
    parser.add_argument(
        "--modes", nargs="+", choices=["none", "stream", "device"], default=["none", "stream", "device"]
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--json-output", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    os.environ["VLLM_MHC_DEBUG_TIMINGS"] = "0"
    os.environ["VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE"] = "1"

    results: list[BenchResult] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for mode in args.modes:
        os.environ["VLLM_MHC_TORCH_FALLBACK_SYNC_MODE"] = mode
        for tokens in args.tokens:
            results.extend(
                _bench_one_size(
                    tokens,
                    args.hidden_size,
                    args.hc_mult,
                    mode,
                    args.warmup_iters,
                    args.bench_iters,
                )
            )
    torch.cuda.synchronize()

    _print_table(results)
    print(f"elapsed_s={time.perf_counter() - started:.2f}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump([result.__dict__ for result in results], f, indent=2)


if __name__ == "__main__":
    main()
