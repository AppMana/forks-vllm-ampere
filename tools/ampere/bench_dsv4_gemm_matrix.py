# SPDX-License-Identifier: Apache-2.0
"""Benchmark the DSV4 dense/expert GEMMs across every sm_86 kernel path.

Compares, at the exact DeepSeek-V4-Flash layer shapes (read from the layer-5
shard headers of the upstream checkpoint):

- cublas_bf16          torch.matmul on BF16 (dequanted baseline)
- marlin_w4a16_mxfp4   Marlin fe2m1f W4A16, the live routed-experts path
- marlin_w4a16_int4    Marlin uint4b8 W4A16, the int4-int8 checkpoint path
- marlin_w4a8_int8     Marlin uint4b8 + per-token INT8 activations (IMMA)
- marlin_w8a16_int8    Marlin uint8b128 W8A16 channelwise
- triton_w8a16         Triton channel W8A16, the live dense-linear path
- allspark_w8a16       AllSpark W8A16 channelwise

The marlin_w4a8_int8 row is reported twice: gemm-only and including the
per-token activation quant, since the runtime pays the quant on every call.

Run on the 2x RTX A5000 dev box (sm_86, same compute capability as the chain):

    .venv/bin/python tools/ampere/bench_dsv4_gemm_matrix.py
    .venv/bin/python tools/ampere/bench_dsv4_gemm_matrix.py \
        --shapes expert_w13,expert_w2 --ms 1,12,48
"""

import argparse
import dataclasses

import torch
import torch.utils.benchmark as benchmark

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import quantize_weights
from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16 import (
    triton_channel_w8a16_gemm,
)
from vllm.scalar_type import scalar_types

# (size_k, size_n) per GEMM, from upstream DeepSeek-V4-Flash layer-5 shard
# headers (weights stored [N, K]).
SHAPES = {
    "wq_a": (4096, 1024),
    "wq_b": (1024, 32768),
    "wkv": (4096, 512),
    "wo_a": (4096, 8192),
    "wo_b": (8192, 4096),
    "shared_w13": (4096, 4096),
    "shared_w2": (2048, 4096),
    "expert_w13": (4096, 4096),
    "expert_w2": (2048, 4096),
}

# Decode microbatches (1-12 active seqs), per-expert prefill M
# (~2048 tokens * top6 / 256 experts ~= 48), and chunked-prefill M.
DEFAULT_MS = [1, 4, 12, 48, 256, 2048]

GROUP_SIZE = 32  # matches the dsv4 int4 requant group size
MIN_RUN_TIME = 0.5


@dataclasses.dataclass
class Row:
    shape: str
    m: int
    k: int
    n: int
    variant: str
    us: float

    @property
    def tflops(self) -> float:
        return 2 * self.m * self.k * self.n / (self.us * 1e-6) / 1e12


def _time(stmt: str, glb: dict, label: str) -> float:
    t = benchmark.Timer(stmt=stmt, globals=glb, label=label)
    return t.blocked_autorange(min_run_time=MIN_RUN_TIME).median * 1e6


def bench_shape(shape: str, size_k: int, size_n: int, m: int) -> list[Row]:
    device = torch.device("cuda")
    a = torch.randn((m, size_k), dtype=torch.bfloat16, device=device)
    b = torch.randn((size_k, size_n), dtype=torch.bfloat16, device=device) / 10
    workspace = marlin_make_workspace_new(device)
    rows: list[Row] = []

    def add(variant: str, stmt: str, glb: dict) -> None:
        glb = dict(glb, a=a, workspace=workspace, ops=ops, torch=torch)
        us = _time(stmt, glb, f"{shape} M={m}")
        rows.append(Row(shape, m, size_k, size_n, variant, us))

    # cuBLAS BF16 baseline
    add("cublas_bf16", "torch.matmul(a, b)", {"b": b})

    # Marlin fe2m1f W4A16 (live routed-experts path)
    w_ref, q_w, s = rand_marlin_weight_mxfp4_like(b.T, GROUP_SIZE)
    add(
        "marlin_w4a16_mxfp4",
        "ops.marlin_gemm(a, None, q_w, None, s, None, None, None, None, None,"
        " workspace, qt, a.shape[0], n, k, True, False, False, False)",
        {"q_w": q_w, "s": s, "qt": scalar_types.float4_e2m1f, "n": size_n, "k": size_k},
    )

    # Marlin uint4b8 W4A16 (int4-int8 checkpoint path)
    _, q_w4, s4, g_idx, sort_idx, _ = marlin_quantize(
        b, scalar_types.uint4b8, GROUP_SIZE, act_order=False
    )
    add(
        "marlin_w4a16_int4",
        "ops.marlin_gemm(a, None, q_w, None, s, None, None, None, g_idx, sort_idx,"
        " workspace, qt, a.shape[0], n, k, True, False, False, False)",
        {
            "q_w": q_w4,
            "s": s4,
            "g_idx": g_idx,
            "sort_idx": sort_idx,
            "qt": scalar_types.uint4b8,
            "n": size_n,
            "k": size_k,
        },
    )

    # Marlin uint4b8 + INT8 activations (integer MMA path)
    _, q_w48, s48, g_idx8, sort_idx8, _ = marlin_quantize(
        b, scalar_types.uint4b8, GROUP_SIZE, act_order=False, input_dtype=torch.int8
    )
    # group_size != -1: weight scales become int16-quantized relative values
    # and the activation scales absorb the global factor
    # (mirrors marlin_act_int8_process_scales / test_marlin_gemm.py).
    global_factor = (s48.max() / 4096).float()
    s48 = (s48 / s48.max() * 4096).round().to(torch.int16).view(torch.bfloat16)
    a_q, a_s = per_token_quant_int8(a)
    a_s = (a_s * global_factor).float()
    int8_glb = {
        "q_w": q_w48,
        "s": s48,
        "a_q": a_q,
        "a_s": a_s,
        "g_idx": g_idx8,
        "sort_idx": sort_idx8,
        "qt": scalar_types.uint4b8,
        "n": size_n,
        "k": size_k,
        "quant": per_token_quant_int8,
        "factor": global_factor,
    }
    add(
        "marlin_w4a8_int8(gemm)",
        "ops.marlin_gemm(a_q, None, q_w, None, s, a_s, None, None, g_idx, sort_idx,"
        " workspace, qt, a_q.shape[0], n, k, True, False, False, False)",
        int8_glb,
    )
    add(
        "marlin_w4a8_int8(+quant)",
        "aq, asc = quant(a)\n"
        "ops.marlin_gemm(aq, None, q_w, None, s, (asc * factor).float(), None, None,"
        " g_idx, sort_idx, workspace, qt, aq.shape[0], n, k, True, False, False,"
        " False)",
        int8_glb,
    )

    # Marlin uint8b128 W8A16 channelwise
    _, q_w8, s8, g_idx_w8, sort_idx_w8, _ = marlin_quantize(
        b, scalar_types.uint8b128, -1, act_order=False
    )
    add(
        "marlin_w8a16_int8",
        "ops.marlin_gemm(a, None, q_w, None, s, None, None, None, g_idx, sort_idx,"
        " workspace, qt, a.shape[0], n, k, True, False, False, False)",
        {
            "q_w": q_w8,
            "s": s8,
            "g_idx": g_idx_w8,
            "sort_idx": sort_idx_w8,
            "qt": scalar_types.uint8b128,
            "n": size_n,
            "k": size_k,
        },
    )

    # Triton channel W8A16 (live dense-linear path); weight [N, K] uint8 +128
    w_int = torch.randint(
        0, 255, (size_n, size_k), dtype=torch.uint8, device=device
    ).contiguous()
    w_scale = (b.abs().amax(dim=0) / 127).contiguous()
    add(
        "triton_w8a16",
        "f(a, w, sc)",
        {"f": triton_channel_w8a16_gemm, "w": w_int, "sc": w_scale},
    )

    # AllSpark W8A16 channelwise
    _, qw_as, s_as, _ = quantize_weights(b, scalar_types.uint8b128, -1, False)
    qw_reorder, s_reorder, _ = ops.allspark_repack_weight(
        qw_as.to(torch.uint8), s_as, None, False
    )
    props = torch.cuda.get_device_properties(device)
    add(
        "allspark_w8a16",
        "ops.allspark_w8a16_gemm(a, qw, sc, None, n, -1, sm_count, sm_version,"
        " thresh, False, True)",
        {
            "qw": qw_reorder,
            "sc": s_reorder,
            "n": size_n,
            "sm_count": props.multi_processor_count,
            "sm_version": props.major * 10 + props.minor,
            "thresh": ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
        },
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--ms", default=",".join(map(str, DEFAULT_MS)))
    parser.add_argument("--out", default=None, help="optional TSV output path")
    args = parser.parse_args()

    shapes = [s for s in args.shapes.split(",") if s]
    ms = [int(x) for x in args.ms.split(",") if x]

    all_rows: list[Row] = []
    for shape in shapes:
        size_k, size_n = SHAPES[shape]
        for m in ms:
            torch.manual_seed(0)
            all_rows.extend(bench_shape(shape, size_k, size_n, m))

    header = f"{'shape':<12} {'M':>5} {'K':>5} {'N':>6} {'variant':<26} {'us':>10} {'TFLOP/s':>8}"
    print(header)
    print("-" * len(header))
    lines = ["shape\tM\tK\tN\tvariant\tus\ttflops"]
    for r in all_rows:
        print(
            f"{r.shape:<12} {r.m:>5} {r.k:>5} {r.n:>6} {r.variant:<26}"
            f" {r.us:>10.1f} {r.tflops:>8.2f}"
        )
        lines.append(
            f"{r.shape}\t{r.m}\t{r.k}\t{r.n}\t{r.variant}\t{r.us:.2f}\t{r.tflops:.3f}"
        )
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
