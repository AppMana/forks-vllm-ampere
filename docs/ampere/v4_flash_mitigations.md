# DeepSeek-V4-Flash on Ampere (sm_86) — Architectural Mitigations

This document is the complete catalogue of code changes the `ampere-v4`
branch carries on top of upstream vLLM (`v0.20.2rc1.dev` plus PRs
[#41694](https://github.com/vllm-project/vllm/pull/41694),
[#40871](https://github.com/vllm-project/vllm/pull/40871),
[#40991](https://github.com/vllm-project/vllm/pull/40991),
[#41653](https://github.com/vllm-project/vllm/pull/41653)) to make
DeepSeek-V4-Flash run on Ampere GPUs (sm_86, e.g. RTX 3090 / A5000) that
**lack native FP8 hardware**.

The baseline cluster target is the 12-node TB-chain in
`appmana-cluster-03` (12× RTX 3090, sm_86, 24 GB each, 1 GPU per node,
no NVLink between boxes); this also runs on a 2× RTX A5000 dev box for
local iteration.

---

## 1. The hardware constraint that drives everything else

DeepSeek-V4-Flash is engineered for sm_89+ (Ada / Hopper / Blackwell).
Its checkpoint format and reference kernels assume:

* **Native FP8 E4M3 tensor cores** (sm_89+) for the FP8 block-quant linear
  layers, sparse-MLA attention, and the MQA logits indexer
* **DeepGEMM / TileLang FP8 paths** that require Hopper `wgmma` / Blackwell
  `tcgen05.mma` instructions for FP8 GEMM
* **TMA (Tensor Memory Accelerator)**, `cp.async.bulk`, and similar
  Hopper-only memory-movement primitives in the SM12x sparse-MLA kernels

On sm_86, **none of those exist**. Concretely, Triton's NVIDIA backend
refuses to lower the `tl.float8e4nv` type below sm_89:

```
ValueError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

Any kernel that does `x.to(tl.float8e4nv)` (encode), `uint8.to(tl.float8e4nv,
bitcast=True)` (decode), or `tl.dot(fp8, fp8, ...)` (FP8 MMA) will refuse to
compile on sm_86 — and most of the V4-Flash compute path goes through such
kernels.

Mitigations fall into three classes:

1. **Replace the cast** with arithmetic-only encode/decode that sm_86
   *can* lower → kernel keeps its structure, only the FP8 type-conversion
   sites change.
2. **Replace the kernel** when the hot inner op is `tl.dot(fp8, fp8)`
   (MMA) → fall back to a pure-PyTorch implementation that dequant→bf16
   matmuls.
3. **Re-route the data layout** when an upstream optimization (e.g.
   Marlin INT-packed weights) is incompatible with how the kernel reads
   the tensor.

---

## 2. The arithmetic FP8 helpers (Class 1 mitigations)

`vllm/v1/attention/ops/deepseek_v4_ops/fp8e4m3_arith.py`:

* **`fp8e4m3_decode_to_fp32(x_uint8)`** — Triton kernel that reads a
  `uint8` byte holding an E4M3FN-encoded value and produces an `fp32`
  via integer arithmetic (sign/exponent/mantissa unpack + `tl.exp2`).
  Bit-exact with `torch.float8_e4m3fn.to(fp32)` for all 256 byte values
  except the two NaN encodings (0x7F, 0xFF), which we map to ±448.0
  (E4M3FN max-finite). KV-cache values cannot be NaN, so this is benign.

* **`fp8e4m3_encode_from_fp32(x)`** — symmetric encoder that produces the
  E4M3FN byte from an fp32. Round-half-away-from-zero rather than IEEE
  round-to-nearest-even (Triton lacks `tl.rint`); ~98.4% byte-identical
  with PyTorch's RNE encoder. The disagreements are at exact half-way
  values where either rounding direction is correct within E4M3 quant
  noise. Saturates at ±448.0 (encodes max-finite, never +/-Inf, since
  E4M3FN has no infinities).

These helpers compile cleanly on sm_86 (verified by `tests/v1/attention/
ops/test_v4_fp8_einsum_torch_fallback.py` and the local smoke test
under `--load-format dummy`).

### Where they are used

| Site | File | Direction |
|---|---|---|
| Sparse-MLA decode/prefill (7 kernel families) | `vllm/v1/attention/backends/mla/sparse_mla_kernels.py` | decode (uint8→fp32) |
| FP8 paged MQA logits indexer (2 kernels) | `vllm/model_executor/layers/deepseek_v4_triton_kernels.py` | decode |
| Fused compress + RMSNorm + RoPE + FP8 quant (compressor sparse + indexer) | `vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py` | encode |
| MegaMoE input-staging quant (DeepSeek expert input quant) | `vllm/model_executor/models/deepseek_v4.py` | encode |

For encode-side sites, the kernel signature is patched to take a `uint8*`
pointer (via `dst_ptr.to(tl.pointer_type(tl.uint8))`) and the wrapper passes
`tensor.view(torch.uint8)`. Strides are unchanged — fp8 and uint8 share a
1-byte element width.

---

## 3. The torch fallbacks (Class 2 mitigations)

These kernels' inner loops do `tl.dot(fp8, fp8, out_dtype=fp32)` — the
multiply-accumulate is performed in FP8 hardware, which sm_86 doesn't have.
Arithmetic dequant doesn't help because the MMA itself is unsupported.
We fall back to PyTorch dequant→bf16-matmul.

| File | Path | Replacement |
|---|---|---|
| `wo_a` output projection FP8 einsum | `vllm/v1/attention/ops/deepseek_v4_ops/fp8_einsum.py` | `_deepseek_v4_fp8_einsum_torch` — fp32 dequant + `torch.bmm` |
| `fused_inv_rope_fp8_quant` | `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` | `_fused_inv_rope_fp8_quant_torch` — full RoPE + block-quant + fp8 cast in PyTorch |
| `fused_indexer_q` | `vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py` | `_fused_indexer_q_rope_fp8_torch` with weight-folding |
| `quantize_and_insert_k_cache` (UE8M0 K-cache write) | `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` | `_quantize_and_insert_k_cache_torch` (3 KV-cache I/O kernels) |
| `dequantize_and_gather_k_cache`, `dequantize_global_slots_k_cache` | `cache_utils.py` | `_dequantize_*_torch` |

Each fallback is gated by `_supports_fp8e4nv_in_triton()` (returns False on
sm_8x), so on sm_89+ the original Triton kernels still run unchanged.

The torch fallbacks for cache I/O were the slowest of all the mitigations
(they execute on every prefill / decode step). A future optimization is to
port them to use the arithmetic helpers — keeps the data on GPU in
Triton, no torch round-trip — but that's a larger refactor.

---

## 4. Layout / dispatch fixes (Class 3 mitigations)

### `wo_a` Marlin bypass

`vllm/model_executor/kernels/linear/scaled_mm/marlin.py` —
`MarlinFP8ScaledMMLinearKernel.process_weights_after_loading` skips
`prepare_fp8_layer_for_marlin` when the layer has `is_bmm=True`.

Why: V4-Flash builds `wo_a` as a fused per-group `ColumnParallelLinear`
(input=4096, output=8192), but at runtime the model **does not call
`wo_a(x)` via the standard linear apply()** — instead the V4 attention
custom op reads `wo_a.weight` and `wo_a.weight_scale_inv` directly and
does its own FP8 einsum. On sm_86 the Marlin kernel is selected for FP8
block-quant linears, and `prepare_fp8_layer_for_marlin` repacks the
weight via `gptq_marlin_repack` into `(size_k // 16, size_n * 16 //
pack_factor)` layout, which mangles the `(out, in)` shape the einsum
expects. Bypassing the repack for `is_bmm=True` keeps the weight in
canonical form for direct access.

### sm_8x capability gates

* `vllm/v1/attention/backends/mla/sparse_mla_env.py` —
  `_is_triton_sparse_mla_compatible_device` widened to include
  `cap.major == 8`.
* `vllm/utils/deep_gemm.py` — three dispatch sites
  (`fp8_mqa_logits`, `fp8_paged_mqa_logits`, `tf32_hc_prenorm_gemm`)
  widened to include `is_device_capability_family(80)`.
* `vllm/model_executor/layers/deepseek_v4_attention.py` —
  `_use_deepseek_v4_sm12x_triton_fp8_einsum` now matches sm_8x in addition
  to sm_12x.
* `vllm/model_executor/layers/sparse_attn_indexer.py` —
  `_sparse_indexer_requires_deep_gemm` skipped on sm_8x.
* `vllm/model_executor/layers/mhc.py` — hyperconnection kernels widened
  for sm_8x torch fallback path.

### KV-cache view → reshape

The chain's KV cache tensor arrives as `(num_blocks, block_size,
head_bytes)` with outer-stride padding > shape[1]. The original Triton
kernel uses raw byte arithmetic via `cache_ptr + block_idx * stride(0)`
and doesn't care; the torch fallback does `k_cache.view(-1)` which fails
with "view size is not compatible with input tensor's size and stride".
Fix in `_gather_token_bytes` and `_quantize_and_insert_k_cache_torch`:
flatten via `k_cache.view(k_cache.shape[0], -1)` regardless of input
rank; then index per-row with `index_select` + `gather` / `index_put_`.

### V4 Compile cache + JIT cache layout

* **`/jit-shared` (RWX-NFS PVC `jit-cache-shared`, 50 GiB)** — content-
  hashed kernel artifacts that match across the 12 RTX-3090 chain ranks.
  First rank to JIT writes; rest load. Survives pod recreate. Holds:
  `TRITON_CACHE_DIR=/jit-shared/triton`,
  `TORCH_EXTENSIONS_DIR=/jit-shared/torch-ext`,
  `CUDA_CACHE_PATH=/jit-shared/cuda`.
* **`/jit-local` (per-pod ephemeral `local-path` PVC, 10 GiB)** — non-
  shareable per-pod state: `TORCHINDUCTOR_CACHE_DIR=/jit-local/inductor`,
  `XDG_CACHE_HOME=/jit-local/xdg`. Lost on pod delete; cheap to rebuild.

`VLLM_DISABLE_COMPILE_CACHE=1` was hardcoded in earlier LWS revisions
and silently disabled vLLM's compile cache; **removed**.

### Sparse-MLA request-prep JIT warmup

DeepSeek V4 sparse prefill has several Triton request-preparation kernels
outside the main model graph: prefill chunk metadata, SWA metadata, and
combined top-k/SWA index construction. Upstream vLLM fixes for similar
first-request JIT latency followed two patterns:

* use `@triton.jit(do_not_specialize=[...])` for scalar inputs whose values
  vary by request but do not need compile-time specialization
  (`e80cfe575`, `_gather_block_tables_kernel`);
* make synthetic warmup mirror the real prefill path instead of compiling an
  idealized shape (`9853a3c15`, `e9f331d72`).

This branch applies the same approach to V4 sparse MLA:

* `_combine_topk_swa_indices_kernel` no longer specializes `M` and `N`;
* request-prep warmup covers prefill chunk metadata, SWA metadata, and both
  combine variants (`TOP_K=0, COMPRESS_RATIO=1` and
  `TOP_K=512, COMPRESS_RATIO=4`);
* combine warmup intentionally covers both aligned base tensors and unaligned
  slices for `query_start_loc`/`seq_lens`, because real mixed prefill can pass
  sliced metadata tensors whose Triton divisibility attributes differ from a
  fresh `torch.arange`;
* a small post-CUDA-graph refresh recompiles only these request-prep kernels
  before `jit_monitor.activate()`, so capture/profiling cannot leave the first
  user request to pay that JIT cost.

The diagnostic knob `VLLM_TRITON_JIT_MONITOR_DETAILS=1` expands JIT monitor
warnings with the Triton cache key and pointer-divisibility config. This was
what exposed the aligned-vs-sliced metadata mismatch.

### `dsv4_int`: AOT integer checkpoint path

The `dsv4_int` quantization method is the current path toward native
integer tensor-core use on Ampere:

* Routed experts: DeepSeek MXFP4 packed weights are converted AOT to
  symmetric INT4 W4A16, group size 32, then repacked for Marlin MoE.
* Dense FP8 linears: two formats are supported.
  * `strategy=block`: signed INT8 with 128x128 scales. This is the
    conservative format and dequantizes once to BF16 during load.
  * `strategy=channel`: AllSpark-compatible biased UINT8 with one scale per
    output channel. Most dense linears run through the Ampere AllSpark W8A16
    kernel without materializing BF16 weights. `attn.wo_a` is still dequanted
    to BF16 because V4 attention reads that tensor directly in the inverse
    RoPE einsum path, whose FP8 helper expects block scales.
* Preserved tensors: embeddings, norms, gates, attention sinks, HC tensors,
  and explicitly BF16/F32 tensors stay in their original precision.

Converter:

```
.venv/bin/python tools/ampere/dsv4_requant_checkpoint.py \
  --src /path/to/deepseek-v4-fp4-fp8 \
  --dst /path/to/deepseek-v4-dsv4-int \
  --device cuda:0 \
  --dense-int8-strategy channel \
  --overwrite
```

Local AllSpark kernel probe on SM86 (`m=12,n=4096,k=4096,bf16`):

| Kernel | Mean ms | Note |
|---|---:|---|
| AllSpark W8A16 | 0.036-0.042 | biased UINT8 weight, per-channel scale |
| BF16 dequantized `F.linear` | 0.065 | same dequantized weight |

The probe measured ~51.4 dB SNR against the BF16-dequantized reference and
~1.5-1.8x speedup for this small decode-like GEMM. The actual model still has
large non-linear bottlenecks in sparse MLA, MoE routing, PP scheduling, and
fallback attention kernels, so this is necessary but not sufficient.

Local smoke results on the 2-layer remapped checkpoint:

| Checkpoint | Mode | GPU | Load memory | KV cache | Output tok/s |
|---|---|---:|---:|---:|---:|
| `v4-flash-2layer-int-channel-vllm` | eager C=1 | GPU0 | 9.84 GiB | 453,810 tokens | 13.0 |
| `v4-flash-2layer-int-vllm` | eager C=1 | GPU0 | 10.03 GiB | 443,230 tokens | 13.2 |
| `v4-flash-2layer-int-channel-vllm` | compile/cudagraph C=12 | GPU1 | 9.84 GiB | 380,924 tokens | 259.9 |

Important local testing rule: GPU0 on `appmana-001` is the active GDM/Xorg
display GPU. Heavy vLLM smokes must use `--cuda-visible-devices 1` or another
non-display device; high `--gpu-memory-utilization` on GPU0 can disrupt the
desktop session.

Latest C=12 compile/cudagraph smoke on GPU1 had no Triton JIT monitor warnings
after activation. The output is nonsense because the local checkpoint is a
2-layer remap used only for kernel/lifecycle validation, not model quality.

---

## 5. PP=12 plumbing (chain integration)

Although strictly not "Ampere mitigations", these are required to land
the model on the chain and the same kernel patches need them to function:

* **`tb-chain-webhook`** mutates pods labeled `appmana.com/tb-chain-hostnet:
  primary` to:
  * `nodeSelector` includes `appmana.com/tb-chain-index = <worker-index>`
  * `hostNetwork=true`, `dnsPolicy=ClusterFirstWithHostNet`
  * `NCCL_SOCKET_IFNAME=tb-lo` injected into every container
  Combined with LWS `worker-index` → the chain-webhook → `chain-index`
  mapping, vLLM PP rank N is always on the physical chain box at
  position N. Adjacent ranks are physically adjacent on the TB chain.

* **`KeyError` cudagraph capture fix** — V4 cudagraph capture requires
  2D inputs at PP=12; we currently run with `--enforce-eager` (no
  cudagraph). This avoids the issue but loses captured-graph latency.

* **Disabled MTP head at PP=12** — `DeepSeekV4MTPModel` does not implement
  `SupportsPP`, asserts on engine init at PP > 1. Workaround: set
  `num_nextn_predict_layers=0` on the in-cluster checkpoint.

* **`/etc/hosts` patch** — under `hostNetwork`, Ubuntu's `127.0.1.1
  <hostname>` line makes Gloo's CPU-side TCPStore advertise `127.0.0.1`
  via `gethostname()`. The LWS entrypoint rewrites it to `${HOST_IP}
  $(hostname)`.

* **NCCL on `tb-lo`, control plane on switched LAN** — NCCL data plane
  uses the TB chain (1 hop adjacent ranks); Ray, Gloo, MASTER_ADDR,
  TCPStore, RAY_ADDRESS all use the switched-LAN HOST_IP (`enp38s0` on
  appmana-002, `eno1` on the rest — slated to homogenize via
  systemd-link rename in playbook_worker).

---

## 6. Performance characteristics (single-user, PP=12)

Bench config: 8 prompts × 256 input tokens × 64 output tokens, concurrency=1,
random dataset, greedy.

| Configuration | TTFT median | TPOT median | Output tok/s | ITL P99 |
|---|---|---|---|---|
| PP=12, NCCL on switched LAN (2.5 Gbps) | 2861 ms | **164 ms** | 4.9 | 180 ms |
| PP=12, NCCL on `tb-lo` (default) | 3009 ms | 188 ms | 4.3 | 543 ms |
| PP=12, NCCL on `tb-lo` + `Ring`+`NSOCKS=4×4` | 2838 ms | 170 ms | 4.6 | 371 ms |
| TP=2 PP=6, NCCL on `tb-lo` + `Ring`+`NSOCKS=4×4` | 2887 ms | 175 ms | 4.6 | 376 ms |

For comparison: native sm_89+ (Ada/Hopper) running V4-Flash without any of
these mitigations would have TPOT in the 30–60 ms range (rough estimate
based on FP8 tensor-core throughput vs sm_86's bf16 fallback).

### Why the chain doesn't accelerate single-user latency

The dominant cost is **per-rank compute**, not network bandwidth. At
PP=12 with concurrency=1, each token's wall-clock is approximately:

```
TPOT = sum_over_ranks(per_rank_compute) + (PP-1) * per_hop_latency
     ≈ 12 * 12 ms compute + 11 * 3 ms hop
     ≈ 144 ms + 33 ms ≈ 177 ms
```

The activation hand-off per layer is `hidden_size × bf16 = 8 KB` —
small enough that switched LAN's 2.5 Gbps and TB's ~10 Gbps deliver it
in roughly equal wall-clock time at this size. The arithmetic FP8
decode adds compute overhead inside *every* Triton kernel that reads the
KV cache, which dominates per-rank cost.

The chain bandwidth advantage materializes only with workloads that
generate larger collectives: longer prompts (KV-cache redistribution
on prefill), batched serve (>1 active request multiplies activation
size), or TP>1 (all-reduce over 8–32 KB tensors). TP=2 PP=6 was tested
to validate this — the higher TP all-reduce frequency on the chain still
leaves per-rank compute as the bottleneck and yields no headline win
for single-user.

---

## 7. Tracing & profiling

### NVIDIA Nsight Systems (`nsys`) — recommended for cross-rank timing

Multi-node distributed profiling is supported. The pattern for the chain:

1. Run `nsys profile -t cuda,nvtx,nccl,osrt,cublas -o /jit-local/trace/rank-${LWS_WORKER_INDEX}.nsys-rep --capture-range=cudaProfilerApi --capture-range-end=stop python -m vllm.entrypoints…` on each rank.
2. vLLM's `--profile` flag (or `VLLM_TORCH_PROFILER_DIR`) emits
   `cudaProfilerStart/Stop` markers — nsys honors these to bound the
   trace window.
3. Pull `.nsys-rep` files from each pod via `kubectl cp`, open them
   simultaneously in the Nsight Systems UI; alignment uses host-side
   `gettimeofday` timestamps.

Multi-node tip: synchronise host clocks via NTP (already configured on
chain nodes); per-rank traces correlate within milliseconds.

### OpenTelemetry PP spans

Enable `--otlp-traces-endpoint=... --collect-detailed-traces=pp` (or `all`)
to emit PP-specific spans from every rank. These spans are also wrapped in
vLLM profiling scopes so the same names show up in torch/NSight captures when
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` or `VLLM_NVTX_SCOPES_FOR_PROFILING=1` is
enabled.

The PP spans are:

* `vllm.pp.recv_intermediate.post` - posts activation receives from the
  previous PP stage.
* `vllm.pp.recv_intermediate.wait` - waits for the activation receive when
  the model first touches the intermediate tensor.
* `vllm.pp.model_execute` - rank-local model execution for the scheduler
  iteration.
* `vllm.pp.send_intermediate.post` and `.prev_wait` - posts this stage's
  activation send and waits for the previous send.
* `vllm.pp.sampled_token.send` / `.recv` - sampled-token handoff used by
  async PP scheduling.

Each span includes `vllm.pp.rank`, `vllm.pp.world_size`, first/last-rank flags,
the PP communication kind, scheduled context/generation token counts, and
tensor byte counts where available.

### Sampled-token handoff modes

`VLLM_PP_ASYNC_TOKEN_COMM` controls how the last PP stage returns sampled token
ids to earlier stages when async scheduling is enabled:

* `broadcast` - upstream PP-group collective.
* `p2p_fanout` / `p2p_first_only` - PyTorch NCCL P2P on the PP device group.
* `pynccl_fanout` / `pynccl_first_only` - vLLM's persistent PyNccl
  communicator, avoiding PyTorch's lazy 2-rank P2P communicator creation.
* `cpu_object_fanout` / `cpu_object_first_only` - CPU-group object payloads
  keyed by request id; safer when async PP stages have different request
  order/count, but it synchronizes the tiny sampled-token payload to CPU.

Start A/B testing with `p2p_first_only`, then `pynccl_first_only`. Use the
`cpu_object_*` modes only as a correctness probe for heterogeneous async PP
batches or MTP.

### Triton-side tracing

* `TRITON_PRINT_AUTOTUNING=1` — prints autotune decisions per kernel,
  useful for confirming kernel configs picked on sm_86.
* `TRITON_INTERPRET=1` — runs Triton kernels in interpreter mode; useful
  for debugging arithmetic FP8 helpers but ~1000× slower.
* **NVTX markers around Triton kernels** — wrap kernel launches with
  `torch.cuda.nvtx.range_push(name)` / `range_pop()`; `nsys` annotates
  the timeline with them. The fp8e4m3_arith helpers should be NVTX-
  tagged so we can see decode time separately from MMA time.
* **NCCL_PROFILER_PLUGIN** — `libnccl-profiler.so` (NCCL ≥2.23) emits a
  per-collective timing trace with topology context. *Currently missing
  from the image* (NCCL warns `Could not find: libnccl-net.so`); landing
  it would give per-collective bandwidth visibility without nsys overhead.
* `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,P2P` — coarse but free; every
  collective logs `[completed in X us, busbw Y GB/s]` already enabled.

### What we want to measure

The hypothesis to test is "per-rank arithmetic FP8 decode is the
dominant cost". Specifically, instrument:

1. The decode helper time vs the dot-product time inside the sparse-MLA
   decode kernel. If decode > dot, replacing decode with a software
   trick (e.g. precomputed lookup table in shared memory) is worth it.
2. The torch fallback for `dequantize_and_gather_k_cache` time vs its
   Triton arithmetic-decode equivalent. If torch is much slower, port
   the fallback to use the arithmetic helpers (Class 1 → Class 2 swap).
3. The PP send/recv time vs per-rank compute time. Confirms the
   compute-bound hypothesis.

---

## 8. Open optimization paths (not yet attempted)

* **Port cache_utils torch fallbacks to arithmetic Triton** — replace
  three torch fallbacks with kernels that use `fp8e4m3_decode_to_fp32` /
  `fp8e4m3_encode_from_fp32`. Removes the torch round-trip on every
  KV-cache I/O.
* **Precomputed FP8→fp32 lookup table** — for the decode helper, the
  arithmetic path is ~10 instructions per byte. A 256-entry constant
  table indexed by the byte gives the same answer in 1 load. Triton
  supports `tl.constexpr` arrays.
* **GLOO backend for PP send/recv** — patch `parallel_state.py` to
  create the PP group with `backend="gloo"`, so PP traffic uses
  ethernet (no chain-multi-hop) and TP traffic uses NCCL on `tb-lo`.
  Aligned with the chain topology.
* **`libnccl-net.so` plugin** — NCCL falls back to `NET/Socket` (warning
  "Could not find: libnccl-net.so"). Building/installing the upstream
  socket plugin or OFI plugin would likely improve the PP=12 small-message
  tail latency.
* **INT4/INT8 requantization tooling** — `quant_method=dsv4_int` now has
  loader support for AOT INT4 routed experts, INT8 attention/shared linears,
  DeepSeek V4 `expert_dtype="int4"` scale mapping, and MTP projection prefixes.
  The prior gibberish checkpoint should not be treated as representative until
  it is regenerated with the audited mapping below.

  ```bash
  python tools/ampere/dsv4_checkpoint_audit.py \
    --checkpoint /path/to/DeepSeek-V4-Flash \
    --fail-on-unknown

  python tools/ampere/dsv4_quant_sweep.py \
    --src /path/to/DeepSeek-V4-Flash \
    --roles routed_expert_mxfp4_weight,dense_fp8_weight,indexer_qk_fp8_weight,mtp_fp8_weight \
    --max-tensors 8 --max-rows 128 --device cuda:0
  ```

  On the local full V4-Flash snapshot the audit classified all 69,187 tensors
  with no unknowns or missing scales: 33,792 routed MXFP4 expert weights plus
  scales, 352 dense FP8 weights/scales, 21 indexer FP8 weights/scales, 2 MTP FP8
  projection weights/scales, and 853 preserved BF16/F32/I64 tensors. Sample
  sweeps show asymmetric UINT4 group-32 beats the current symmetric INT4 expert
  path by about 2.5 dB SNR, and asymmetric UINT8 block quantization beats
  symmetric INT8 on dense FP8 weights by roughly 0.4-1.6 dB. `mtp.0.h_proj` is
  much harder (~26 dB best sampled asymmetric UINT8), so leave MTP conversion
  behind the main model path until quality is measured end-to-end.

  A local 2-layer vLLM smoke checkpoint has been regenerated from
  `/home/administrator/inference/v4-flash-2layer-fp` into
  `/home/administrator/inference/v4-flash-2layer-int-vllm`:

  ```bash
  python tools/ampere/dsv4_requant_checkpoint.py \
    --src /home/administrator/inference/v4-flash-2layer-fp \
    --dst /home/administrator/inference/v4-flash-2layer-int-vllm \
    --device cuda:0 --overwrite
  ```

  The converter remapped the sparse source layer IDs `{0, 42}` to vLLM layers
  `{0, 1}`, wrote `quantization_config.quant_method="dsv4_int"`, and produced
  1,536 INT4 routed-expert tensors, 17 INT8 linear tensors, and 41 preserved
  tensors. Spot checks against the original FP4/FP8 tensors show routed-expert
  W4A16 SNR near 20 dB and dense W8A16 SNR near 40 dB.

  Current local verification:

  ```bash
  python -m pytest tests/quantization/test_dsv4_int.py -q

  CUDA_VISIBLE_DEVICES=1 python tools/ampere/dsv4_int_vllm_smoke.py \
    --model /home/administrator/inference/v4-flash-2layer-int-vllm \
    --cuda-visible-devices 1 \
    --num-prompts 12 --max-tokens 8 \
    --compile --allow-sparse-mla-warmup --allow-mhc-warmup
  ```

  The focused quant suite currently covers MXFP4->INT4 SNR, FP8->INT8 SNR,
  FP8->AllSpark-channel-INT8 SNR, asymmetric UINT4 quality search, DeepSeek V4
  scale-name mapping, checkpoint audit roles, checkpoint layer remap/index/
  config rewrite, CUDA Marlin INT4 MoE repack, and the `dsv4_int` runtime
  AllSpark W8A16 linear method against a BF16-dequant reference. The compiled
  C=12 vLLM smoke loads the quantized checkpoint, warms mHC and sparse MLA,
  captures CUDA graphs, allocates FP8 KV cache, and generates 96 tokens. The
  sparse MLA warmup now clamps its prefill warmup to `runner.max_model_len`,
  fixing the prior CUDA index assert when the smoke checkpoint was run with
  `max_model_len=512`.

  Local smoke matrix on the A5000 host:

  | Checkpoint | Mode | Settings | Result |
  | --- | --- | --- | --- |
  | `v4-flash-2layer-int-vllm` | eager | C=1, 8 output tokens, max len 512 | pass |
  | `v4-flash-2layer-int-vllm` | compile/cudagraph | C=12, 8 output tokens, max len 512 | pass, 96 output tokens |
  | `v4-flash-2layer-mtp-int` | eager | C=12, 8 output tokens, max len 512 | pass, validates MTP projection mapping at smoke level |
  | `v4-flash-4layer-int` | eager | C=12, 8 output tokens, max len 512, `gpu_memory_utilization=0.95` | pass, model load ~17.3 GiB, ~0.36 GiB KV left |
  | `v4-flash-4layer-int` | compile/cudagraph | C=1, max len 512, graph memory profiling enabled | fails before serving: no KV blocks after graph-memory reservation |
  | `v4-flash-4layer-int` | compile/cudagraph | C=1, max len 128, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | pass, but GPU memory reaches ~22.0 GiB |

  Remaining quant risks: the current runtime dequants INT8 linears to BF16 once
  after loading, so it validates checkpoint semantics but not final INT8 GEMM
  speed/memory. Sample sweeps on the 2-layer source show routed expert
  asymmetric UINT4 group-32 around 22.5 dB SNR vs symmetric INT4 around 20 dB;
  INT5/INT6 would buy much more quality if the memory budget permits. The
  indexer FP8->INT8 sample is only around 33-35 dB SNR, so validate sparse
  index recall before quantizing that path in production.
* **Cudagraph capture at PP=12** — currently disabled via `--enforce-
  eager` because cudagraph capture requires 2D input tensors at PP > 1.
  Fix the warmup harness to materialize 2D inputs and re-enable; expect
  10–30 ms TPOT savings from removed launch overhead.
