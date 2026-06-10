# AppMana vLLM Ampere Fork

This fork is AppMana's DeepSeek V4 enablement stack for Ampere GPUs. It is
focused on running DeepSeek V4 quantized checkpoints on RTX 3090-class
hardware with pipeline parallelism, native MTP speculative decode, sparse MLA,
and controlled cluster benchmarking.

## Branch Map

`vllm-ampere` is the production branch. It must point at the commit currently
running on the cluster unless the cluster image is intentionally rolled
forward.

```text
production branch: vllm-ampere
live commit:       7c5968b7255ca7c9740f29841105070cef418edb
live image:        harbor.appmana.com/appmana/vllm-ampere:bench-ppdecodefill-7c5968b72-20260527221421
live subject:      Tolerate async placeholder underflow
author:            doctorpangloss <2229300+doctorpangloss@users.noreply.github.com>
```

All work after the live commit belongs on experiment branches. Current local
branch organization:

| Branch | Status | Tip | Meaning |
|---|---|---:|---|
| `vllm-ampere` | production | `7c5968b72` | Exact live cluster code. |
| `dsv4-ampere` | legacy production alias | `7c5968b72` | Same live code; keep until callers move to `vllm-ampere`. |
| `ampere-develop` | legacy alias | `7c5968b72` | Same live code. |
| `ampere-openwebui-progress` | legacy alias | `7c5968b72` | Same live code. |
| `experiment/nccl-upgrade-live-7c5968b` | active experiment | `7c5968b72+local` | Runtime overlay for NCCL package canary from the live image. |
| `ampere-lmcache-experimental` | experiment/archive | `9a3d79c43` | LMCache grouped KV work after the live production line. |
| `ampere-upstream-rework` | experiment/archive | `3d95f3150` | Upstream rework line, not live. |
| `dsv4-ampere-upstream-rework` | experiment/archive | `3a4ea49dc` | Later upstream rework, not live. |
| `backup/dsv4-ampere-upstream-merge-c0b5e49a` | backup | `c0b5e49a0` | Backup of the upstream-merge/int4-int8 line. |
| `ampere-semantic-baseline-07898` | historical baseline | `07898f8e6` | Older semantic baseline image family. |

Every AppMana branch above is doctorpangloss-authored at its tip. Upstream
`main` is not AppMana production code.

## What Differs From Upstream

### DeepSeek V4 on Ampere

- Adds DeepSeek V4 runtime support for Ampere (`sm_86`) where upstream paths
  were primarily authored and tuned around newer NVIDIA targets.
- Adds runtime feature detection and Ampere-safe fallbacks for DeepSeek V4
  sparse MLA, MHC, and quantized projection paths.
- Adds Triton fallbacks for MHC and sparse MLA paths used by DSV4 on Ampere.
- Reduces Triton/JIT shape specialization for prompt and decode shapes so
  production traffic does not repeatedly compile for incidental sizes.

### Quantized DSV4 Checkpoints

- Supports AppMana DeepSeek V4 quantized checkpoints including:
  - `appmana/deepseek-v4-mxfp4-int8`
  - `appmana/deepseek-v4-int4-int8`
- Handles DSV4 checkpoint scale-name variants for int and mxfp4/int8 layouts.
- Restores and extends expert scale mapping for quantized MTP and MoE weights.
- Keeps QAT expert weights in their checkpoint-native format where possible.

### Native DeepSeek V4 MTP

- Adds DSV4-native MTP support using the checkpoint's `mtp.*` layers rather
  than an external EAGLE draft model.
- Implements DSV4-specific MTP layers with separate `e_proj` / `h_proj`, HC-head
  residual handling, V4 checkpoint remapping, and DSV4 quantization config
  derivation.
- Supports pipeline parallel serving by passing draft/target state across PP
  stages and by fixing scheduler/model-runner handoff cases that upstream MTP
  did not cover for DSV4 PP.
- Adds decode-path fixes for sparse MLA/MTP verification, cudagraph shape
  selection, one-token rows, and forced-reject/debug paths.
- MTP is currently used for decode; prefill is intentionally handled by the
  target model path.

### Sparse MLA and MHC

- Adds and wires DSV4 sparse MLA direct decode paths for Ampere.
- Adds sparse MLA prefill matmul support and direct-kernel warmup controls.
- Adds MHC Triton paths (`pre`, `head`, `post`) and synchronization controls
  used by the Ampere deployment.
- Adds warmup paths for observed live DSV4 MTP/PP token shapes to avoid
  request-time JIT stalls.

### Prefix Caching and LMCache

- Enables vLLM prefix caching with LMCache external cache for PP deployments.
- Fixes LMCache grouped KV metadata for DSV4's heterogeneous compressed MLA KV
  layout.
- Stores and retrieves physical grouped KV shapes rather than assuming every
  group has the logical chunk length.
- Fixes PP remote cache initialization for non-uniform grouped MLA chunks so
  every PP rank stores and retrieves its own Redis keys.
- Validated behavior after Redis flush:
  - Redis keys are created for every worker id: `@12@0@` through `@12@11@`.
  - Second identical request retrieves `256/256` remote tokens on all PP ranks.

### Production and Debuggability

- Adds LWS/Ray/PP chain operational fixes for 12 single-GPU workers.
- Adds PP trace and timing controls, cudagraph metrics, profiler wiring, and
  JIT monitor controls used during production diagnosis.
- Adds build support for cached BuildKit and `sccache`-backed Ampere images.

## Tested Configurations

### Current LWS Deployment

```text
model: appmana/deepseek-v4-mxfp4-int8
image: harbor.appmana.com/appmana/vllm-ampere:bench-ppdecodefill-7c5968b72-20260527221421
hardware: 12 x RTX 3090-class Ampere GPUs, one GPU per worker
parallelism: PP=12, TP=1
max_model_len: 81920
max_num_seqs: 1
max_num_batched_tokens: 4096
kv_cache_dtype: fp8
kv_cache_memory_bytes: 5637144576
chunked_prefill: enabled
prefix_caching: disabled unless explicitly canaried
LMCache: disabled unless explicitly canaried
speculative_config: {"method":"mtp","num_speculative_tokens":4}
tool calling: --enable-auto-tool-choice --tool-call-parser deepseek_v4
NCCL: 2.28.9+cuda13.0
NCCL transport: sockets over Thunderbolt tb-lo
```

Key runtime environment:

```text
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_TRITON_MLA_SPARSE=1
VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1
VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL=1
VLLM_PP_ASYNC_TOKEN_COMM=pynccl_fanout
VLLM_PP_MAX_CONCURRENT_BATCHES=12
VLLM_PP_LAYER_PARTITION=3,3,3,3,4,4,4,4,4,4,4,3
VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP=1
VLLM_ENABLE_DEEPSEEK_V4_REQUEST_PREP_WARMUP=1
VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_DIRECT_KERNEL_WARMUP=1
VLLM_MHC_PRE_TRITON=1
VLLM_MHC_HEAD_TRITON=1
VLLM_MHC_POST_TRITON=1
NCCL_SOCKET_IFNAME=tb-lo
NCCL_IB_DISABLE=1
```

### 10-Node PP Candidate

DSV4 can be canaried on 10 nodes by reducing KV cache and using a 10-way layer
split:

```text
parallelism: PP=10, TP=1
VLLM_PP_LAYER_PARTITION=4,4,4,4,4,4,5,5,5,4
```

This is not the current production shape. It is acceptable only with a smaller
KV budget/context target than the current 81,920-token production profile.

### Production Milestone Tag

`dsv4-lws-pre-mtp-prod-20260518` marks the pre-MTP production milestone:

```text
models: appmana/deepseek-v4-mxfp4-int8, appmana/deepseek-v4-int4-int8
status: working in production on LWS
parallelism: PP>=6, TP>1 tested
concurrency: C=12 tested
context: 262k tested
```

### LMCache Smoke Test

Synthetic document bundle smoke test after the PP remote cache fix:

```text
prompt tokens: 6463
completion tokens: 96
run 1 wall time: 47.072s
run 2 wall time: 8.658s
prefix_cache_hits_total: 6144
external_prefix_cache_hits_total: 256
prompt_tokens_cached_total: 6400
request errors: 0
```

The request-level external hit metric reports logical tokens. PP-rank logs
confirmed all 12 ranks retrieved `256/256` remote tokens.

## Benchmarking

The headline benchmark for this fork means:

```text
parallel requests: 48
context target:    16k tokens per request
model:             appmana/deepseek-v4-mxfp4-int8
endpoint:          live OpenAI-compatible vLLM service
metric:            post-TTFT decode tok/s per stream and aggregate output tok/s
```

Use the AppMana benchmark harness from the appmana repo, not ad-hoc curl loops.
Run it from inside the cluster or from the leader pod to avoid API server
port-forward artifacts:

```bash
appmana-management/src/appmana_management/scripts/llm_openai_bench.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model appmana/deepseek-v4-mxfp4-int8 \
  --label dsv4-48x16k \
  --reasoning-mode thinking \
  --context-tokens 16000 \
  --concurrency 48 \
  --requests 48 \
  --max-tokens 2048 \
  --stream \
  --unique-prompts \
  --out /tmp/dsv4-48x16k-$(date -u +%Y%m%dT%H%M%SZ).json
```

For an NCCL upgrade canary, keep the model config and benchmark command fixed.
Only change the image/runtime NCCL package. A valid canary report must include:

- branch and image tag;
- `torch.cuda.nccl.version()`;
- vLLM startup `NCCL version ...` banner;
- `NCCL_DEBUG=INFO` transport lines proving socket mode is still `NET/Socket`;
- p50/p95 TTFT;
- post-TTFT tok/s per stream;
- aggregate output tok/s;
- any correctness or stuck-request failures.

For the current PP=12 deployment, benchmark at `--max-concurrency 1` unless
the LWS manifest is changed from `--max-num-seqs 1`.
