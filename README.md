# dsv4-ampere

This branch is AppMana's DeepSeek V4 enablement stack for Ampere GPUs,
currently maintained on `dsv4-ampere`. It is a vLLM fork focused on running
DeepSeek V4 quantized checkpoints on RTX 3090-class hardware with pipeline
parallelism, native MTP speculative decode, sparse MLA, and prefix-cache reuse.

The deployed production image built from this branch is:

```text
harbor.appmana.com/appmana/vllm-ampere:lmcache-v8-pp-remote-9a3d79c43
```

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
image: harbor.appmana.com/appmana/vllm-ampere:lmcache-v8-pp-remote-9a3d79c43
hardware: 12 x RTX 3090-class Ampere GPUs, one GPU per worker
parallelism: PP=12, TP=1
max_model_len: 81920
max_num_seqs: 1
max_num_batched_tokens: 3077
kv_cache_dtype: fp8
kv_cache_memory_bytes: 2818572288
chunked_prefill: enabled
prefix_caching: enabled
LMCache: Redis remote, chunk_size=256, GPU connector v3
speculative_config: {"method":"mtp","num_speculative_tokens":4}
tool calling: --enable-auto-tool-choice --tool-call-parser deepseek_v4
```

Key runtime environment:

```text
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_TRITON_MLA_SPARSE=1
VLLM_TRITON_MLA_SPARSE_ALLOW_CUDAGRAPH=1
VLLM_TRITON_MLA_SPARSE_MATMUL_PREFILL=1
VLLM_PP_ASYNC_TOKEN_COMM=pynccl_fanout
VLLM_PP_MAX_CONCURRENT_BATCHES=12
VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP=1
VLLM_ENABLE_DEEPSEEK_V4_REQUEST_PREP_WARMUP=1
VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_DIRECT_KERNEL_WARMUP=1
VLLM_MHC_PRE_TRITON=1
VLLM_MHC_HEAD_TRITON=1
VLLM_MHC_POST_TRITON=1
LMCACHE_EXTRA_CONFIG={"save_only_first_rank":false,"remote_enable_mla_worker_id_as0":false}
```

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

For serving benchmarks against the production OpenAI-compatible endpoint, prefer:

```bash
python -m vllm.entrypoints.cli.main bench serve
```

Use `--dataset-name custom` with JSONL when testing real tasks over varied
contexts. Each JSONL row should contain:

```json
{"prompt": "context plus task", "output_tokens": 256}
```

Useful benchmark modes:

- `custom`: real task prompts with arbitrary document contexts.
- `prefix_repetition`: isolates prefix-cache behavior by sharing prefixes
  across requests.
- `random`: synthetic load for token-throughput and scheduler stress.
- `speed_bench`: broader task categories when an external benchmark suite is
  desired.

For the current PP=12 deployment, benchmark at `--max-concurrency 1` unless
the LWS manifest is changed from `--max-num-seqs 1`.
