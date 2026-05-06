#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

image_repo="${IMAGE_REPO:-harbor.appmana.com/appmana/vllm-ampere}"
commit="${COMMIT:-$(git -C "${repo_root}" rev-parse --short=9 HEAD)}"
tag="${TAG:-${image_repo}:${commit}}"
cache_ref="${CACHE_REF:-${image_repo}:buildcache}"
builder="${BUILDER:-default}"

max_jobs="${MAX_JOBS:-2}"
nvcc_threads="${NVCC_THREADS:-8}"
torch_arch_list="${TORCH_CUDA_ARCH_LIST:-8.6}"

docker buildx build "${repo_root}" \
  --builder "${builder}" \
  --file "${repo_root}/docker/Dockerfile" \
  --target vllm-openai \
  --tag "${tag}" \
  --build-arg "max_jobs=${max_jobs}" \
  --build-arg "nvcc_threads=${nvcc_threads}" \
  --build-arg "torch_cuda_arch_list=${torch_arch_list}" \
  --build-arg "VLLM_BUILD_COMMIT=${commit}" \
  --build-arg "VLLM_IMAGE_TAG=${tag}" \
  --cache-from "type=registry,ref=${cache_ref}" \
  --cache-to "type=registry,ref=${cache_ref},mode=max" \
  "$@"
