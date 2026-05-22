#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://10.152.184.214:8080}"
MODEL="${MODEL:-appmana/deepseek-v4-int4-int8}"
NAMESPACE="${NAMESPACE:-inference}"
POD="${POD:-vllm-deepseek-v4-0}"
export ENDPOINT MODEL

python3 - <<'PY'
import json
import os
import time

import requests

endpoint = os.environ.get("ENDPOINT", "http://10.152.184.214:8080").rstrip("/")
model = os.environ.get("MODEL", "appmana/deepseek-v4-int4-int8")
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "What is 2+2? Answer with one number only.",
        }
    ],
    "max_tokens": 32,
    "temperature": 0,
}
started = time.time()
response = requests.post(
    f"{endpoint}/v1/chat/completions", json=payload, timeout=180
)
elapsed = time.time() - started
print(f"status={response.status_code} elapsed_s={elapsed:.3f}")
try:
    data = response.json()
    print(json.dumps(data, indent=2)[:4000])
except Exception:
    print(response.text[:4000])
PY

echo
echo "---- recent MTP trace / spec-decode logs ----"
kubectl -n "${NAMESPACE}" logs "${POD}" --tail=500 \
  | rg 'DSV4_MTP_TRACE|DSV4_MTP_VERIFY|DSV4_MTP_DRAFT|SpecDecoding|Dropping DeepSeek V4 MTP|DeepSeekV4MTPModel|speculative_config' || true
