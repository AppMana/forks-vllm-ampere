# SPDX-License-Identifier: Apache-2.0
"""Reproduce the PP mass-abort wedge locally on the DSV4-Flash-mini testbed.

Cluster incident 2026-06-10 22:56: 36 benchmark clients timed out and
disconnected mid-stream; every PP worker raised Gloo "Application timeout
caused pair closure"; the engine kept reporting phantom running requests with
vllm:generation_tokens_total frozen until the LWS was restarted.

This script starts a local PP=2 server on the mini checkpoint, opens N
streaming completions, hard-closes the client sockets mid-decode, then checks
whether the engine still serves a tiny probe and whether token throughput
resumes.

    .venv/bin/python tools/ampere/dsv4_abort_wedge_reproducer.py \
        --model /var/lib/inference/v4-flash-mini-int4mse
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request


def wait_ready(port: int, deadline_s: float) -> None:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ).read()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("server did not become ready")


def metrics(port: int) -> dict[str, float]:
    out: dict[str, float] = {}
    body = (
        urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10)
        .read()
        .decode()
    )
    for line in body.splitlines():
        for key in (
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:generation_tokens_total",
        ):
            if line.startswith(key):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def abortable_stream(port: int, model: str, abort_after_s: float) -> None:
    """Open a streaming completion and hard-close the socket mid-decode."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Count upward from 1, one number per line."}
            ],
            "max_tokens": 4000,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode() + payload
    sock.sendall(req)
    sock.settimeout(abort_after_s)
    end = time.time() + abort_after_s
    try:
        while time.time() < end:
            if not sock.recv(4096):
                break
    except socket.timeout:
        pass
    # Hard close: RST instead of FIN, mimicking a killed client.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/var/lib/inference/v4-flash-mini-int4mse")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--abort-after", type=float, default=4.0)
    parser.add_argument("--settle", type=float, default=15.0)
    parser.add_argument(
        "--executor-backend",
        default=None,
        help="vLLM distributed executor backend (e.g. ray) to match the "
        "cluster deployment; default lets vLLM choose (mp).",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", "0")
    env.setdefault("VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP", "0")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--trust-remote-code",
            "--pipeline-parallel-size",
            "2",
            "--max-model-len",
            "8192",
            "--kv-cache-dtype",
            "fp8",
            "--gpu-memory-utilization",
            "0.6",
            "--enforce-eager",
            "--port",
            str(args.port),
        ]
        + (
            ["--distributed-executor-backend", args.executor_backend]
            if args.executor_backend
            else []
        ),
        env=env,
        stdout=open("/tmp/dsv4_wedge_server.log", "wb"),
        stderr=subprocess.STDOUT,
    )
    try:
        wait_ready(args.port, 600)
        print("server ready; opening", args.streams, "streams")
        threads = [
            threading.Thread(
                target=abortable_stream, args=(args.port, args.model, args.abort_after)
            )
            for _ in range(args.streams)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"all {args.streams} clients hard-closed; settling {args.settle}s")
        time.sleep(args.settle)

        m1 = metrics(args.port)
        time.sleep(10)
        m2 = metrics(args.port)
        tok_delta = m2.get("vllm:generation_tokens_total", 0) - m1.get(
            "vllm:generation_tokens_total", 0
        )
        print("metrics after aborts:", m2, "token delta over 10s:", tok_delta)

        probe_ok = False
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": args.model,
                        "messages": [{"role": "user", "content": "Say ok."}],
                        "max_tokens": 4,
                        "temperature": 0,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=90).read()
            probe_ok = True
        except Exception as exc:
            print("probe failed:", exc)

        running = m2.get("vllm:num_requests_running", 0)
        if probe_ok and running == 0:
            print("RESULT: RECOVERED (aborts cleaned up, probe served)")
            return 0
        if probe_ok:
            print(f"RESULT: PARTIAL (probe ok but {running} phantom running)")
            return 1
        print("RESULT: WEDGED (probe failed)")
        return 2
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
