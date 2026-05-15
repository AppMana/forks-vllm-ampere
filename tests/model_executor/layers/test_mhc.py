# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import mhc


def _mhc_inputs(num_tokens: int = 2):
    hc_mult = 4
    hidden_size = 8
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32)
    return residual, fn, hc_scale, hc_base


def test_mhc_pre_torch_fallback_synchronizes_before_return(monkeypatch):
    calls = []

    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc, "_synchronize_mhc_torch_fallback", lambda: calls.append("sync"))
    monkeypatch.setenv("VLLM_MHC_DEBUG_TIMINGS", "0")

    post_mix, comb_mix, layer_input = mhc.mhc_pre(
        *_mhc_inputs(),
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=1,
    )

    assert calls == ["sync"]
    assert post_mix.shape == (2, 4, 1)
    assert comb_mix.shape == (2, 4, 4)
    assert layer_input.shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 3])
@pytest.mark.parametrize("hidden_size", [128, 7168])
def test_mhc_pre_triton_matches_torch_reference(monkeypatch, num_tokens, hidden_size):
    hc_mult = 4
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn(
        hc_mult3, hc_mult * hidden_size, device="cuda", dtype=torch.float32
    )
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, device="cuda", dtype=torch.float32)

    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc, "_synchronize_mhc_torch_fallback", lambda: None)
    monkeypatch.setenv("VLLM_MHC_PRE_TRITON", "0")
    expected = mhc.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=1,
    )

    actual = mhc._mhc_pre_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=1,
    )
    torch.cuda.synchronize()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-2, atol=4e-3)


def test_mhc_post_torch_fallback_synchronizes_before_return(monkeypatch):
    calls = []

    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc, "_synchronize_mhc_torch_fallback", lambda: calls.append("sync"))

    out = mhc.mhc_post(
        x=torch.randn(2, 8, dtype=torch.bfloat16),
        residual=torch.randn(2, 4, 8, dtype=torch.bfloat16),
        post_layer_mix=torch.randn(2, 4, 1, dtype=torch.float32),
        comb_res_mix=torch.randn(2, 4, 4, dtype=torch.float32),
    )

    assert calls == ["sync"]
    assert out.shape == (2, 4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 3])
@pytest.mark.parametrize("hidden_size", [128, 7168])
def test_mhc_post_triton_matches_torch_reference(monkeypatch, num_tokens, hidden_size):
    hc_mult = 4
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    post = torch.randn(num_tokens, hc_mult, 1, device="cuda", dtype=torch.float32)
    comb = torch.randn(num_tokens, hc_mult, hc_mult, device="cuda", dtype=torch.float32)

    expected = (
        torch.einsum("...ij,...ih->...jh", comb, residual.to(torch.float32))
        + post * x.unsqueeze(-2).to(torch.float32)
    ).to(torch.bfloat16)
    actual = torch.empty_like(residual)

    mhc._mhc_post_triton(x, residual, post, comb, actual)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mhc_triton_public_paths_do_not_force_sync(monkeypatch):
    hc_mult = 4
    hidden_size = 128
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    num_tokens = 12

    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(
        mhc,
        "_synchronize_mhc_torch_fallback",
        lambda: pytest.fail("Triton MHC path should not force stream sync"),
    )
    monkeypatch.setenv("VLLM_MHC_PRE_TRITON", "1")
    monkeypatch.setenv("VLLM_MHC_POST_TRITON", "1")
    monkeypatch.setenv("VLLM_MHC_HEAD_TRITON", "1")

    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    fn_pre = torch.randn(
        hc_mult3, hc_mult * hidden_size, device="cuda", dtype=torch.float32
    )
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, device="cuda", dtype=torch.float32)

    post_mix, comb_mix, layer_input = mhc.mhc_pre(
        residual,
        fn_pre,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=1,
    )
    out_post = mhc.mhc_post(layer_input, residual, post_mix, comb_mix)

    out_head = torch.empty(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    mhc._hc_head_fused_kernel(
        hs_flat=residual,
        fn=torch.randn(
            hc_mult, hc_mult * hidden_size, device="cuda", dtype=torch.float32
        ),
        hc_scale=torch.randn(1, device="cuda", dtype=torch.float32),
        hc_base=torch.randn(hc_mult, device="cuda", dtype=torch.float32),
        out=out_head,
        hidden_size=hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=hc_mult,
    )
    torch.cuda.synchronize()

    assert out_post.shape == residual.shape
    assert out_head.shape == (num_tokens, hidden_size)


def test_hc_head_torch_fallback_synchronizes_before_return(monkeypatch):
    calls = []
    out = torch.empty(2, 8, dtype=torch.bfloat16)

    monkeypatch.setattr(mhc, "_should_use_mhc_torch_fallback", lambda: True)
    monkeypatch.setattr(mhc, "_synchronize_mhc_torch_fallback", lambda: calls.append("sync"))

    mhc._hc_head_fused_kernel(
        hs_flat=torch.randn(2, 4, 8, dtype=torch.bfloat16),
        fn=torch.randn(4, 32, dtype=torch.float32),
        hc_scale=torch.randn(1, dtype=torch.float32),
        hc_base=torch.randn(4, dtype=torch.float32),
        out=out,
        hidden_size=8,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=4,
    )

    assert calls == ["sync"]
    assert out.shape == (2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 3])
@pytest.mark.parametrize("hidden_size", [128, 7168])
def test_hc_head_triton_matches_torch_reference(num_tokens, hidden_size):
    hc_mult = 4
    hs_flat = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn(
        hc_mult, hc_mult * hidden_size, device="cuda", dtype=torch.float32
    )
    hc_scale = torch.randn(1, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(hc_mult, device="cuda", dtype=torch.float32)
    expected = torch.empty(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(expected)

    mhc._hc_head_fused_reference(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        expected,
        hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=hc_mult,
    )
    mhc._hc_head_triton(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        actual,
        hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=hc_mult,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=4e-3)


def test_mhc_fallback_stream_sync_is_default(monkeypatch):
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))

    monkeypatch.delenv("VLLM_MHC_TORCH_FALLBACK_SYNC_MODE", raising=False)
    monkeypatch.setenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "1")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("device"))

    mhc._synchronize_mhc_torch_fallback()

    assert calls == ["stream"]


def test_mhc_fallback_sync_can_be_disabled(monkeypatch):
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))

    monkeypatch.setenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "0")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("device"))

    mhc._synchronize_mhc_torch_fallback()

    assert calls == []


@pytest.mark.parametrize("mode", ["none", "device"])
def test_mhc_fallback_sync_modes(monkeypatch, mode):
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))

    monkeypatch.setenv("VLLM_MHC_TORCH_FALLBACK_SYNCHRONIZE", "1")
    monkeypatch.setenv("VLLM_MHC_TORCH_FALLBACK_SYNC_MODE", mode)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("device"))

    mhc._synchronize_mhc_torch_fallback()

    assert calls == ([] if mode == "none" else ["device"])
