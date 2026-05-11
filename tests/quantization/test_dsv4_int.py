# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.ampere.dsv4_checkpoint_audit import classify_tensor, matched_scale_name
from tools.ampere.dsv4_requant_checkpoint import convert_checkpoint
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.dsv4_int import (
    Dsv4Int4MoEMethod,
    Dsv4Int8LinearMethod,
    Dsv4IntConfig,
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
    dequantize_allspark_uint8_w8a16,
    dequantize_int4_w4a16,
    dequantize_int8_w8a16,
    dequantize_uint4_asym_w4a16,
    quantize_fp32_to_uint4_asym_w4a16,
    requantize_fp8_to_allspark_uint8_w8a16,
    requantize_fp8_to_int8_w8a16,
    requantize_mxfp4_to_int4_w4a16,
)
from vllm.model_executor.models.deepseek_v4 import _make_deepseek_v4_weights_mapper


def _snr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    noise = (reference.float() - actual.float()).norm()
    return (20 * torch.log10(reference.float().norm() / noise)).item()


def _pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (((high & 0x0F) << 4) | (low & 0x0F)).view(torch.int8)


def test_dsv4_int_quantization_config_registered():
    assert get_quantization_config("dsv4_int") is Dsv4IntConfig
    cfg = Dsv4IntConfig.from_config({"quant_method": "dsv4_int"})
    assert cfg.get_name() == "dsv4_int"
    assert cfg.weight_block_size == (128, 128)

    channel_cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    assert channel_cfg.int8_weight_strategy == "channel"
    assert channel_cfg.weight_block_size is None


def test_mxfp4_to_int4_requant_roundtrip():
    torch.manual_seed(0)
    rows = 16
    cols = 256
    nibbles = torch.randint(0, 16, (rows, cols), dtype=torch.uint8)
    packed = _pack_nibbles(nibbles)
    scale_bytes = torch.randint(120, 134, (rows, cols // 32), dtype=torch.uint8)

    result = requantize_mxfp4_to_int4_w4a16(packed, scale_bytes)
    int4_dequant = dequantize_int4_w4a16(
        result["qweight_packed"], result["scales"], group_size=32
    )

    fp4 = _e2m1_nibble_to_fp32(_unpack_int4_pairs(packed))
    scale = _e8m0_to_fp32_scale(scale_bytes)
    fp4_truth = (fp4.reshape(rows, -1, 32) * scale.unsqueeze(-1)).reshape(rows, cols)

    assert _snr_db(fp4_truth, int4_dequant) > 17.0


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_fp8_to_int8_requant_roundtrip():
    torch.manual_seed(1)
    n = 256
    k = 256
    source = (torch.randn(n, k) * 0.5).clamp(-4, 4)
    weight_fp8 = source.to(torch.float8_e4m3fn)
    scale_bytes = torch.randint(123, 131, (2, 2), dtype=torch.uint8)
    scale_e8m0 = scale_bytes.view(torch.float8_e8m0fnu)

    result = requantize_fp8_to_int8_w8a16(weight_fp8, scale_e8m0)
    int8_dequant = dequantize_int8_w8a16(
        result["qweight"], result["scales"], block_size=(128, 128)
    )

    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    fp8_truth = weight_fp8.to(torch.float32) * scale_full[:n, :k]

    assert _snr_db(fp8_truth, int8_dequant) > 30.0


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_fp8_to_allspark_uint8_channel_requant_roundtrip():
    torch.manual_seed(11)
    n = 256
    k = 256
    source = (torch.randn(n, k) * 0.5).clamp(-4, 4)
    weight_fp8 = source.to(torch.float8_e4m3fn)
    scale_e8m0 = torch.randint(123, 131, (2, 2), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )

    result = requantize_fp8_to_allspark_uint8_w8a16(weight_fp8, scale_e8m0)
    int8_dequant = dequantize_allspark_uint8_w8a16(
        result["qweight"], result["scales"]
    )

    scale = _e8m0_to_fp32_scale(scale_e8m0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    fp8_truth = weight_fp8.to(torch.float32) * scale_full[:n, :k]

    assert result["qweight"].dtype is torch.uint8
    assert result["scales"].shape == (n,)
    assert _snr_db(fp8_truth, int8_dequant) > 30.0


def test_asymmetric_uint4_improves_biased_groups():
    x = torch.linspace(-0.1, 1.0, 256, dtype=torch.float32).reshape(8, 32)

    asym = quantize_fp32_to_uint4_asym_w4a16(x, group_size=32)
    asym_dequant = dequantize_uint4_asym_w4a16(
        asym["qweight_packed"],
        asym["scales"],
        asym["zero_points"],
        group_size=32,
    )

    scale = x.abs().amax(dim=-1, keepdim=True).clamp(
        min=torch.finfo(torch.float32).tiny
    ) / 7.0
    sym_dequant = torch.round(x / scale).clamp(-8, 7) * scale

    asym_rmse = torch.sqrt(torch.mean((x - asym_dequant.float()) ** 2))
    sym_rmse = torch.sqrt(torch.mean((x - sym_dequant) ** 2))
    assert asym_rmse < sym_rmse


def test_deepseek_v4_int4_mapper_keeps_expert_scale_suffix():
    mapper = _make_deepseek_v4_weights_mapper("int4")
    names = mapper.apply_list(
        [
            "layers.0.ffn.experts.0.w1.scale",
            "layers.0.attn.wq_a.scale",
        ]
    )

    assert names == [
        "model.layers.0.ffn.experts.0.w1.weight_scale",
        "model.layers.0.attn.wq_a.weight_scale_inv",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_allspark_channel_int8_linear_method_matches_dequant_reference():
    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "allspark_w8a16_gemm"
    ):
        pytest.skip("AllSpark W8A16 op is not available")
    props = torch.cuda.get_device_properties()
    sm_version = props.major * 10 + props.minor
    if sm_version < 80 or sm_version >= 90:
        pytest.skip("AllSpark Ampere path only runs on sm_8x")

    torch.manual_seed(12)
    m = 12
    n = 1024
    k = 1024
    dtype = torch.bfloat16
    device = torch.device("cuda")
    weight = torch.randn(n, k, device=device, dtype=torch.float32) * 0.02
    scale = weight.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q_signed = torch.round(weight / scale.unsqueeze(1)).clamp(-128, 127)
    q_biased = (q_signed.to(torch.int16) + 128).to(torch.uint8)

    class FakeLayer(torch.nn.Module):
        pass

    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(q_biased, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(scale.to(dtype), requires_grad=False)

    cfg = Dsv4IntConfig.from_config(
        {
            "quant_method": "dsv4_int",
            "config_groups": {
                "linears_w8a16": {
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                }
            },
        }
    )
    method = Dsv4Int8LinearMethod(cfg, "model.layers.0.attn.wq_a")
    assert method._try_process_allspark(layer)

    x = torch.randn(m, k, device=device, dtype=dtype) * 0.02
    actual = method.apply(layer, x)
    ref_weight = dequantize_allspark_uint8_w8a16(q_biased, scale.to(dtype)).to(dtype)
    reference = torch.nn.functional.linear(x, ref_weight)
    torch.cuda.synchronize()

    assert _snr_db(reference, actual) > 45.0


def test_checkpoint_audit_classifies_deepseek_v4_precision_roles():
    assert classify_tensor("layers.2.ffn.experts.0.w1.weight", "I8") == (
        "routed_expert_mxfp4_weight",
        "quantize_asym_int4_awq_candidate",
    )
    assert classify_tensor("layers.2.attn.indexer.wq_b.weight", "F8_E4M3") == (
        "indexer_qk_fp8_weight",
        "measure_recall_then_quantize",
    )
    assert classify_tensor("layers.2.attn.compressor.wkv.weight", "BF16") == (
        "preserved_precision_tensor",
        "preserve",
    )
    assert classify_tensor("mtp.0.h_proj.scale", "F8_E8M0") == (
        "mtp_fp8_scale",
        "quantize_int8_w8a16_candidate",
    )
    assert (
        matched_scale_name("layers.0.attn.wq_a.weight")
        == "layers.0.attn.wq_a.scale"
    )


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"),
    reason="requires torch float8 dtypes",
)
def test_requant_checkpoint_rewrites_remapped_layers_and_quant_config(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    expert_nibbles = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
    expert_packed = _pack_nibbles(expert_nibbles)
    fp8_weight = torch.randn(130, 129).clamp(-2, 2).to(torch.float8_e4m3fn)
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": expert_packed,
        "layers.0.ffn.experts.0.w1.scale": torch.full(
            (4, 2), 127, dtype=torch.uint8
        ),
        "layers.42.attn.wq_a.weight": fp8_weight,
        "layers.42.attn.wq_a.scale": torch.full(
            (2, 2), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "layers.42.attn.attn_sink": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file(tensors, str(src / shard_name))
    (src / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "num_hidden_layers": 2,
                "expert_dtype": "fp4",
            }
        )
    )
    (src / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": "0"},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
    )

    convert_checkpoint(
        src,
        dst,
        device="cpu",
        out_scale_dtype=torch.bfloat16,
        overwrite=False,
        layer_remap=None,
    )

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["expert_dtype"] == "int4"
    assert cfg["quantization_config"]["quant_method"] == "dsv4_int"
    assert cfg["num_hidden_layers"] == 2

    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert "layers.42.attn.wq_a.weight" not in index["weight_map"]
    assert "layers.1.attn.wq_a.weight" in index["weight_map"]

    with safe_open(dst / shard_name, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert "layers.42.attn.wq_a.weight" not in keys
        assert "layers.1.attn.wq_a.weight" in keys
        assert "layers.1.attn.wq_a.scale" in keys
        assert "layers.1.attn.attn_sink" in keys
        assert handle.get_tensor("layers.0.ffn.experts.0.w1.weight").dtype is torch.int8
        assert handle.get_tensor("layers.0.ffn.experts.0.w1.scale").dtype is torch.bfloat16
        assert handle.get_tensor("layers.1.attn.wq_a.weight").dtype is torch.int8
        assert handle.get_tensor("layers.1.attn.wq_a.scale").dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int4_moe_marlin_repack_smoke():
    torch.manual_seed(2)
    num_experts = 2
    size_k = 128
    size_n = 256
    weight = torch.randint(
        0,
        256,
        (num_experts, size_n, size_k // 2),
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.int8)

    repacked = Dsv4Int4MoEMethod._repack_int4_for_marlin(
        weight,
        size_n=size_n,
        size_k=size_k,
    )
    torch.cuda.synchronize()

    assert repacked.shape[0] == num_experts
    assert repacked.dtype == torch.int32
    assert repacked.is_cuda
