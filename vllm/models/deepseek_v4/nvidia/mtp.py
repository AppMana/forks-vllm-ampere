# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MTP draft model for DeepSeek V4 (internal codename: DeepseekV4).

Split from ``deepseek_mtp.py`` because the V4 architecture introduces several
pieces that have no analogue in V3/V32:
  * separate ``e_proj`` / ``h_proj`` with fp8 linear quantization (instead of
    the fused ``eh_proj``);
  * ``hc_head`` hypercompressed vocab projection applied in ``compute_logits``;
  * ``DeepseekV4DecoderLayer`` with its own aux-stream management;
  * V4-specific checkpoint weight-name remapping in ``load_weights``.
"""

import os
import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import is_pp_missing_parameter, maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)


def _use_serial_mtp_mhc() -> bool:
    return os.getenv("VLLM_DSV4_MTP_SERIAL_MHC", "0") != "0"

# MoE expert scales are fused into per-layer w13/w2 tensors. The exact
# parameter suffix depends on which FusedMoE method handles the experts:
# - fp4 experts (Mxfp4MoEMethod) register ``w{1,2,3}_weight_scale``;
# - fp8 experts (Fp8MoEMethod with block_quant=True) register
#   ``w{1,2,3}_weight_scale_inv``.
# Other FP8 linear scales (including shared experts) always use
# ``.weight_scale_inv``. Mirrors the per-instance mapper built by
# ``_make_deepseek_v4_weights_mapper`` in deepseek_v4.py.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def should_compile_deepseek_v4_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    return speculative_config is None or not speculative_config.enforce_eager


def _mtp_scale_can_be_mapped_later(
    name: str,
    stacked_params_mapping: list[tuple[str, str, int]],
) -> bool:
    """Whether a checkpoint scale name is valid before later name remaps.

    Some checkpoint scale tensors use the original checkpoint names
    (for example ``attn.wq_a.scale`` or ``shared_experts.w1.scale``), while
    runtime parameters are registered under fused names such as
    ``fused_wqa_wkv.weight_scale_inv`` or ``gate_up_proj.weight_scale_inv``.
    Do not reject those names before the stacked/renamed loader path runs.
    """
    if any(weight_name in name for _, weight_name, _ in stacked_params_mapping):
        return True
    return ".shared_experts.w2." in name


class DeepSeekV4MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        if quant_config is None:
            hf_quant_config = getattr(config, "quantization_config", None)
            quantization = getattr(vllm_config.model_config, "quantization", None)
            if quantization is None and isinstance(hf_quant_config, dict):
                quantization = hf_quant_config.get("quant_method")
            if quantization in ("dsv4_int", "dsv4_mxfp4_int8"):
                quant_config_cls = get_quantization_config(quantization)
                quant_config = quant_config_cls.from_config(hf_quant_config or {})
        self.rms_norm_eps = config.rms_norm_eps

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # V4 keeps e_ and h_ proj separate (with fp8 linear quant) rather than
        # fusing them the way V3 does with eh_proj.
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "e_proj"),
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "h_proj"),
        )

        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = DeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

        self.hc_head_op = HCHeadOp()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)

        # Target stashes pre-hc_head residual as flat (T, hc_mult * D);
        # reshape to (T, hc_mult, D) — the training-time layout.
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.config.hidden_size
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        h_proj = self.h_proj(
            previous_hidden_states.reshape(-1, self.config.hidden_size).contiguous()
        ).view(-1, self.hc_mult, self.config.hidden_size)
        hidden_states = (
            h_proj + self.e_proj(inputs_embeds).unsqueeze(-2)
        ).contiguous()
        if current_platform.is_cuda() and _use_serial_mtp_mhc():
            hidden_states, residual, post_mix, res_mix = self.mtp_block._forward_rocm(
                positions=positions, x=hidden_states, input_ids=input_ids
            )
        else:
            hidden_states, residual, post_mix, res_mix = self.mtp_block(
                positions=positions, x=hidden_states, input_ids=input_ids
            )
        if current_platform.is_cuda() and residual is not None:
            hidden_states = self.mtp_block.hc_post(
                hidden_states, residual, post_mix, res_mix
            )
        # Return the flat pre-hc_head residual so it can be re-fed as the
        # next spec step's `previous_hidden_states` when
        # num_speculative_tokens > 1. hc_head is deferred to compute_logits.
        return hidden_states.flatten(1)


class DeepSeekV4MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.device = current_platform.device_type

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Three aux streams shared across all MTP layers, mirroring
        # DeepseekV4Model. ROCm runs the same work serially for now.
        aux_stream_list = (
            None
            if current_platform.is_rocm()
            else [torch.cuda.Stream() for _ in range(3)]
        )

        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekV4MultiTokenPredictorLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{idx}",
                    aux_stream_list=aux_stream_list,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self._target_lm_head: nn.Module | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def bind_lm_head(self, lm_head: nn.Module) -> None:
        """Bind the target LM head to every MTP layer.

        DeepSeek V4 checkpoints do not carry ``mtp.*.head.weight``. The draft
        model must use the target head after applying its own ``hc_head`` and
        norm; otherwise ``shared_head.head`` remains an uninitialized
        ``ParallelLMHead`` and draft acceptance collapses.
        """
        self._target_lm_head = lm_head
        for layer in self.layers.values():
            shared_head = getattr(layer, "shared_head", None)
            if shared_head is not None and getattr(shared_head, "head", None) is not lm_head:
                shared_head.head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        if self._target_lm_head is not None and (
            getattr(mtp_layer.shared_head, "head", None) is not self._target_lm_head
        ):
            # The proposer normally binds this during load_model(); keep a
            # lazy guard so a later module replacement cannot silently restore
            # the uninitialized per-MTP ParallelLMHead.
            mtp_layer.shared_head.head = self._target_lm_head
        # MTP forward returns the pre-hc_head residual (T, hc_mult * D); apply
        # hc_head here so logits are computed from the dense hidden state.
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = mtp_layer.hc_head_op(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            mtp_layer.rms_norm_eps,
            mtp_layer.hc_eps,
        )
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits


@support_torch_compile(enable_if=should_compile_deepseek_v4_mtp)
class DeepSeekV4MTP(nn.Module, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def bind_lm_head(self, lm_head: nn.Module) -> None:
        self.model.bind_lm_head(lm_head)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weight name remapping for checkpoint compatibility.
        # Maps checkpoint weight paths to model parameter paths.
        WEIGHT_NAME_REMAPPING: dict[str, str] = {
            ".emb.tok_emb.weight": ".embed_tokens.weight",
            ".head.weight": ".shared_head.head.weight",
            ".norm.weight": ".shared_head.norm.weight",
            # Pre-MoE norm + gate are now owned by
            # ``DeepseekV4MoE.norm_gate`` (see NormGatedLinear).
            ".ffn_norm.weight": ".ffn.norm_gate.norm.weight",
            ".ffn.gate.weight": ".ffn.norm_gate.gate.weight",
            ".ffn.gate.tid2eid": ".ffn.norm_gate.tid2eid",
        }

        def _remap_weight_name(name: str) -> str:
            """Remap checkpoint weight names to model parameter names."""
            for old_pattern, new_pattern in WEIGHT_NAME_REMAPPING.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
            return name

        def _find_mtp_layer_idx(name: str) -> int:
            subnames = name.split(".")
            for subname in subnames:
                try:
                    # we return the first encountered integer
                    return int(subname)
                except ValueError:
                    continue
            return 0

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        first_layer = next(iter(self.model.layers.values()))
        if first_layer.mtp_block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        # FP8 experts register ``..._weight_scale_inv`` (block_quant) while
        # FP4/MXFP4 experts register ``..._weight_scale``. Choose the suffix
        # for the rename below based on the model's expert dtype.
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") in ("fp4", "int4")
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            mtp_layer_idx = _find_mtp_layer_idx(name)
            # V4 checkpoints store MTP weights as `mtp.{i}.*`; remap to
            # `model.layers.{num_hidden_layers + i}.*` so that
            # get_spec_layer_idx_from_weight_name can identify them.
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
            )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            name = _remap_weight_name(name)
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if spec_layer != self.model.mtp_start_layer_idx and ".layers" not in name:
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                base_name = name.removesuffix(".scale")
                name = base_name + suffix
                # Expert parameters are stored fused (w1/w3 -> w13) and mapped
                # below by expert_mapping. The pre-mapped per-expert scale name
                # is therefore not expected to exist in params_dict.
                if ".experts." in name:
                    pass
                elif name not in params_dict:
                    alt_suffix = (
                        ".weight_scale"
                        if suffix == ".weight_scale_inv"
                        else ".weight_scale_inv"
                    )
                    alt_name = base_name + alt_suffix
                    if alt_name in params_dict:
                        name = alt_name
                    elif _mtp_scale_can_be_mapped_later(
                        name, stacked_params_mapping
                    ):
                        pass
                    else:
                        logger.warning_once(
                            "Skipping unmatched DeepSeek V4 MTP scale %s; "
                            "no %s or %s parameter is registered.",
                            base_name + ".scale",
                            base_name + suffix,
                            alt_name,
                        )
                        continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(name, self):
                    break

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # Reinterpret E8M0 scales as uint8 to preserve raw
                    # exponent bytes; numeric copy_() would zero them.
                    # Mirrors the main DeepseekV4 loader.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            break
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            loaded_params.add(name_mapped)
                            break
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    if ".shared_experts.w2" in name:
                        name = name.replace(
                            ".shared_experts.w2", ".shared_experts.down_proj"
                        )
                    if name.endswith(".ffn.gate.bias"):
                        # ``e_score_correction_bias`` lives on
                        # ``norm_gate`` directly (not on the inner gate).
                        name = name.replace(
                            ".ffn.gate.bias",
                            ".ffn.norm_gate.e_score_correction_bias",
                        )
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may have "
                    f"been quantized without including the MTP layers. "
                    f"Use a checkpoint that includes MTP layer weights, "
                    f"or disable speculative decoding."
                )
        self.finalize_mega_moe_weights()
        logger.info_once("MTP draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.model.layers.values():
            layer.mtp_block.ffn.finalize_mega_moe_weights()

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "h_proj",
            "e_proj",
            "shared_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
