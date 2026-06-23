# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

from .quant_config import DeepseekV4FP8Config

# The per-platform model/MTP classes are imported LAZILY (PEP 562 ``__getattr__``)
# rather than at package-import time. Importing the heavy model here makes a leaf
# op module (e.g. ``common.ops.fp8e4m3_arith``, pulled in by the Triton kernels)
# unimportable on its own: model -> attention -> deepseek_v4_triton_kernels ->
# (back into this package) is a circular import. Deferring the model import until
# someone actually accesses the class breaks that cycle while keeping
# ``deepseek_v4.DeepseekV4ForCausalLM`` working for the registry / quant lookup.
if TYPE_CHECKING:
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]


def __getattr__(name: str):
    if name in ("DeepseekV4ForCausalLM", "DeepSeekV4MTP"):
        # NVIDIA is the default; ROCm overrides at runtime.
        if current_platform.is_rocm():
            from .amd.model import DeepseekV4ForCausalLM
            from .amd.mtp import DeepSeekV4MTP
        else:
            from .nvidia.model import DeepseekV4ForCausalLM
            from .nvidia.mtp import DeepSeekV4MTP
        globals()["DeepseekV4ForCausalLM"] = DeepseekV4ForCausalLM
        globals()["DeepSeekV4MTP"] = DeepSeekV4MTP
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
