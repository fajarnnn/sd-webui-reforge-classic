"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI

- Edited by. Forge Official
- Edited by. Haoming02
"""

import argparse
import enum


class EnumAction(argparse.Action):
    """Argparse `action` for handling Enum"""

    def __init__(self, **kwargs):
        enum_type = kwargs.pop("type", None)
        assert issubclass(enum_type, enum.Enum)

        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        value = self._enum(values)
        setattr(namespace, self.dest, value)


parser = argparse.ArgumentParser()


parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")
parser.add_argument("--directml", action="store_true")
parser.add_argument("--disable-ipex-hijack", action="store_true")
parser.add_argument("--disable-attention-upcast", action="store_true")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--all-in-fp32", action="store_true")
fp_group.add_argument("--all-in-fp16", action="store_true")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--unet-in-bf16", action="store_true")
fpunet_group.add_argument("--unet-in-fp16", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true")
fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--vae-in-fp16", action="store_true")
fpvae_group.add_argument("--vae-in-fp32", action="store_true")
fpvae_group.add_argument("--vae-in-bf16", action="store_true")

parser.add_argument("--clip-in-gpu", action="store_true")
parser.add_argument("--vae-in-cpu", action="store_true")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
fpte_group.add_argument("--clip-in-fp16", action="store_true")
fpte_group.add_argument("--clip-in-fp32", action="store_true")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--always-gpu", action="store_true")
vram_group.add_argument("--always-high-vram", action="store_true")
vram_group.add_argument("--always-normal-vram", action="store_true")
vram_group.add_argument("--always-low-vram", action="store_true")
vram_group.add_argument("--always-no-vram", action="store_true")
vram_group.add_argument("--always-cpu", action="store_true")

parser.add_argument("--disable-sage", action="store_true")
parser.add_argument("--disable-flash", action="store_true")
parser.add_argument("--disable-xformers", action="store_true")
parser.add_argument("--attention-pytorch", action="store_true")

parser.add_argument("--always-offload-from-vram", action="store_true")
parser.add_argument("--pytorch-deterministic", action="store_true")

parser.add_argument("--cuda-malloc", action="store_true")
parser.add_argument("--cuda-stream", action="store_true")
parser.add_argument("--pin-shared-memory", action="store_true")

parser.add_argument("--fast-fp16", action="store_true")
parser.add_argument("--persistent-patches", action="store_true")


class SageAttentionFuncs(enum.Enum):
    auto = "auto"
    fp16_triton = "fp16_triton"
    fp16_cuda = "fp16_cuda"
    fp8_cuda = "fp8_cuda"


class Sage_quantization_backend(enum.Enum):
    cuda = "cuda"
    triton = "triton"


class Sage_qk_quant_gran(enum.Enum):
    per_warp = "per_warp"
    per_thread = "per_thread"


class Sage_pv_accum_dtype(enum.Enum):
    fp16 = "fp16"
    fp32 = "fp32"
    fp16fp32 = "fp16+fp32"
    fp32fp32 = "fp32+fp32"


parser.add_argument("--sage2-function", type=SageAttentionFuncs, default=SageAttentionFuncs.auto, action=EnumAction)
parser.add_argument("--sage-quantization-backend", type=Sage_quantization_backend, default=Sage_quantization_backend.triton, action=EnumAction)
parser.add_argument("--sage-quant-gran", type=Sage_qk_quant_gran, default=Sage_qk_quant_gran.per_thread, action=EnumAction)
parser.add_argument("--sage-accum-dtype", type=Sage_pv_accum_dtype, default=Sage_pv_accum_dtype.fp32, action=EnumAction)


args = parser.parse_args([])
