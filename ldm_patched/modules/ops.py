"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI

- Edited by. Forge Official
- Edited by. Haoming02
"""

import contextlib
import torch

from ldm_patched.modules.model_management import cast_to_device
from modules_forge import stream


# https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855/files
stash = {}


@contextlib.contextmanager
def use_patched_ops(operations):
    names = ("Linear", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm")
    backups = {name: getattr(torch.nn, name) for name in names}

    try:
        for name in names:
            setattr(torch.nn, name, getattr(operations, name))
        yield
    finally:
        for name in names:
            setattr(torch.nn, name, backups[name])
        return


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias, signal = None, None

    with (
        stream.stream_context()(stream.mover_stream)
        if stream.using_stream
        else contextlib.nullcontext()
    ):
        if s.bias is not None:
            bias = cast_to_device(
                s.bias,
                device=device,
                dtype=bias_dtype,
            )
        weight = cast_to_device(
            s.weight,
            device=device,
            dtype=dtype,
        )
        if stream.using_stream:
            signal = stream.mover_stream.record_event()

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if not stream.using_stream or signal is None:
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        stash[id(finished_signal)] = (weight, bias, finished_signal)

    garbage = []
    for k, (_, _, s) in stash.items():
        if s.query():
            garbage.append(k)

    for k in garbage:
        del stash[k]
    return


def cleanup_cache():
    if not stream.using_stream:
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    stash.clear()
    return


class CastWeightBiasOp:
    ldm_patched_cast_weights = False
    weight_function = None
    bias_function = None


class disable_weight_init:
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.group_norm(
                    input,
                    self.num_groups,
                    weight,
                    bias,
                    self.eps,
                )

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.layer_norm(
                    input, self.normalized_shape, weight, bias, self.eps
                )

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        ldm_patched_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        ldm_patched_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        ldm_patched_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        ldm_patched_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        ldm_patched_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        ldm_patched_cast_weights = True


def fp8_linear(self, input):
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    tensor_2d = False
    if len(input.shape) == 2:
        tensor_2d = True
        input = input.unsqueeze(1)

    if len(input.shape) == 3:
        w, bias, signal = cast_bias_weight(
            self, input, dtype=dtype, bias_dtype=input.dtype
        )
        w = w.t()

        scale_weight = self.scale_weight
        scale_input = self.scale_input
        if scale_weight is None:
            scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
        else:
            scale_weight = scale_weight.to(input.device)

        if scale_input is None:
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            inn = input.reshape(-1, input.shape[2]).to(dtype)
        else:
            scale_input = scale_input.to(input.device)
            inn = (
                (input * (1.0 / scale_input).to(input.dtype))
                .reshape(-1, input.shape[2])
                .to(dtype)
            )

        with main_stream_worker(w, bias, signal):
            if bias is not None:
                o = torch._scaled_mm(
                    inn,
                    w,
                    out_dtype=input.dtype,
                    bias=bias,
                    scale_a=scale_input,
                    scale_b=scale_weight,
                )
            else:
                o = torch._scaled_mm(
                    inn,
                    w,
                    out_dtype=input.dtype,
                    scale_a=scale_input,
                    scale_b=scale_weight,
                )

        if isinstance(o, tuple):
            o = o[0]

        if tensor_2d:
            return o.reshape(input.shape[0], -1)

        return o.reshape((-1, input.shape[1], self.weight.shape[0]))

    return None


class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_ldm_patched_cast_weights(self, input):
            out = fp8_linear(self, input)
            if out is not None:
                return out

            weight, bias, signal = cast_bias_weight(self, input)
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(input, weight, bias)
