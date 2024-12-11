"""
Credit: ComfyUI
https://github.com/comfyanonymous/ComfyUI/blob/v0.3.7/comfy_extras/nodes_model_advanced.py#L257
"""

from modules.shared import opts
import torch


def try_patch_cfg(model):
    multiplier: float = getattr(opts, "rescale_cfg", 0.0)
    if multiplier < 0.05:
        return model

    @torch.inference_mode()
    def rescale_cfg(args):
        cond = args["cond"]
        uncond = args["uncond"]
        cond_scale = args["cond_scale"]
        sigma = args["sigma"]
        sigma = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        x_orig = args["input"]

        x = x_orig / (sigma * sigma + 1.0)
        cond = ((x - (x_orig - cond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)
        uncond = ((x - (x_orig - uncond)) * (sigma**2 + 1.0) ** 0.5) / (sigma)

        x_cfg = uncond + cond_scale * (cond - uncond)
        ro_pos = torch.std(cond, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(x_cfg, dim=(1, 2, 3), keepdim=True)

        x_rescaled = x_cfg * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * x_cfg

        return x_orig - (x - x_final * sigma / (sigma * sigma + 1.0) ** 0.5)

    model.set_model_sampler_cfg_function(rescale_cfg)
    print("Applied Rescale CFG: ", multiplier)

    return model
