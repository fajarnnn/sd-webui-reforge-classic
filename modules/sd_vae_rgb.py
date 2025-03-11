# Reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.26/latent_preview.py#L39

from ldm_patched.modules.latent_formats import SD15, SDXL
from modules import shared, devices
from torch.nn.functional import linear
from torch import tensor


class Latent2RGBPreviewer:

    def __init__(self):
        mdl = (SDXL if shared.sd_model.is_sdxl else SD15)()
        self.latent_rgb_factors = tensor(
            mdl.latent_rgb_factors,
            dtype=devices.dtype,
            device="cpu",
        ).transpose(0, 1)

    def __call__(self, x0):
        if x0.ndim == 5:
            x0 = x0[0, :, 0]
        else:
            x0 = x0[0]

        preview = linear(x0.movedim(0, -1), self.latent_rgb_factors)
        return preview.permute(2, 0, 1).unsqueeze(0)


def model():
    rgb = Latent2RGBPreviewer()
    return rgb
