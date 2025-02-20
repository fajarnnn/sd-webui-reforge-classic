import os
from abc import abstractmethod

import PIL
from PIL import Image

import modules.shared
from modules import modelloader, shared


LANCZOS = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
NEAREST = getattr(Image, "NEAREST", Image.Resampling.LANCZOS)

# hardcode
UPSCALE_ITERATIONS = 4


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    scalers: list["UpscalerData"] = []
    tile = True

    def __init__(self, create_dirs=False):
        self.mod_pad_h = None
        self.tile_size = modules.shared.opts.ESRGAN_tile
        self.tile_pad = modules.shared.opts.ESRGAN_tile_overlap
        self.device = modules.shared.device
        self.img = None
        self.output = None
        self.scale = 1
        self.half = not modules.shared.cmd_opts.no_half
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = None

        if self.model_path is None and self.name:
            self.model_path = os.path.join(shared.models_path, self.name)
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

        try:
            import cv2  # noqa: F401
            self.can_tile = True
        except ImportError:
            pass

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        raise NotImplementedError

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(UPSCALE_ITERATIONS):
            img = self.do_upscale(img, selected_model)
            if ((img.width >= dest_w) and (img.height >= dest_h)) or (int(scale) == 1):
                break

        if (img.width != dest_w) or (img.height != dest_h):
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img

    @abstractmethod
    def load_model(self, path: str):
        raise NotImplementedError

    def find_models(self, ext_filter=None) -> list:
        return modelloader.load_models(
            model_path=self.model_path,
            model_url=self.model_url,
            command_path=self.user_path,
            ext_filter=ext_filter,
        )

    def update_status(self, prompt):
        print(f"\nextras: {prompt}", file=shared.progress_print_out)


class UpscalerData:
    name = None
    data_path = None
    scale: int = 4
    scaler: Upscaler = None
    model: None

    def __init__(
        self,
        name: str,
        path: str,
        upscaler: Upscaler = None,
        scale: int = 4,
        model=None,
    ):
        self.name = name
        self.data_path = path
        self.local_data_path = path
        self.scaler = upscaler
        self.scale = scale
        self.model = model

    def __repr__(self):
        return (
            f"<UpscalerData name={self.name}"
            + f" path={self.data_path}"
            + f" scale={self.scale}>"
        )


class UpscalerNone(Upscaler):

    def load_model(self, _):
        return

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "None"
        self.scalers = [UpscalerData("None", None, self)]


class UpscalerLanczos(Upscaler):

    def load_model(self, _):
        return

    def do_upscale(self, img: PIL.Image.Image, selected_model=None):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=LANCZOS,
        )

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"
        self.scalers = [UpscalerData("Lanczos", None, self)]


class UpscalerNearest(Upscaler):

    def load_model(self, _):
        return

    def do_upscale(self, img, selected_model=None):
        return img.resize(
            size=(int(img.width * self.scale), int(img.height * self.scale)),
            resample=NEAREST,
        )

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"
        self.scalers = [UpscalerData("Nearest", None, self)]
