from ldm_patched.modules import model_management
from modules.ui_components import FormRow
from modules import scripts, shared

import gradio as gr


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = False
        self.original_vram_state = model_management.vram_state

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(label=self.title(), open=False):
            with FormRow():
                unet_enable = gr.Checkbox(value=False, label="Enable for UNet", info="always offload to memory")
                vae_enable = gr.Checkbox(value=False, label="Enabled for VAE", info="always tiled encoding/decoding")
            with FormRow():
                tile_size = gr.Slider(minimum=64, maximum=1024, step=64, value=512, label="Tile Size", info="in pixels")
                tile_overlap = gr.Slider(minimum=16, maximum=256, step=4, value=64, label="Tile Overlap", info="in pixels")

        return unet_enable, vae_enable, tile_size, tile_overlap

    def process(self, p, unet_enable: bool, vae_enable: bool, tile_size: int, tile_overlap: int, **kwargs):

        if unet_enable:
            print("[Never OOM] Enabled for UNet")

        if vae_enable:
            print("[Never OOM] Enabled for VAE")
            shared.opts.tile_size = tile_size
            shared.opts.tile_overlap = min(tile_size // 4, tile_overlap)

        model_management.VAE_ALWAYS_TILED = vae_enable

        if self.previous_unet_enabled != unet_enable:
            self.previous_unet_enabled = unet_enable

            model_management.unload_all_models()
            if unet_enable:
                self.original_vram_state = model_management.vram_state
                model_management.vram_state = model_management.VRAMState.NO_VRAM
            else:
                model_management.vram_state = self.original_vram_state
            print(f"Changed VRAM State To {model_management.vram_state.name}")
