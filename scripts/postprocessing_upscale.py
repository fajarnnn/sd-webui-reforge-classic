import re

import gradio as gr
import numpy as np
from modules import scripts_postprocessing, shared
from modules.processing import setup_color_correction, apply_color_correction
from modules.ui import switch_values_symbol
from modules.ui_components import FormRow, ToolButton
from PIL import Image

upscale_cache = {}


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def ui(self):
        selected_tab = gr.Number(value=0, visible=False)

        with gr.Column():
            with FormRow():
                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.Tab(
                        "Scale by",
                        elem_id="extras_scale_by_tab",
                    ) as tab_scale_by:
                        upscaling_resize = gr.Slider(
                            label="Resize",
                            value=2.0,
                            minimum=1.0,
                            maximum=8.0,
                            step=0.05,
                            elem_id="extras_upscaling_resize",
                        )
                    with gr.Tab(
                        "Scale to",
                        elem_id="extras_scale_to_tab",
                    ) as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id="upscaling_column_size", scale=4):
                                upscaling_resize_w = gr.Slider(
                                    minimum=64,
                                    maximum=8192,
                                    step=8,
                                    label="Width",
                                    value=512,
                                    elem_id="extras_upscaling_resize_w",
                                )
                                upscaling_resize_h = gr.Slider(
                                    minimum=64,
                                    maximum=8192,
                                    step=8,
                                    label="Height",
                                    value=512,
                                    elem_id="extras_upscaling_resize_h",
                                )
                            with gr.Column(
                                elem_id="upscaling_dimensions_row",
                                elem_classes="dimensions-tools",
                                scale=1,
                            ):
                                upscaling_res_switch_btn = ToolButton(
                                    value=switch_values_symbol,
                                    elem_id="upscaling_res_switch_btn",
                                    tooltip="Switch width/height",
                                )
                                upscaling_crop = gr.Checkbox(
                                    value=True,
                                    label="Crop to fit",
                                    elem_id="extras_upscaling_crop",
                                )

            with gr.Column(scale=4):
                with gr.Row():
                    extras_upscaler_1 = gr.Dropdown(
                        label="Upscaler 1",
                        elem_id="extras_upscaler_1",
                        choices=[x.name for x in shared.sd_upscalers],
                        value=shared.sd_upscalers[0].name,
                    )
                    extras_upscaler_2 = gr.Dropdown(
                        label="Upscaler 2",
                        elem_id="extras_upscaler_2",
                        choices=[x.name for x in shared.sd_upscalers],
                        value=shared.sd_upscalers[0].name,
                        interactive=False,
                    )
                with gr.Row():
                    extras_color_correction = gr.Checkbox(
                        value=True,
                        label="Color Correction",
                        elem_id="extras_color_correction",
                    )
                    extras_upscaler_2_visibility = gr.Slider(
                        label="Upscaler 2 visibility",
                        value=0.0,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        elem_id="extras_upscaler_2_visibility",
                    )

        upscaling_res_switch_btn.click(
            lambda w, h: (h, w),
            inputs=[upscaling_resize_w, upscaling_resize_h],
            outputs=[upscaling_resize_w, upscaling_resize_h],
            show_progress="hidden",
        )

        tab_scale_by.select(fn=lambda: 0, outputs=[selected_tab])
        tab_scale_to.select(fn=lambda: 1, outputs=[selected_tab])

        def on_selected_upscale_method(upscale_method: str):
            if not (match := re.search(r"(\d)[xX]|[xX](\d)", upscale_method)):
                return gr.skip()

            return gr.update(value=int(match.group(1) or match.group(2)))

        extras_upscaler_1.change(
            on_selected_upscale_method,
            inputs=[extras_upscaler_1],
            outputs=[upscaling_resize],
            show_progress="hidden",
        )

        extras_upscaler_2_visibility.change(
            fn=lambda vis: gr.update(interactive=(vis > 0.0)),
            inputs=[extras_upscaler_2_visibility],
            outputs=[extras_upscaler_2],
            show_progress="hidden",
        )

        return {
            "upscale_mode": selected_tab,
            "upscale_cc": extras_color_correction,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        }

    def _upscale(
        self,
        image,
        info,
        upscaler,
        upscale_mode,
        upscale_by,
        upscale_to_width,
        upscale_to_height,
        upscale_crop,
    ):
        if upscale_mode == 0:
            info["Postprocess upscale by"] = upscale_by
        else:
            info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
            upscale_by = max(
                upscale_to_width / image.width,
                upscale_to_height / image.height,
            )

        cache_key = (
            hash(np.array(image.getdata()).tobytes()),
            upscaler.name,
            upscale_mode,
            upscale_by,
            upscale_to_width,
            upscale_to_height,
            upscale_crop,
        )

        cached_image = upscale_cache.pop(cache_key, None)

        if cached_image is not None:
            image = cached_image
        else:
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        upscale_cache[cache_key] = image
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        if upscale_mode == 1 and upscale_crop:
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            cropped.paste(
                image,
                box=(
                    upscale_to_width // 2 - image.width // 2,
                    upscale_to_height // 2 - image.height // 2,
                ),
            )
            image = cropped
            info["Postprocess crop to"] = f"{image.width}x{image.height}"

        return image

    def process_firstpass(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_mode=1,
        upscale_cc=False,
        upscale_by=2.0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        else:
            pp.shared.target_width = int(pp.image.width * upscale_by)
            pp.shared.target_height = int(pp.image.height * upscale_by)

        if upscale_cc and "cc" not in upscale_cache:
            upscale_cache["cc"] = setup_color_correction(pp.image)

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        upscale_mode=1,
        upscale_cc=False,
        upscale_by=2.0,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=None,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    ):
        if upscaler_1_name == "None":
            return

        upscaler1 = None
        for x in shared.sd_upscalers:
            if x.name == upscaler_1_name:
                upscaler1 = x
                break

        assert upscaler1 is not None, f'Could not find upscaler "{upscaler_1_name}"'

        upscaled_image = self._upscale(
            pp.image,
            pp.info,
            upscaler1,
            upscale_mode,
            upscale_by,
            upscale_to_width,
            upscale_to_height,
            upscale_crop,
        )

        pp.info["Postprocess upscaler"] = upscaler1.name

        if upscaler_2_visibility > 0.0:
            if upscaler_2_name != "None":
                upscaler2 = next(
                    (x for x in shared.sd_upscalers if x.name == upscaler_2_name),
                    None,
                )

                assert (
                    upscaler2 is not None
                ), f'Could not find upscaler "{upscaler_2_name}"'

                second_upscale = self._upscale(
                    pp.image,
                    pp.info,
                    upscaler2,
                    upscale_mode,
                    upscale_by,
                    upscale_to_width,
                    upscale_to_height,
                    upscale_crop,
                )

                upscaled_image = Image.blend(
                    upscaled_image,
                    second_upscale,
                    upscaler_2_visibility,
                )

                pp.info["Postprocess upscaler 2"] = upscaler2.name
                pp.info["Postprocess upscaler 2 visibility"] = upscaler_2_visibility

        if upscale_cc and "cc" in upscale_cache:  # postprocess during txt2img
            pp.image = apply_color_correction(upscale_cache["cc"], upscaled_image)
        else:
            pp.image = upscaled_image

    def image_changed(self):
        upscale_cache.clear()
