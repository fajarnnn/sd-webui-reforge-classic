import gradio as gr
from modules import scripts_postprocessing
from modules.ui_components import InputAccordion
from PIL import Image


class FlippedCopiesPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = "Create Flipped Copies"
    order = 4030

    def ui(self):
        with InputAccordion(False, label="Create Flipped Copies") as enable:
            option = gr.CheckboxGroup(
                value=["X Flip"],
                choices=("X Flip", "Y Flip", "XY Flip"),
                show_label=False,
            )

        return {
            "enable": enable,
            "option": option,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, option):
        if not enable:
            return

        if "X Flip" in option:
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

        if "Y Flip" in option:
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))

        if "XY Flip" in option:
            pp.extra_images.append(
                pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
            )
