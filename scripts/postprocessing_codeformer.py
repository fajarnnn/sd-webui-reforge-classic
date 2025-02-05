import gradio as gr
import numpy as np
from modules import codeformer_model, scripts_postprocessing
from modules.ui_components import InputAccordion
from PIL import Image


class CodeFormerPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = "CodeFormer"
    order = 3000

    def ui(self):
        with InputAccordion(False, label="CodeFormer") as enable:
            with gr.Row():
                codeformer_visibility = gr.Slider(
                    label="Visibility",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    elem_id="extras_codeformer_visibility",
                )
                codeformer_weight = gr.Slider(
                    label="Weight (0 = maximum effect, 1 = minimum effect)",
                    value=0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    elem_id="extras_codeformer_weight",
                )

        return {
            "enable": enable,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        enable,
        codeformer_visibility,
        codeformer_weight,
    ):
        if not enable or codeformer_visibility < 0.05:
            return

        restored_img = codeformer_model.codeformer.restore(
            np.array(pp.image, dtype=np.uint8), w=codeformer_weight
        )

        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(pp.image, res, codeformer_visibility)

        pp.image = res
        pp.info["CodeFormer Visibility"] = round(codeformer_visibility, 2)
        pp.info["CodeFormer Weight"] = round(codeformer_weight, 2)
