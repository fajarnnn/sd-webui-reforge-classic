import math

import gradio as gr
import modules.scripts as scripts
from modules import images
from modules.processing import fix_seed, process_images
from modules.shared import opts, state


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[images.GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[images.GridAnnotation(x_label(x))] for x in xs]

    first_processed = None

    state.job_count = len(xs) * len(ys)

    for y_i, y in enumerate(ys):
        for x_i, x in enumerate(xs):
            state.job = f"{x_i + y_i * len(xs) + 1} out of {len(xs) * len(ys)}"

            processed = cell(x, y)
            if first_processed is None:
                first_processed = processed

            res.append(processed.images[0])

    grid = images.image_grid(res, rows=len(ys))
    grid = images.draw_grid_annotations(
        grid, res[0].width, res[0].height, hor_texts, ver_texts
    )

    first_processed.images = [grid]
    return first_processed


class PromptMatrix(scripts.Script):
    def title(self):
        return "Prompt Matrix"

    def ui(self, is_img2img):
        gr.HTML("<br>")
        with gr.Row():
            with gr.Column():
                put_at_start = gr.Checkbox(
                    value=False,
                    label="Put variable parts at start of prompt",
                    elem_id=self.elem_id("put_at_start"),
                )
                different_seeds = gr.Checkbox(
                    value=False,
                    label="Use different seed for each picture",
                    elem_id=self.elem_id("different_seeds"),
                )
            with gr.Column():
                prompt_type = gr.Radio(
                    label="Select prompt",
                    choices=("positive", "negative"),
                    value="positive",
                    elem_id=self.elem_id("prompt_type"),
                )
                variations_delimiter = gr.Radio(
                    label="Select joining char",
                    choices=("comma", "space"),
                    value="comma",
                    elem_id=self.elem_id("variations_delimiter"),
                )
            with gr.Column():
                margin_size = gr.Slider(
                    label="Grid margins (px)",
                    value=0,
                    minimum=0,
                    maximum=500,
                    step=2,
                    elem_id=self.elem_id("margin_size"),
                )

        return [
            put_at_start,
            different_seeds,
            prompt_type,
            variations_delimiter,
            margin_size,
        ]

    def run(
        self,
        p,
        put_at_start,
        different_seeds,
        prompt_type,
        variations_delimiter,
        margin_size,
    ):
        fix_seed(p)

        # Raise error if prompt type is not positive or negative
        if prompt_type not in ("positive", "negative"):
            raise ValueError(f"Unknown prompt type {prompt_type}")

        # Raise error if variations delimiter is not comma or space
        if variations_delimiter not in ("comma", "space"):
            raise ValueError(f"Unknown variations delimiter {variations_delimiter}")

        prompt = p.prompt if prompt_type == "positive" else p.negative_prompt
        original_prompt = prompt[0] if isinstance(prompt, list) else prompt
        positive_prompt = p.prompt[0] if isinstance(p.prompt, list) else p.prompt

        delimiter = ", " if variations_delimiter == "comma" else " "

        all_prompts = []
        prompt_matrix_parts = original_prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [
                text.strip().strip(",")
                for n, text in enumerate(prompt_matrix_parts[1:])
                if combination_num & (1 << n)
            ]

            if put_at_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(delimiter.join(selected_prompts))

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"PromptMatrix: creating {len(all_prompts)} images in {p.n_iter} batches")

        if prompt_type == "positive":
            p.prompt = all_prompts
        else:
            p.negative_prompt = all_prompts

        p.prompt_for_display = positive_prompt
        p.seed = [
            p.seed + (i if different_seeds else 0) for i in range(len(all_prompts))
        ]

        processed = process_images(p)

        grid = images.image_grid(
            processed.images,
            p.batch_size,
            rows=1 << ((len(prompt_matrix_parts) - 1) // 2),
        )
        grid = images.draw_prompt_matrix(
            grid,
            processed.images[0].width,
            processed.images[0].height,
            prompt_matrix_parts,
            margin_size,
        )

        processed.images.insert(0, grid)
        processed.index_of_first_image = 1
        processed.infotexts.insert(0, processed.infotexts[0])

        if opts.grid_save:
            images.save_image(
                processed.images[0],
                p.outpath_grids,
                "prompt_matrix",
                extension=opts.grid_format,
                prompt=original_prompt,
                seed=processed.seed,
                grid=True,
                p=p,
            )

        return processed
