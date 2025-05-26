import re
from typing import TYPE_CHECKING

from modules import script_callbacks, scripts, shared

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing


def strip_comments(text):
    text = re.sub("(^|\n)#[^\n]*(\n|$)", "\n", text)  # while line comment
    text = re.sub("#[^\n]*(\n|$)", "\n", text)  # in the middle of the line comment

    return text


class ScriptComments(scripts.Script):
    def title(self):
        return "Comments"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p: "StableDiffusionProcessing", *args):
        if not shared.opts.enable_prompt_comments:
            return

        p.all_prompts = [strip_comments(x) for x in p.all_prompts]
        p.all_negative_prompts = [strip_comments(x) for x in p.all_negative_prompts]

        p.main_prompt = strip_comments(p.main_prompt)
        p.main_negative_prompt = strip_comments(p.main_negative_prompt)

        if getattr(p, "enable_hr", False):
            p.all_hr_prompts = [strip_comments(x) for x in p.all_hr_prompts]
            p.all_hr_negative_prompts = [strip_comments(x) for x in p.all_hr_negative_prompts]

            p.hr_prompt = strip_comments(p.hr_prompt)
            p.hr_negative_prompt = strip_comments(p.hr_negative_prompt)


def before_token_counter(params: script_callbacks.BeforeTokenCounterParams):
    if not shared.opts.enable_prompt_comments:
        return

    params.prompt = strip_comments(params.prompt)


script_callbacks.on_before_token_counter(before_token_counter)


shared.options_templates.update(
    shared.options_section(
        ("ui_alternatives", "UI Alternatives", "ui"),
        {"enable_prompt_comments": shared.OptionInfo(True, "Enable Comments").info("Ignore the texts between # and the end of the line from the prompts")},
    )
)
