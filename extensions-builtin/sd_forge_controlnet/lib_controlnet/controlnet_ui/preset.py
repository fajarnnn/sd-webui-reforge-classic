from modules_forge.supported_preprocessor import Preprocessor
from modules import scripts

from lib_controlnet.controlnet_ui.tool_button import ToolButton
from lib_controlnet.infotext import parse_unit, serialize_unit
from lib_controlnet.global_state import get_preprocessor
from lib_controlnet.external_code import ControlNetUnit
from lib_controlnet.logging import logger

import gradio as gr
import os

NEW_PRESET = "New Preset"
PRESET_FOLDER = os.path.join(scripts.basedir(), "presets")

reset_symbol = "\U000021A9"  # ↩
save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f"  # 🗑️
refresh_symbol = "\U0001f504"  # 🔄


def load_presets(preset_dir: str) -> dict[str, str]:
    if not os.path.exists(preset_dir):
        os.makedirs(preset_dir)
        return {}

    presets = {}
    for filename in os.listdir(preset_dir):
        if filename.endswith(".txt"):
            name = filename.split(".txt")[0]
            assert name != NEW_PRESET
            with open(os.path.join(preset_dir, filename), "r") as f:
                presets[name] = f.read()
    return presets


class ControlNetPresetUI:
    presets = load_presets(PRESET_FOLDER)

    def __init__(self, id_prefix: str):
        """UIs for the Preset Row"""

        with gr.Row():
            self.dropdown = gr.Dropdown(
                label="Presets",
                show_label=True,
                elem_classes=["cnet-preset-dropdown"],
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
            )
            self.reset_button = ToolButton(
                value=reset_symbol,
                elem_id="cnet-preset-reset",
                tooltip="Apply preset",
                visible=True,
                interactive=False,
            )
            self.save_button = ToolButton(
                value=save_symbol,
                elem_id="cnet-preset-save",
                tooltip="Save preset",
                visible=True,
                interactive=True,
            )
            self.delete_button = ToolButton(
                value=delete_symbol,
                elem_id="cnet-preset-delete",
                tooltip="Delete preset",
                visible=True,
                interactive=False,
            )
            self.refresh_button = ToolButton(
                value=refresh_symbol,
                elem_id="cnet-preset-refresh",
                tooltip="Refresh preset",
                visible=True,
                interactive=True,
            )

        with gr.Group(
            visible=False,
            elem_id=f"{id_prefix}_cnet_preset_enter_name",
            elem_classes=["popup-dialog", "cnet-preset-enter-name"],
        ) as self.name_dialog:
            with gr.Row(elem_id="cnet-preset-dialog-row"):
                self.preset_name = gr.Textbox(
                    label="Preset name",
                    show_label=True,
                    elem_classes=["cnet-preset-name"],
                    max_lines=1,
                    lines=1,
                )
                self.confirm_preset_name = ToolButton(
                    value=save_symbol,
                    elem_id="cnet-preset-confirm-name",
                    tooltip="Save preset",
                )

    def register_callbacks(self, control_type: gr.Radio, *ui_states):
        """Interactions with the main ControlNet tab"""

        def apply_preset(name: str, *ui_states) -> tuple[dict]:
            assert name in (*self.presets, NEW_PRESET)
            if name == NEW_PRESET:
                return (gr.skip(),) * (len(ControlNetUnit.infotext_fields()) + 1)

            infotext = self.presets[name]
            preset_unit = parse_unit(infotext)
            current_unit = self.init_with_ui_states(*ui_states)

            preset_unit.image = None
            current_unit.image = None
            new_control_type = self.infer_control_type(preset_unit.module)

            # Do not compare module param that are not used in preset
            for module_param in ("processor_res", "threshold_a", "threshold_b"):
                if getattr(preset_unit, module_param) == -1:
                    setattr(current_unit, module_param, -1)

            return (
                gr.update(value=new_control_type),
                *[
                    gr.update(value=value) if value is not None else gr.skip()
                    for field in ControlNetUnit.infotext_fields()
                    for value in (getattr(preset_unit, field),)
                ],
            )

        for element, action in (
            (self.dropdown, "change"),
            (self.reset_button, "click"),
        ):
            getattr(element, action)(
                fn=apply_preset,
                inputs=[self.dropdown, *ui_states],
                outputs=[control_type, *ui_states],
                show_progress="hidden",
            ).success(
                fn=lambda: NEW_PRESET,
                outputs=[self.dropdown],
                show_progress="hidden",
            )

        def on_save_preset(name: str) -> dict:
            return gr.update(visible=(name == NEW_PRESET))

        self.save_button.click(
            fn=on_save_preset,
            inputs=[self.dropdown],
            outputs=[self.name_dialog],
            show_progress="hidden",
        ).success(
            fn=None,
            inputs=[self.dropdown],
            _js=f"""
            (name) => {{
                if (name === "{NEW_PRESET}")
                    popup(document.getElementById('{self.name_dialog.elem_id}'));
            }}""",
        )

        def save_new_preset(new_name: str, *ui_states) -> tuple[dict]:
            if new_name == NEW_PRESET:
                logger.error(f'"{NEW_PRESET}" is a reserved name...')
                return (gr.update(visible=False), gr.skip())

            self.save_preset(new_name, self.init_with_ui_states(*ui_states))
            return (
                gr.update(visible=False),
                gr.update(
                    value=new_name,
                    choices=ControlNetPresetUI.dropdown_choices(),
                ),
            )

        self.confirm_preset_name.click(
            fn=save_new_preset,
            inputs=[self.preset_name, *ui_states],
            outputs=[self.name_dialog, self.dropdown],
            show_progress="hidden",
        ).success(fn=None, _js="closePopup")

        def on_delete_preset(name: str) -> dict:
            self.delete_preset(name)
            return gr.update(
                value=NEW_PRESET,
                choices=ControlNetPresetUI.dropdown_choices(),
            )

        self.delete_button.click(
            fn=on_delete_preset,
            inputs=[self.dropdown],
            outputs=[self.dropdown],
            show_progress="hidden",
        )

        self.refresh_button.click(
            fn=self.refresh_preset,
            inputs=None,
            outputs=[self.dropdown],
            show_progress="hidden",
        )

        def update_buttons(preset_name: str):
            return [
                gr.update(interactive=(preset_name != NEW_PRESET)),
                gr.update(interactive=(preset_name == NEW_PRESET)),
                gr.update(interactive=(preset_name != NEW_PRESET)),
            ]

        self.dropdown.change(
            fn=update_buttons,
            inputs=[self.dropdown],
            outputs=[self.reset_button, self.save_button, self.delete_button],
        )

    @classmethod
    def dropdown_choices(cls) -> list[str]:
        return [NEW_PRESET] + list(cls.presets.keys())

    @classmethod
    def save_preset(cls, name: str, unit: ControlNetUnit):
        infotext = serialize_unit(unit)
        with open(os.path.join(PRESET_FOLDER, f"{name}.txt"), "w+") as f:
            f.write(infotext)

        cls.presets[name] = infotext

    @classmethod
    def delete_preset(cls, name: str):
        assert name in cls.presets
        del cls.presets[name]

        file = os.path.join(PRESET_FOLDER, f"{name}.txt")
        if os.path.exists(file):
            os.remove(file)

    @classmethod
    def refresh_preset(cls):
        cls.presets = load_presets(PRESET_FOLDER)
        return gr.update(choices=cls.dropdown_choices())

    @staticmethod
    def init_with_ui_states(*ui_states) -> ControlNetUnit:
        return ControlNetUnit(
            **{
                field: value
                for field, value in zip(ControlNetUnit.infotext_fields(), ui_states)
            }
        )

    @staticmethod
    def infer_control_type(module: str) -> str:
        preprocessor: Preprocessor = get_preprocessor(module)
        if preprocessor is None:
            return "All"

        return getattr(preprocessor, "tags", ["All"])[0]
