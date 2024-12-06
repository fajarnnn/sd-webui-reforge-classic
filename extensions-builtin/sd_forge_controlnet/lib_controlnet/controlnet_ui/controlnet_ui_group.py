from modules.ui_components import FormRow
from modules_forge.forge_util import HWC3
from modules import shared

from dataclasses import dataclass
from typing import Optional
import gradio as gr
import numpy as np
import functools
import json

from lib_controlnet.logging import logger
from lib_controlnet.controlnet_ui.tool_button import ToolButton
from lib_controlnet.controlnet_ui.openpose_editor import OpenposeEditor
from lib_controlnet.controlnet_ui.preset import ControlNetPresetUI, NEW_PRESET
from lib_controlnet.utils import svg_preprocess, judge_image_type
from lib_controlnet.enums import InputMode, HiResFixOption
from lib_controlnet.external_code import UiControlNetUnit
from lib_controlnet import global_state, external_code


@dataclass
class A1111Context:
    """Contains all components from A1111"""

    txt2img_submit_button: Optional[gr.components.Component] = None
    img2img_submit_button: Optional[gr.components.Component] = None

    txt2img_w_slider: Optional[gr.components.Component] = None
    txt2img_h_slider: Optional[gr.components.Component] = None
    img2img_w_slider: Optional[gr.components.Component] = None
    img2img_h_slider: Optional[gr.components.Component] = None

    txt2img_enable_hr: Optional[gr.components.Component] = None

    img2img_img2img_tab: Optional[gr.components.Component] = None
    img2img_img2img_sketch_tab: Optional[gr.components.Component] = None
    img2img_batch_tab: Optional[gr.components.Component] = None
    img2img_inpaint_tab: Optional[gr.components.Component] = None
    img2img_inpaint_sketch_tab: Optional[gr.components.Component] = None
    img2img_inpaint_upload_tab: Optional[gr.components.Component] = None
    img2img_inpaint_area: Optional[gr.components.Component] = None

    @property
    def img2img_inpaint_tabs(self) -> tuple[gr.components.Component]:
        return (
            self.img2img_inpaint_tab,
            self.img2img_inpaint_sketch_tab,
            self.img2img_inpaint_upload_tab,
        )

    @property
    def img2img_non_inpaint_tabs(self) -> tuple[gr.components.Component]:
        return (
            self.img2img_img2img_tab,
            self.img2img_img2img_sketch_tab,
            self.img2img_batch_tab,
        )

    @property
    def ui_initialized(self) -> bool:
        optional = (
            "img2img_img2img_tab",
            "img2img_img2img_sketch_tab",
            "img2img_batch_tab",
            "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab",
        )

        return all(c for name, c in vars(self).items() if name not in optional)

    def set_component(self, component: gr.components.Component):
        elem_id = getattr(component, "elem_id", None)
        id_mapping = {
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "txt2img_width": "txt2img_w_slider",
            "txt2img_height": "txt2img_h_slider",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
            "img2img_img2img_tab": "img2img_img2img_tab",
            "img2img_img2img_sketch_tab": "img2img_img2img_sketch_tab",
            "img2img_batch_tab": "img2img_batch_tab",
            "img2img_inpaint_tab": "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab": "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab": "img2img_inpaint_upload_tab",
            "img2img_inpaint_full_res": "img2img_inpaint_area",
            "txt2img_hr-checkbox": "txt2img_enable_hr",
        }
        # Do not set component if it has already been set
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)
            logger.debug(f"Setting {elem_id}.")
            logger.debug(
                f"A1111 initialized {sum(c is not None for c in vars(self).values())}/{len(vars(self).keys())}."
            )


class ControlNetUiGroup:
    a1111_context = A1111Context()

    all_callbacks_registered: bool = False
    all_ui_groups: list["ControlNetUiGroup"] = []
    """All ControlNetUiGroup instances created"""

    refresh_symbol = "\U0001f504"  # 🔄
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"  # ⤴
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝
    switch_values_symbol = "\U000021C5"  # ⇅

    tooltips = {
        "🔄": "Refresh",
        "📷": "Enable webcam",
        "⇄": "Mirror webcam",
        "⤴": "Sync resolution",
        "💥": "Run preprocessor",
        "📝": "Open new canvas",
    }

    @property
    def width_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_w_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_w_slider

    @property
    def height_slider(self):
        if self.is_img2img:
            return ControlNetUiGroup.a1111_context.img2img_h_slider
        else:
            return ControlNetUiGroup.a1111_context.txt2img_h_slider

    def __init__(
        self,
        is_img2img: bool,
        default_unit: external_code.ControlNetUnit,
    ):
        # Whether callbacks have been registered
        self.callbacks_registered: bool = False
        # Whether the render method on this object has been called
        self.ui_initialized: bool = False

        self.is_img2img = is_img2img
        self.default_unit = default_unit
        self.webcam_enabled = False
        self.webcam_mirrored = False

        self.enabled = None
        self.image = None
        self.generated_image_group = None
        self.generated_image = None
        self.mask_image_group = None
        self.mask_image = None
        self.create_canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.canvas_create_button = None
        self.canvas_cancel_button = None
        self.open_new_canvas_button = None
        self.webcam_enable = None
        self.webcam_mirror = None
        self.send_dimen_button = None
        self.pixel_perfect = None
        self.preprocessor_preview = None
        self.mask_upload = None
        self.type_filter = None
        self.module = None
        self.trigger_preprocessor = None
        self.model = None
        self.refresh_models = None
        self.weight = None
        self.guidance_start = None
        self.guidance_end = None
        self.processor_res = None
        self.threshold_a = None
        self.threshold_b = None
        self.control_mode = None
        self.resize_mode = None
        self.use_preview_as_input = None
        self.openpose_editor = None
        self.preset_panel = None
        self.upload_independent_img_in_img2img = None
        self.image_upload_panel = None
        self.save_detected_map = None
        self.input_mode = gr.State(InputMode.SIMPLE)
        self.hr_option = None

        self.dummy_update_trigger = None
        """For components without event subscriber, update this counter to trigger a sync update of UiControlNetUnit"""

        self.applying_preset_module = False
        self.applying_preset_sliders = False

        ControlNetUiGroup.all_ui_groups.append(self)

    def render(self, tabname: str, elem_id_tabname: str):
        """
        The pure HTML structure of a single ControlNetUnit.
        Calling this function will populate `self` with all
        gradio element declared in local scope.

        Args:
            tabname:
            elem_id_tabname:

        Returns:
            None
        """
        self.save_detected_map = gr.Checkbox(value=True, visible=False)
        self.dummy_update_trigger = gr.Number(value=0, visible=False)
        self.openpose_editor = OpenposeEditor()

        if self.is_img2img:
            with gr.Row(elem_classes="controlnet_img2img_options"):
                self.upload_independent_img_in_img2img = gr.Checkbox(
                    value=False,
                    label="Upload Independent Control Image",
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_same_img2img_checkbox",
                    elem_classes=["cnet-unit-same_img2img"],
                )

        with gr.Group(visible=(not self.is_img2img)) as self.image_upload_panel:
            with gr.Row(elem_classes=["cnet-image-row"], variant="panel"):
                with gr.Group(elem_classes=["cnet-input-image-group"]):
                    self.image = gr.Image(
                        value=None,
                        label="Control Image",
                        height=250,
                        source="upload",
                        brush_radius=20,
                        mirror_webcam=False,
                        type="numpy",
                        tool="sketch",
                        elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                        elem_classes=["cnet-image"],
                        brush_color=(
                            getattr(
                                shared.opts,
                                "img2img_inpaint_mask_brush_color",
                                None,
                            )
                        ),
                    )
                    self.image.preprocess = functools.partial(
                        svg_preprocess, preprocess=self.image.preprocess
                    )
                    self.openpose_editor.render_upload()

                with gr.Group(
                    visible=False, elem_classes=["cnet-generated-image-group"]
                ) as self.generated_image_group:
                    self.generated_image = gr.Image(
                        value=None,
                        label="Preprocessor Preview",
                        height=250,
                        elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                        elem_classes=["cnet-image"],
                        interactive=True,
                    )

                    with gr.Group(elem_classes=["cnet-generated-image-control-group"]):
                        self.openpose_editor.render_edit()
                        preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"
                        preview_close_button_js = f"document.querySelector('#{preview_check_elem_id} input[type=\\'checkbox\\']').click();"
                        gr.HTML(
                            value=f"""<a title="Close Preview" onclick="{preview_close_button_js}">Close</a>""",
                            visible=True,
                            elem_classes=["cnet-close-preview"],
                        )

                with gr.Group(
                    visible=False, elem_classes=["cnet-mask-image-group"]
                ) as self.mask_image_group:
                    self.mask_image = gr.Image(
                        value=None,
                        label="Mask",
                        height=250,
                        elem_id=f"{elem_id_tabname}_{tabname}_mask_image",
                        elem_classes=["cnet-mask-image"],
                        interactive=True,
                        brush_radius=20,
                        type="numpy",
                        tool="sketch",
                        brush_color=(
                            getattr(
                                shared.opts,
                                "img2img_inpaint_mask_brush_color",
                                None,
                            )
                        ),
                    )

            with gr.Accordion(
                label="Open New Canvas", visible=False
            ) as self.create_canvas:
                self.canvas_width = gr.Slider(
                    label="New Canvas Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_width",
                )
                self.canvas_height = gr.Slider(
                    label="New Canvas Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_height",
                )
                with gr.Row():
                    self.canvas_create_button = gr.Button(
                        value="Create New Canvas",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_create_button",
                    )
                    self.canvas_cancel_button = gr.Button(
                        value="Cancel",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button",
                    )

            with gr.Row(elem_classes="controlnet_image_controls"):
                gr.HTML(value="", elem_classes="controlnet_invert_warning")
                self.open_new_canvas_button = ToolButton(
                    value=ControlNetUiGroup.open_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.open_symbol],
                )
                self.webcam_enable = ToolButton(
                    value=ControlNetUiGroup.camera_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_enable",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.camera_symbol],
                )
                self.webcam_mirror = ToolButton(
                    value=ControlNetUiGroup.reverse_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_mirror",
                    tooltip=ControlNetUiGroup.tooltips[
                        ControlNetUiGroup.reverse_symbol
                    ],
                )
                self.send_dimen_button = ToolButton(
                    value=ControlNetUiGroup.tossup_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossup_symbol],
                )

        with FormRow(elem_classes=["controlnet_main_options"]):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=self.default_unit.enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                elem_classes=["cnet-unit-enabled"],
            )
            self.pixel_perfect = gr.Checkbox(
                label="Pixel Perfect",
                value=self.default_unit.pixel_perfect,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )
            self.preprocessor_preview = gr.Checkbox(
                label="Show Preview",
                value=False,
                elem_classes=["cnet-allow-preview"],
                elem_id=preview_check_elem_id,
                interactive=not self.is_img2img,
            )
            self.mask_upload = gr.Checkbox(
                label="Use Mask",
                value=False,
                elem_classes=["cnet-mask-upload"],
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_mask_upload_checkbox",
                interactive=not self.is_img2img,
            )
            self.use_preview_as_input = gr.Checkbox(
                label="Preview as Input",
                value=False,
                elem_classes=["cnet-preview-as-input"],
                visible=False,
            )

        with gr.Row(elem_classes=["controlnet_control_type", "controlnet_row"]):
            self.type_filter = gr.Radio(
                global_state.get_all_preprocessor_tags(),
                label=f"Control Type",
                value="All",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_radio",
                elem_classes="controlnet_control_type_filter_group",
            )

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            self.module = gr.Dropdown(
                label="Preprocessor",
                info="Set to None if the image is already processed (eg. greyscale depth map)",
                choices=global_state.get_all_preprocessor_names(),
                value=self.default_unit.module,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            self.trigger_preprocessor = ToolButton(
                value=ControlNetUiGroup.trigger_symbol,
                visible=not self.is_img2img,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                elem_classes=["cnet-run-preprocessor"],
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.trigger_symbol],
            )
            self.model = gr.Dropdown(
                label="Model",
                info="Ensure the model version matches the checkpoint version (SD1 / SDXL)",
                choices=global_state.get_all_controlnet_names(),
                value=self.default_unit.model,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            self.refresh_models = ToolButton(
                value=ControlNetUiGroup.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.refresh_symbol],
            )

        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            self.weight = gr.Slider(
                label=f"Control Weight",
                value=self.default_unit.weight,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            self.guidance_start = gr.Slider(
                label="Starting Control Step",
                value=self.default_unit.guidance_start,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider",
                elem_classes="controlnet_start_control_step_slider",
            )
            self.guidance_end = gr.Slider(
                label="Ending Control Step",
                value=self.default_unit.guidance_end,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider",
                elem_classes="controlnet_ending_control_step_slider",
            )

        with gr.Column():
            self.processor_res = gr.Slider(
                label="Preprocessor Resolution",
                value=self.default_unit.processor_res,
                minimum=64,
                maximum=2048,
                interactive=False,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            with gr.Row():
                self.threshold_a = gr.Slider(
                    label="Threshold A",
                    value=self.default_unit.threshold_a,
                    minimum=64,
                    maximum=1024,
                    interactive=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
                )
                self.threshold_b = gr.Slider(
                    label="Threshold B",
                    value=self.default_unit.threshold_b,
                    minimum=64,
                    maximum=1024,
                    interactive=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
                )

        self.control_mode = gr.Radio(
            label="Control Mode",
            choices=[e.value for e in external_code.ControlMode],
            value=self.default_unit.control_mode.value,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
            elem_classes="controlnet_control_mode_radio",
        )

        self.resize_mode = gr.Radio(
            label="Resize Mode",
            choices=[e.value for e in external_code.ResizeMode],
            value=self.default_unit.resize_mode.value,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
            elem_classes="controlnet_resize_mode_radio",
            visible=not self.is_img2img,
        )

        self.hr_option = gr.Radio(
            label="Hires-Fix Option",
            choices=[e.value for e in HiResFixOption],
            value=self.default_unit.hr_option.value,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_hr_option_radio",
            elem_classes="controlnet_hr_option_radio",
            visible=False,
        )

        self.preset_panel = ControlNetPresetUI(f"{elem_id_tabname}_{tabname}_")

        unit_args = [
            self.input_mode,
            self.use_preview_as_input,
            self.generated_image,
            self.mask_image,
            self.hr_option,
            self.enabled,
            self.module,
            self.model,
            self.weight,
            self.image,
            self.resize_mode,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.guidance_start,
            self.guidance_end,
            self.pixel_perfect,
            self.control_mode,
        ]

        unit = gr.State(self.default_unit)
        event_subscribers = []

        for comp in [*unit_args, self.dummy_update_trigger]:
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

        for event in event_subscribers:
            event(fn=UiControlNetUnit, inputs=unit_args, outputs=unit)

        button_id = f'{"img" if self.is_img2img else "txt"}2img_submit_button'
        getattr(ControlNetUiGroup.a1111_context, button_id).click(
            fn=UiControlNetUnit,
            inputs=unit_args,
            outputs=unit,
            queue=False,
        )

        self.register_core_callbacks()
        self.ui_initialized = True
        return unit

    def register_send_dimensions(self):
        """Register event handler for send dimension button"""

        def send_dimensions(image):
            def closest(num):
                rem = num % 64
                if rem <= 32:
                    return round(num - rem)
                else:
                    return round(num + (64 - rem))

            if image:
                dim = np.asarray(image.get("image"))
                return (closest(dim.shape[1]), closest(dim.shape[0]))
            else:
                return (gr.skip(), gr.skip())

        self.send_dimen_button.click(
            fn=send_dimensions,
            inputs=[self.image],
            outputs=[self.width_slider, self.height_slider],
            show_progress=False,
        )

    def register_webcam_toggle(self):
        def webcam_toggle():
            self.webcam_enabled = not self.webcam_enabled
            return {
                "value": None,
                "source": "webcam" if self.webcam_enabled else "upload",
                "__type__": "update",
            }

        self.webcam_enable.click(webcam_toggle, outputs=self.image, show_progress=False)

    def register_webcam_mirror_toggle(self):
        def mirror_toggle():
            self.webcam_mirrored = not self.webcam_mirrored
            return {"mirror_webcam": self.webcam_mirrored, "__type__": "update"}

        self.webcam_mirror.click(mirror_toggle, outputs=self.image, show_progress=False)

    def register_refresh_all_models(self):
        def refresh_all_models():
            global_state.update_controlnet_filenames()
            return gr.update(choices=global_state.get_all_controlnet_names())

        self.refresh_models.click(
            refresh_all_models,
            outputs=[self.model],
        )

    def register_build_sliders(self):
        def build_sliders(module: str, pixel_perfect: bool):
            logger.debug(f'Building Sliders for Module "{module}"')
            module = global_state.get_preprocessor(module)
            resolution_kwargs = module.slider_resolution.gradio_update_kwargs.copy()
            slider1_kwargs = module.slider_1.gradio_update_kwargs.copy()
            slider2_kwargs = module.slider_2.gradio_update_kwargs.copy()

            if pixel_perfect:
                resolution_kwargs["interactive"] = False
            if self.applying_preset_sliders:
                self.applying_preset_sliders = False
                resolution_kwargs.pop("value", None)
                slider1_kwargs.pop("value", None)
                slider2_kwargs.pop("value", None)

            return [
                gr.update(**resolution_kwargs),
                gr.update(**slider1_kwargs),
                gr.update(**slider2_kwargs),
                (
                    gr.update(value="None", interactive=False)
                    if module.do_not_need_model
                    else gr.update(interactive=True)
                ),
                gr.update(interactive=module.show_control_mode),
            ]

        inputs = [self.module, self.pixel_perfect]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.model,
            self.control_mode,
        ]

        self.module.change(
            fn=build_sliders,
            inputs=inputs,
            outputs=outputs,
            show_progress=False,
        )
        self.pixel_perfect.change(
            fn=build_sliders,
            inputs=inputs,
            outputs=outputs,
            show_progress=False,
        )

        def filter_selected(mode: str):
            logger.debug(f'Switching to Control Type "{mode}"')

            filtered_preprocessors = global_state.get_filtered_preprocessor_names(mode)
            filtered_cnet_names = global_state.get_filtered_controlnet_names(mode)

            if self.applying_preset_module:
                self.applying_preset_module = False
                return (
                    gr.update(choices=filtered_preprocessors),
                    gr.update(choices=filtered_cnet_names),
                    gr.skip(),
                )

            if mode == "All":
                default_preprocessor = filtered_preprocessors[0]
                default_controlnet_name = filtered_cnet_names[0]
            else:
                default_preprocessor = filtered_preprocessors[
                    1 if len(filtered_preprocessors) > 1 else 0
                ]
                default_controlnet_name = filtered_cnet_names[
                    1 if len(filtered_cnet_names) > 1 else 0
                ]

            return (
                gr.update(value=default_preprocessor, choices=filtered_preprocessors),
                gr.update(value=default_controlnet_name, choices=filtered_cnet_names),
                gr.update(value=NEW_PRESET),
            )

        self.type_filter.change(
            fn=filter_selected,
            inputs=[self.type_filter],
            outputs=[self.module, self.model, self.preset_panel.dropdown],
            show_progress=False,
        )

    def register_run_annotator(self):
        def run_annotator(image, module, p_res, a, b, w, h, pp, rm):
            if (image is None) or (module == "None"):
                return (
                    gr.update(value=None, visible=True),
                    gr.skip(),
                    *self.openpose_editor.update(""),
                )

            img = HWC3(image["image"])
            mask = HWC3(image["mask"])

            if not (mask > 16).any():
                mask = None

            if pp:
                p_res = external_code.pixel_perfect_resolution(
                    img,
                    target_H=h,
                    target_W=w,
                    resize_mode=external_code.resize_mode_from_value(rm),
                )

            logger.info(f"Preview Resolution: {p_res}")

            is_openpose = "openpose" in module
            if is_openpose:

                # Only OpenPose Preprocessor returns a JSON
                class JsonAcceptor:
                    def __init__(self):
                        self.value = ""

                    def accept(self, json_dict: dict):
                        self.value = json.dumps(json_dict)

                json_acceptor = JsonAcceptor()
                callback = json_acceptor.accept

            else:
                callback = None

            preprocessor = global_state.get_preprocessor(module)
            result = preprocessor(
                input_image=img,
                resolution=p_res,
                slider_1=a,
                slider_2=b,
                input_mask=mask,
                json_pose_callback=callback,
            )

            is_image = judge_image_type(result)
            if not is_image:
                result = img

            result = external_code.visualize_inpaint_mask(result)
            return (
                gr.update(value=result, visible=True, interactive=False),
                gr.update(value=True),
                *self.openpose_editor.update(
                    json_acceptor.value if is_openpose else ""
                ),
            )

        self.trigger_preprocessor.click(
            fn=run_annotator,
            inputs=[
                self.image,
                self.module,
                self.processor_res,
                self.threshold_a,
                self.threshold_b,
                self.width_slider,
                self.height_slider,
                self.pixel_perfect,
                self.resize_mode,
            ],
            outputs=[
                self.generated_image,
                self.preprocessor_preview,
                *self.openpose_editor.outputs(),
            ],
        )

    def register_shift_preview(self):
        def shift_preview(is_on):
            return (
                gr.skip() if is_on else gr.update(value=None),
                gr.update(visible=is_on),
                gr.update(visible=False),
                gr.skip() if is_on else gr.update(value=None),
            )

        self.preprocessor_preview.change(
            fn=shift_preview,
            inputs=[self.preprocessor_preview],
            outputs=[
                self.generated_image,
                self.generated_image_group,
                self.use_preview_as_input,
                self.openpose_editor.download_link,
            ],
            show_progress=False,
        )

    def register_create_canvas(self):
        self.open_new_canvas_button.click(
            lambda: gr.update(visible=True),
            outputs=self.create_canvas,
            show_progress=False,
        )
        self.canvas_cancel_button.click(
            lambda: gr.update(visible=False),
            outputs=self.create_canvas,
            show_progress=False,
        )

        def fn_canvas(h, w):
            return (
                gr.update(value=np.zeros(shape=(h, w, 3), dtype=np.uint8)),
                gr.update(visible=False),
            )

        self.canvas_create_button.click(
            fn=fn_canvas,
            inputs=[self.canvas_height, self.canvas_width],
            outputs=[self.image, self.create_canvas],
            show_progress=False,
        )

    def register_img2img_same_input(self):
        def upload_independent(x):
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=x),
            )

        self.upload_independent_img_in_img2img.change(
            fn=upload_independent,
            inputs=[self.upload_independent_img_in_img2img],
            outputs=[
                self.image,
                self.preprocessor_preview,
                self.image_upload_panel,
                self.trigger_preprocessor,
                self.resize_mode,
            ],
            show_progress=False,
        )

    def register_shift_hr_options(self):
        ControlNetUiGroup.a1111_context.txt2img_enable_hr.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=[ControlNetUiGroup.a1111_context.txt2img_enable_hr],
            outputs=[self.hr_option],
            show_progress=False,
        )

    def register_shift_upload_mask(self):
        """Controls whether the upload mask input should be visible"""

        def on_checkbox_click(checked: bool, h: int, w: int):
            if not checked:
                # Clear mask_image if unchecked
                return (gr.update(visible=False), gr.update(value=None))
            else:
                # Init an empty canvas the same size as the generation target
                empty_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
                return (gr.update(visible=True), gr.update(value=empty_canvas))

        self.mask_upload.change(
            fn=on_checkbox_click,
            inputs=[self.mask_upload, self.height_slider, self.width_slider],
            outputs=[self.mask_image_group, self.mask_image],
            show_progress=False,
        )

        if self.is_img2img:
            self.upload_independent_img_in_img2img.change(
                fn=lambda checked: (gr.update(interactive=checked),) * 2,
                inputs=[self.upload_independent_img_in_img2img],
                outputs=[self.preprocessor_preview, self.mask_upload],
                show_progress=False,
            )

    def register_clear_preview(self):
        def clear_preview(preview_as_input):
            if preview_as_input:
                logger.info("Preview as Input is disabled")
            return gr.update(value=False), gr.update(value=None)

        event_subscribers = []
        for comp in (
            self.pixel_perfect,
            self.module,
            self.image,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.upload_independent_img_in_img2img,
        ):
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

        for event in event_subscribers:
            event(
                fn=clear_preview,
                inputs=self.use_preview_as_input,
                outputs=[self.use_preview_as_input, self.generated_image],
                show_progress=False,
            )

    def register_core_callbacks(self):
        """
        Register core callbacks that only involves gradio components defined within this ui group
        """
        self.register_webcam_toggle()
        self.register_webcam_mirror_toggle()
        self.register_refresh_all_models()
        self.register_build_sliders()
        self.register_shift_preview()
        self.register_create_canvas()
        self.register_clear_preview()
        self.openpose_editor.register_callbacks(
            self.generated_image,
            self.use_preview_as_input,
            self.model,
        )
        assert self.type_filter is not None
        self.preset_panel.register_callbacks(
            self,
            self.type_filter,
            *[
                getattr(self, key)
                for key in external_code.ControlNetUnit.infotext_fields()
            ],
        )
        if self.is_img2img:
            self.register_img2img_same_input()

    def register_callbacks(self):
        """Register callbacks that involves A1111 context gradio components"""
        # Prevent recursion
        if self.callbacks_registered:
            return

        self.callbacks_registered = True
        self.register_send_dimensions()
        self.register_run_annotator()
        self.register_shift_upload_mask()
        if not self.is_img2img:
            self.register_shift_hr_options()

    @staticmethod
    def reset():
        ControlNetUiGroup.a1111_context = A1111Context()
        ControlNetUiGroup.callbacks_registered = False
        ControlNetUiGroup.all_ui_groups = []

    @staticmethod
    def try_register_all_callbacks():
        unit_count = shared.opts.data.get("control_net_unit_count", 3)
        total = unit_count * 2  # txt2img + img2img
        if (total == len(ControlNetUiGroup.all_ui_groups)) and all(
            ((g.ui_initialized) and (not g.callbacks_registered))
            for g in ControlNetUiGroup.all_ui_groups
        ):
            for ui_group in ControlNetUiGroup.all_ui_groups:
                ui_group.register_callbacks()

            logger.info("ControlNet UI callback registered")
            ControlNetUiGroup.all_callbacks_registered = True

    @staticmethod
    def on_after_component(component, **kwargs):
        """Register the A1111 component"""
        ControlNetUiGroup.a1111_context.set_component(component)
        if (
            ControlNetUiGroup.a1111_context.ui_initialized
            and not ControlNetUiGroup.all_callbacks_registered
        ):
            ControlNetUiGroup.try_register_all_callbacks()
