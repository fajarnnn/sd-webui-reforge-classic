import os

import ldm_patched.controlnet
import ldm_patched.modules.utils
import torch
from ldm_patched.modules.controlnet import ControlLora, ControlNet, load_t2i_adapter
from modules_forge.controlnet import apply_controlnet_advanced
from modules_forge.shared import add_supported_control_model


class ControlModelPatcher:

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        return None

    def __init__(self, model_patcher=None):
        self.model_patcher = model_patcher
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        pass

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        pass

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        pass


class ControlNetPatcher(ControlModelPatcher):

    @staticmethod
    def try_build_from_state_dict(controlnet_data, ckpt_path):
        if "lora_controlnet" in controlnet_data:
            return ControlNetPatcher(ControlLora(controlnet_data))

        controlnet_config = None

        if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:
            unet_dtype = ldm_patched.modules.model_management.unet_dtype()
            controlnet_config = ldm_patched.modules.model_detection.unet_config_from_diffusers_unet(controlnet_data, unet_dtype)
            diffusers_keys = ldm_patched.modules.utils.unet_to_diffusers(controlnet_config)
            diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
            diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"
            suffix = (".weight", ".bias")

            count = 0
            loop = True
            while loop:
                for s in suffix:
                    k_in = f"controlnet_down_blocks.{count}{s}"
                    k_out = f"zero_convs.{count}.0{s}"
                    if k_in not in controlnet_data:
                        loop = False
                        break
                    diffusers_keys[k_in] = k_out
                count += 1

            count = 0
            loop = True
            while loop:
                for s in suffix:
                    if count == 0:
                        k_in = f"controlnet_cond_embedding.conv_in{s}"
                    else:
                        k_in = f"controlnet_cond_embedding.blocks.{count - 1}{s}"
                    k_out = f"input_hint_block.{count * 2}{s}"
                    if k_in not in controlnet_data:
                        k_in = f"controlnet_cond_embedding.conv_out{s}"
                        loop = False
                    diffusers_keys[k_in] = k_out
                count += 1

            new_sd = {}
            for k in diffusers_keys:
                if k in controlnet_data:
                    new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

            # leftover_keys = controlnet_data.keys()
            # if len(leftover_keys) > 0:
            #     logger.debug(f"leftover keys: {leftover_keys}")
            controlnet_data = new_sd

        pth_key = "control_model.zero_convs.0.0.weight"
        pth = False
        key = "zero_convs.0.0.weight"
        if pth_key in controlnet_data:
            pth = True
            key = pth_key
            prefix = "control_model."
        elif key in controlnet_data:
            prefix = ""
        else:
            net = load_t2i_adapter(controlnet_data)
            if net is None:
                return None
            return ControlNetPatcher(net)

        if controlnet_config is None:
            unet_dtype = ldm_patched.modules.model_management.unet_dtype()
            controlnet_config = ldm_patched.modules.model_detection.model_config_from_unet(controlnet_data, prefix, unet_dtype, True).unet_config
        load_device = ldm_patched.modules.model_management.get_torch_device()
        manual_cast_dtype = ldm_patched.modules.model_management.unet_manual_cast(unet_dtype, load_device)
        if manual_cast_dtype is not None:
            controlnet_config["operations"] = ldm_patched.modules.ops.manual_cast
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = controlnet_data[f"{prefix}input_hint_block.0.weight"].shape[1]
        control_model = ldm_patched.controlnet.cldm.ControlNet(**controlnet_config)

        if pth:
            if "difference" in controlnet_data:
                print("WARNING: Please use an official ControlNet model for the best performance")

            class WeightsLoader(torch.nn.Module):
                pass

            w = WeightsLoader()
            w.control_model = control_model
            missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
        else:
            missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

        # logger.debug(missing, unexpected)

        global_average_pooling = False
        filename = os.path.splitext(ckpt_path)[0]
        if "shuffle" in filename:
            # TODO: better way of enabling global_average_pooling
            global_average_pooling = True

        control = ControlNet(
            control_model,
            global_average_pooling=global_average_pooling,
            load_device=load_device,
            manual_cast_dtype=manual_cast_dtype,
        )

        return ControlNetPatcher(control)

    def __init__(self, model_patcher):
        super().__init__(model_patcher)

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = apply_controlnet_advanced(
            unet=unet,
            controlnet=self.model_patcher,
            image_bchw=cond,
            strength=self.strength,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            positive_advanced_weighting=self.positive_advanced_weighting,
            negative_advanced_weighting=self.negative_advanced_weighting,
            advanced_frame_weighting=self.advanced_frame_weighting,
            advanced_sigma_weighting=self.advanced_sigma_weighting,
            advanced_mask_weighting=self.advanced_mask_weighting,
        )

        process.sd_model.forge_objects.unet = unet


add_supported_control_model(ControlNetPatcher)
