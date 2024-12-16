from modules import shared, paths
import os


sd_configs_path = shared.sd_configs_path
sd_repo_configs_path = os.path.join(paths.paths['Stable Diffusion'], "configs", "stable-diffusion")
sd_xl_repo_configs_path = os.path.join(paths.paths['Stable Diffusion XL'], "configs", "inference")

config_default = shared.sd_default_config
config_sd2 = os.path.join(sd_repo_configs_path, "v2-inference.yaml")
config_sd2v = os.path.join(sd_repo_configs_path, "v2-inference-v.yaml")
config_sd2_inpainting = os.path.join(sd_repo_configs_path, "v2-inpainting-inference.yaml")
config_sdxl = os.path.join(sd_xl_repo_configs_path, "sd_xl_base.yaml")
config_sdxlv = os.path.join(sd_configs_path, "sd_xl_v.yaml")
config_sdxl_refiner = os.path.join(sd_xl_repo_configs_path, "sd_xl_refiner.yaml")
config_sdxl_inpainting = os.path.join(sd_configs_path, "sd_xl_inpaint.yaml")
config_depth_model = os.path.join(sd_repo_configs_path, "v2-midas-inference.yaml")
config_unclip = os.path.join(sd_repo_configs_path, "v2-1-stable-unclip-l-inference.yaml")
config_unopenclip = os.path.join(sd_repo_configs_path, "v2-1-stable-unclip-h-inference.yaml")
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
config_instruct_pix2pix = os.path.join(sd_configs_path, "instruct-pix2pix.yaml")


def guess_model_config_from_state_dict(sd, filename):
    sd2_cond_proj_weight = sd.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight', None)
    diffusion_model_input = sd.get('model.diffusion_model.input_blocks.0.0.weight', None)
    sd2_variations_weight = sd.get('embedder.model.ln_final.weight', None)

    if sd.get('conditioner.embedders.1.model.ln_final.weight', None) is not None:
        if diffusion_model_input.shape[1] == 9:
            return config_sdxl_inpainting
        elif 'v_pred' in sd:
            return config_sdxlv
        else:
            return config_sdxl

    if sd.get('conditioner.embedders.0.model.ln_final.weight', None) is not None:
        return config_sdxl_refiner
    elif sd.get('depth_model.model.pretrained.act_postprocess3.0.project.0.bias', None) is not None:
        return config_depth_model
    elif sd2_variations_weight is not None and sd2_variations_weight.shape[0] == 768:
        return config_unclip
    elif sd2_variations_weight is not None and sd2_variations_weight.shape[0] == 1024:
        return config_unopenclip

    if sd2_cond_proj_weight is not None and sd2_cond_proj_weight.shape[1] == 1024:
        if diffusion_model_input.shape[1] == 9:
            return config_sd2_inpainting
        elif 'v_pred' in sd:
            return config_sd2v
        else:
            return config_sd2

    if diffusion_model_input is not None:
        if diffusion_model_input.shape[1] == 9:
            return config_inpainting
        if diffusion_model_input.shape[1] == 8:
            return config_instruct_pix2pix

    return config_default


def find_checkpoint_config(state_dict, info):
    if info is None:
        return guess_model_config_from_state_dict(state_dict, "")

    config = find_checkpoint_config_near_filename(info)
    if config is not None:
        return config

    return guess_model_config_from_state_dict(state_dict, info.filename)


def find_checkpoint_config_near_filename(info):
    if info is None:
        return None

    config = f"{os.path.splitext(info.filename)[0]}.yaml"
    if os.path.exists(config):
        return config

    return None

