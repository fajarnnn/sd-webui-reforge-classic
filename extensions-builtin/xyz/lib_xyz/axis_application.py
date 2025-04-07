from modules import sd_models, sd_vae, shared
from modules.processing import StableDiffusionProcessing
from modules.shared import opts

from .utils import find_vae


def apply_field(field):
    def fun(p, x, xs):
        setattr(p, field, x)

    return fun


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        raise RuntimeError(f"Prompt S/R did not find {xs[0]} in prompt or negative prompt.")

    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_order(p, x, xs):
    token_order = []

    # Initially grab the tokens from the prompt, so they can be replaced in order of earliest seen
    for token in x:
        token_order.append((p.prompt.find(token), token))

    token_order.sort(key=lambda t: t[0])

    prompt_parts = []

    # Split the prompt up, taking out the tokens
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token) :]

    # Rebuild the prompt with the tokens in the order we want
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_checkpoint(p, x, xs):
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        raise RuntimeError(f"Unknown checkpoint: {x}")
    p.override_settings["sd_model_checkpoint"] = info.name


def apply_clip_skip(p, x, xs):
    opts.data["CLIP_stop_at_last_layers"] = x


def apply_upscale_latent_space(p, x, xs):
    if x.lower().strip() != "0":
        opts.data["use_scale_latent_for_hires_fix"] = True
    else:
        opts.data["use_scale_latent_for_hires_fix"] = False


def apply_vae(p, x, xs):
    sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))


def apply_styles(p: StableDiffusionProcessing, x: str, _):
    p.styles.extend(x.split(","))


def apply_uni_pc_order(p, x, xs):
    opts.data["uni_pc_order"] = min(x, p.steps - 1)


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == "codeformer":
        is_active = True
        p.face_restoration_model = "CodeFormer"
    elif opt == "gfpgan":
        is_active = True
        p.face_restoration_model = "GFPGAN"
    else:
        is_active = opt in ("true", "yes", "y", "1")

    p.restore_faces = is_active


def apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        p.override_settings[field] = x

    return fun


def do_nothing(p, x, xs):
    pass
