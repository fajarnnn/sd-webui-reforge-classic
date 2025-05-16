import logging

from modules import sd_samplers_cfgpp, sd_samplers_common, sd_samplers_kdiffusion


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name, scheduler_name=None):
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(sd_samplers_cfgpp, f"sample_{self.sampler_name}", None)
        if sampler_function is None:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        super().__init__(sampler_function, sd_model, None)

    def sample(self, p, *args, **kwargs):
        # self.scheduler_name = p.scheduler
        if p.cfg_scale > 2.0:
            logging.warning("Low CFG is recommended when using CFG++ samplers")
        return super().sample(p, *args, **kwargs)

    def sample_img2img(self, p, *args, **kwargs):
        # self.scheduler_name = p.scheduler
        if p.cfg_scale > 2.0:
            logging.warning("Low CFG is recommended when using CFG++ samplers")
        return super().sample_img2img(p, *args, **kwargs)


def build_constructor(sampler_name):
    def constructor(model):
        return AlterSampler(model, sampler_name)

    return constructor


samplers_data_alter = [
    sd_samplers_common.SamplerData("DPM++ 2M CFG++", build_constructor(sampler_name="dpmpp_2m_cfg_pp"), ["dpmpp_2m_cfg_pp"], {}),
    sd_samplers_common.SamplerData("DPM++ SDE CFG++", build_constructor(sampler_name="dpmpp_sde_cfg_pp"), ["dpmpp_sde_cfg_pp"], {}),
    sd_samplers_common.SamplerData("DPM++ 3M SDE CFG++", build_constructor(sampler_name="dpmpp_3m_sde_cfg_pp"), ["dpmpp_3m_sde_cfg_pp"], {}),
    sd_samplers_common.SamplerData("Euler a CFG++", build_constructor(sampler_name="euler_ancestral_cfg_pp"), ["euler_ancestral_cfg_pp"], {}),
    sd_samplers_common.SamplerData("Euler CFG++", build_constructor(sampler_name="euler_cfg_pp"), ["euler_cfg_pp"], {}),
]
