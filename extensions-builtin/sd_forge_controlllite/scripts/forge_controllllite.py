from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_controllllite.lib_controllllite import LLLiteLoader


class ControlLLLitePatcher(ControlModelPatcher):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict

    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any("lllite" in k for k in state_dict.keys()):
            return None
        return ControlLLLitePatcher(state_dict)

    def process_before_every_sampling(self, process, cond, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = LLLiteLoader.load_lllite(
            model=unet,
            state_dict=self.state_dict,
            cond_image=cond,
            strength=self.strength,
            steps=process.steps,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
        )

        process.sd_model.forge_objects.unet = unet


add_supported_control_model(ControlLLLitePatcher)
