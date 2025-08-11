from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.sd_models import CheckpointInfo
    from modules_forge.forge_loader import ForgeObjects


class WebuiSdModel:
    """This class is only used for typing"""

    sd_checkpoint_info: "CheckpointInfo"
    """structure with additional information about the file with model's weights"""

    sd_model_checkpoint: str
    """path to the file on disk that model weights were obtained from"""

    sd_model_hash: str
    """
    short hash, 10 first characters of SHA1 hash of the model file;
    may be None if --no-hashing flag is used
    """

    ztsnr: bool
    """Zero Terminal SNR"""

    is_sd1: bool
    """True if the model's architecture is SD 1.x"""

    is_sd2: bool
    """True if the model's architecture is SD 2.x"""

    is_sdxl: bool
    """True if the model's architecture is SD XL"""

    forge_objects_original: "ForgeObjects"
    """The model patchers freshly created in `forge_loader`"""

    forge_objects: "ForgeObjects"
    """The model patchers actively used during generation"""

    forge_objects_after_applying_lora: "ForgeObjects"
    """The model patchers after LoRA is applied; `None` if not using LoRA"""
