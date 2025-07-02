from __future__ import annotations

import enum
from collections import namedtuple

from modules import cache, errors, hashes, sd_models, shared

NetworkWeights = namedtuple("NetworkWeights", ["network_key", "sd_key", "w", "sd_module"])

metadata_tags_order = {
    "ss_sd_model_name": 1,
    "ss_resolution": 2,
    "ss_clip_skip": 3,
    "ss_num_train_images": 10,
    "ss_tag_frequency": 20,
}


class SDVersion(enum.Enum):
    Unknown = -1
    SD1 = 1
    SDXL = 3


class NetworkOnDisk:
    def __init__(self, name: str, filename: str):
        self.name: str = name
        self.filename: str = filename
        self.metadata: dict[str, str] = {}
        self.is_safetensors: bool = filename.lower().endswith(".safetensors")

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)
            return metadata

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file("safetensors-metadata", f"lora/{self.name}", filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.alias: str = self.metadata.get("ss_output_name", self.name)

        self.hash: str = None
        self.shorthash: str = None
        self.set_hash(self.metadata.get("sshs_model_hash") or hashes.sha256_from_cache(self.filename, f"lora/{self.name}", use_addnet_hash=self.is_safetensors) or "")

        self.sd_version: "SDVersion" = self.detect_version()

    def detect_version(self):
        if str(self.metadata.get("ss_base_model_version", "")).startswith("sdxl_"):
            return SDVersion.SDXL
        elif len(self.metadata):
            return SDVersion.SD1

        return SDVersion.Unknown

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            import networks

            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, f"lora/{self.name}", use_addnet_hash=self.is_safetensors) or "")

    def get_alias(self):
        import networks

        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.bundle_embeddings = {}
        self.mtime = None

        self.mentioned_name = None
        """the text that was used to add the network to prompt - can be either name or an alias"""
