from os.path import join

from modules import paths
from modules.paths_internal import normalized_filepath


def preload(parser):
    parser.add_argument(
        "--lora-dir",
        type=normalized_filepath,
        help="Path to directory with LoRA networks",
        default=join(paths.models_path, "Lora"),
    )
