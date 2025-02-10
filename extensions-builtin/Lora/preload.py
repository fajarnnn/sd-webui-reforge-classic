from modules.paths_internal import normalized_filepath
from modules import paths
import os


def preload(parser):
    parser.add_argument(
        "--lora-dir",
        type=normalized_filepath,
        help="Path to directory with Lora networks.",
        default=os.path.join(paths.models_path, "Lora"),
    )
