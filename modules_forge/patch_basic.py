import os
import warnings
from functools import wraps

import safetensors
import torch


def build_loaded(module, loader_name):
    original_loader_name = f"{loader_name}_origin"

    if not hasattr(module, original_loader_name):
        setattr(module, original_loader_name, getattr(module, loader_name))

    original_loader = getattr(module, original_loader_name)

    @wraps(original_loader)
    def loader(*args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                return original_loader(*args, **kwargs)
        except Exception:
            exc = "\n"
            for path in list(args) + list(kwargs.values()):
                if isinstance(path, str) and os.path.isfile(path):
                    exc += f'Failed to read file "{path}"\n'
                    backup_file = f"{path}.corrupted"
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.replace(path, backup_file)
                    exc += f'Forge has moved the corrupted file to "{backup_file}"\n'
                    exc += f"Please try downloading the model again\n"
            print(exc)
            raise ValueError from None

    setattr(module, loader_name, loader)


def patch_all_basics():
    build_loaded(safetensors.torch, "load_file")
    build_loaded(torch, "load")
