from modules.safe import unsafe_torch_load
from modules import processing

from PIL import Image, ImageFilter, ImageOps
from typing import Optional, Callable
import safetensors.torch
import numpy as np
import functools
import logging
import base64
import torch
import time
import cv2
import os
import io

from lib_controlnet.lvminthin import lvmin_thin, nake_nms
from lib_controlnet.logging import logger
from lib_controlnet import external_code

try:
    from reportlab.graphics import renderPM
    from svglib.svglib import svg2rlg
except ImportError:
    svgSupport = False
else:
    svgSupport = True


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = unsafe_torch_load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logger.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def get_state_dict(d):
    return d.get("state_dict", d)


def timer_decorator(func):
    """Time the decorated function and output the result to debug logger"""
    if logger.level != logging.DEBUG:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # Only report function that are significant enough
        if duration > 1e-3:
            logger.debug(f"{func.__name__} ran in: {duration:.3f} sec")
        return result

    return wrapper


class TimeMeta(type):
    """
    Metaclass to record execution time on all methods of the child class
    """

    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = timer_decorator(attr_value)
        return super().__new__(cls, name, bases, attrs)


@functools.lru_cache(maxsize=1, typed=False)
def _blank_mask() -> str:
    with io.BytesIO() as buffer:
        black = Image.new("RGB", (4, 4))
        black.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
    return b64


def svg_preprocess(inputs: dict, preprocess: Callable):
    if not inputs:
        return None

    if svgSupport and inputs["image"].startswith("data:image/svg+xml;base64,"):
        svg_data = base64.b64decode(inputs["image"].replace("data:image/svg+xml;base64,", ""))
        drawing = svg2rlg(io.BytesIO(svg_data))
        png_data = renderPM.drawToString(drawing, fmt="PNG")
        encoded_string = base64.b64encode(png_data)
        base64_str = str(encoded_string, "utf-8")
        base64_str = "data:image/png;base64," + base64_str
        inputs["image"] = base64_str

    if inputs.get("mask", None) is None:
        inputs["mask"] = _blank_mask()

    return preprocess(inputs)


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]


def align_dim_latent(x: int) -> int:
    """
    Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel
    i.e. 1 latent unit == 8 pixel unit
    """
    return (x // 8) * 8


def prepare_mask(mask: Image.Image, p: processing.StableDiffusionProcessing) -> Image.Image:
    """
    Prepare an image mask for the inpainting process.

    This function takes as input a PIL Image object and an instance of the
    StableDiffusionProcessing class, and performs the following steps to prepare the mask:

    1. Convert the mask to grayscale (mode "L").
    2. If the 'inpainting_mask_invert' attribute of the processing instance is True,
       invert the mask colors.
    3. If the 'mask_blur' attribute of the processing instance is greater than 0,
       apply a Gaussian blur to the mask with a radius equal to 'mask_blur'.

    Args:
        mask (Image.Image): The input mask as a PIL Image object.
        p (processing.StableDiffusionProcessing): An instance of the StableDiffusionProcessing class
                                                   containing the processing parameters.

    Returns:
        mask (Image.Image): The prepared mask as a PIL Image object.
    """
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)

    if hasattr(p, "mask_blur_x"):
        if getattr(p, "mask_blur_x", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
            mask = Image.fromarray(np_mask)
        if getattr(p, "mask_blur_y", 0) > 0:
            np_mask = np.array(mask)
            kernel_size = 2 * int(2.5 * p.mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
            mask = Image.fromarray(np_mask)
    else:
        if getattr(p, "mask_blur", 0) > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    return mask


def set_numpy_seed(p: processing.StableDiffusionProcessing) -> Optional[int]:
    """
    Set the random seed for NumPy based on the provided parameters.

    Args:
        p (processing.StableDiffusionProcessing): The instance of the StableDiffusionProcessing class.

    Returns:
        Optional[int]: The computed random seed if successful, or None if an exception occurs.

    This function sets the random seed for NumPy using the seed and subseed values from the given instance of
    StableDiffusionProcessing. If either seed or subseed is -1, it uses the first value from `all_seeds`.
    Otherwise, it takes the maximum of the provided seed value and 0.

    The final random seed is computed by adding the seed and subseed values, applying a bitwise AND operation
    with 0xFFFFFFFF to ensure it fits within a 32-bit integer.
    """
    try:
        tmp_seed = int(p.all_seeds[0] if p.seed == -1 else max(int(p.seed), 0))
        tmp_subseed = int(p.all_seeds[0] if p.subseed == -1 else max(int(p.subseed), 0))
        seed = (tmp_seed + tmp_subseed) & 0xFFFFFFFF
        np.random.seed(seed)
        return seed
    except Exception as e:
        logger.warning(e)
        logger.warning("Warning: Failed to use consistent random seed.")
        return None


def safe_numpy(x):
    """A very safe method to make sure that Mac works"""
    y = x
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def high_quality_resize(x, size):
    """
    Written by lvmin
    Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges
    """

    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
        new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
        unique_color_count = len(get_unique_axis0(x.reshape(-1, x.shape[2])))
        is_one_pixel_edge = False
        is_binary = False
        if unique_color_count == 2:
            is_binary = np.min(x) < 16 and np.max(x) > 240
            if is_binary:
                xc = x
                xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                all_edge_count = np.where(x > 127)[0].shape[0]
                is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

        if 2 < unique_color_count < 200:
            interpolation = cv2.INTER_NEAREST
        elif new_size_is_smaller:
            interpolation = cv2.INTER_AREA
        else:
            # Must be CUBIC because we now use nms
            interpolation = cv2.INTER_CUBIC  # NEVER CHANGE THIS

        y = cv2.resize(x, size, interpolation=interpolation)

        if is_binary:
            y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
            if is_one_pixel_edge:
                y = nake_nms(y)
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                y = lvmin_thin(y, prunings=new_size_is_bigger)
            else:
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            y = np.stack([y] * 3, axis=2)
    else:
        y = x

    return y


def crop_and_resize_image(detected_map, resize_mode, h, w, fill_border_with_255=False):
    if resize_mode == external_code.ResizeMode.RESIZE:
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == external_code.ResizeMode.OUTER_FIT:
        k = min(k0, k1)
        borders = np.concatenate(
            [
                detected_map[0, :, :],
                detected_map[-1, :, :],
                detected_map[:, 0, :],
                detected_map[:, -1, :],
            ],
            axis=0,
        )
        high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
        if fill_border_with_255:
            high_quality_border_color = np.zeros_like(high_quality_border_color) + 255
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h : pad_h + h, pad_w : pad_w + w]
        detected_map = safe_numpy(detected_map)
        return detected_map


def judge_image_type(img):
    return isinstance(img, np.ndarray) and img.ndim == 3 and int(img.shape[2]) in [3, 4]
