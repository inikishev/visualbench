from typing import TYPE_CHECKING

import numpy as np
import torch

from .padding import pad_to_shape
from .renderer import OpenCVRenderer, make_hw3, render_frames
from .format import tonumpy

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _maybe_progress(x, enable):
    if enable:
        from tqdm import tqdm
        return tqdm(x)
    return x

def _repeat_to_largest(images: list[np.ndarray]):
    """for each elemnt of x if both height and width are 2 or more times smaller than largest element repeat them

    x must be hwc"""
    max_h, max_w = np.max([i.shape for i in images], axis = 0)[:-1]
    for i,img in enumerate(images.copy()):
        h,w = img.shape[:-1]
        ratio = min(max_h/h, max_w/w)
        if ratio >= 2:
            images[i] = np.repeat(np.repeat(img, ratio, 0), ratio, 1)
    return images

def _make_collage(images: list[np.ndarray]):
    """make a collage from images"""
    images = [make_hw3(i) for i in images]
    if len(images) == 1: return images[0]

    images = _repeat_to_largest([i for i in images])
    max_shape = np.max([i.shape for i in images], axis = 0)
    max_shape[:-1] += 2 # add 2 pixel to spatial dims
    stacked = np.stack([pad_to_shape(i, max_shape, mode = 'constant', value=128) for i in images])
    # it is now (image, H, W, 3)

    # compose them
    ncols = len(stacked) ** (0.6 * (max_shape[0]/max_shape[1]))
    nrows = round(len(stacked) / ncols)
    ncols = round(ncols)
    nrows = max(nrows, 1)
    ncols = max(ncols, 1)
    c = True
    while nrows * ncols < len(stacked):
        if c: ncols += 1
        else: nrows += 1
        c = not c
    n_tiles = nrows * ncols
    if len(stacked) < n_tiles: stacked = np.concatenate([stacked, np.zeros_like(stacked[:n_tiles - len(stacked)])])
    stacked = stacked.reshape(nrows, ncols, *max_shape)
    stacked = np.concatenate(np.concatenate(stacked, 1), 1)
    return stacked


def _check_image(image: np.ndarray | torch.Tensor, name=None) -> np.ndarray | torch.Tensor:
    """checks image also returns squeezed"""
    if isinstance(image, np.ndarray): lib = np
    elif isinstance(image, torch.Tensor): lib = torch
    else: raise TypeError(f"Invalid image {name}, type must be np.ndarray or torch.Tensor, got {type(image)}")
    if image.dtype != lib.uint8: raise TypeError(f"Invalid image {name}, dtype must be uint8 but got {image.dtype}")
    if image.ndim > 3: image = lib.squeeze(image) # type:ignore
    if image.ndim not in (2, 3):
        raise ValueError(f"Invalid image {name}, must be 2D or 3D but got shape {image.shape}")
    return image

def _isclose(x, y, tol=2):
    return y-tol <= x <= y+tol

@torch.no_grad
def _render(self: "Benchmark", file: str, fps: int = 60, scale: int | float = 1, progress=True,):
    """renders a video of how current and best solution evolves on each step, if applicable to this benchmark."""

    logger_images = {}
    lowest_images = {}
    length = max(len(v) for v in self.logger.values())

    # initialize all keys
    for key, value in self.logger.items():
        if key in self._image_keys:
            if (not self._plot_perturbed) and key.endswith(' (perturbed)'): continue
            images_list = logger_images[key] = list(value.values())
            if len(images_list) != 0: _check_image(images_list[0])
            assert _isclose(len(logger_images[key]), length), f'images must be logged on all steps, "{key}" was logged {len(logger_images[key])} times, expected {length} times'
            while len(logger_images[key]) < length:
                logger_images[key].append(logger_images[key][-1])
            while len(logger_images[key]) > length:
                logger_images[key] = logger_images[key][:-1]

        if key in self._image_lowest_keys:
            lowest_images[key] = logger_images[key][0]


    for key, value in self._reference_images.items():
        _check_image(value, f'reference_images[{key}]')

    if len(logger_images) + len(lowest_images) == 0:
        raise NotImplementedError(f'Solution plotting is not implemented for {self.__class__.__name__}')

    with OpenCVRenderer(file, fps = fps, scale=scale) as renderer:
        lowest_loss = float('inf')

        for step, loss in enumerate(_maybe_progress(list(self.logger['train loss'].values()), enable=progress)):
            # add current and best image
            images: list[np.ndarray | torch.Tensor] = []

            # add reference image
            for image in self._reference_images.values(): images.append(image)

            # check if new params are better
            if loss <= lowest_loss:
                lowest_loss = loss

                # set to new best images
                for key in lowest_images:
                    if key in logger_images:
                        lowest_images[key] = logger_images[key][step]

            # add logger images
            for key, value in logger_images.items(): images.append(value[step])

            # add best images
            for image in lowest_images.values(): images.append(image)

            # make a collage
            collage = _make_collage([tonumpy(i) for i in images])
            renderer.write(collage)

