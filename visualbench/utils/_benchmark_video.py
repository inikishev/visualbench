import os
import textwrap
from typing import TYPE_CHECKING
from importlib.util import find_spec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .format import tonumpy
from .padding import pad_to_shape
from .python_tools import format_number
from .renderer import OpenCVRenderer, make_hw3, render_frames


if TYPE_CHECKING:
    from ..benchmark import Benchmark

GIF_POST_PBAR_MESSAGE = "rendering GIF, this can take some time (tip: saving as mp4 is much faster)"

def _better_load_default(size):
    """loads default font out of more fonts beacuse default one is so bad its crazy"""
    for font_path in ("arial.ttf", "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"):
        try:
            return ImageFont.truetype(font_path, size=size)
        except IOError:
            pass
    return ImageFont.load_default(size=size)

def _add_title(image: np.ndarray, title: str, size_per_px:float=0.04, wrap=True,):
    """image is (H,W,3) uint8"""
    h, w, c = image.shape
    assert c == 3, f"shape is wrong and it is {image.shape} and it should be (H, W, 3)"

    font_size = max(int(size_per_px * w), 7)

    if wrap:
        title = '\n'.join(textwrap.wrap(title, width=(w//font_size)*2))

    nlines = title.count("\n") + 1
    bar_size = (font_size * 1.05) * nlines
    font = _better_load_default(size=font_size)

    # calculate padding with fake draw
    # removed because value can change leading to different resolution
    # text_bbox = ImageDraw.Draw(Image.new('RGB', (0,0))).multiline_textbbox((0,0), title, font=font)
    # text_height = text_bbox[3] - text_bbox[1]
    # font_size = 25 text_height = 20
    pad_height = font_size*0.7 + bar_size

    pad = np.full((int(pad_height), w, c), 255, dtype=np.uint8)
    image = np.concatenate([pad, image], 0)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.multiline_text(
        (w / 2, pad_height / 2),
        title,
        font=font,
        fill="black",
        anchor="ms", # m means middle horizontal s means middle vertical
        align="center"
    )

    return np.array(pil_image)

def _maybe_progress(x, enable):
    if enable and find_spec("tqdm") is not None:
        from tqdm import tqdm
        return tqdm(x)
    return x

def _repeat_to_largest(images: dict[str, np.ndarray]):
    """for each elemnt of x if both height and width are 2 or more times smaller than largest element repeat them

    x must be hwc"""
    max_h, max_w = np.max([i.shape for i in images.values()], axis = 0)[:-1]
    for k,img in images.copy().items():
        h,w = img.shape[:-1]
        ratio = min(max_h/h, max_w/w)
        if ratio >= 2:
            images[k] = np.repeat(np.repeat(img, ratio, 0), ratio, 1)
    return images

def _make_collage(images: dict[str, np.ndarray], titles:bool, w_cols):
    """make a collage from images"""
    if len(images) == 1: return next(iter(images.values())), 1

    images = _repeat_to_largest(images)
    if titles: images_titles = [_add_title(v, k) for k,v in images.items()]
    else: images_titles = images.values()

    max_shape = np.max([i.shape for i in images_titles], axis = 0)
    max_shape[:-1] += 2 # add 2 pixel to spatial dims
    stacked = np.stack([pad_to_shape(i, max_shape, mode = 'constant', value=128) for i in images_titles])
    # it is now (image, H, W, 3)

    # compose them
    ncols = len(stacked) ** (w_cols * (max_shape[0]/max_shape[1]))
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
    return stacked, ncols


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

def _rescale(x, scale):
    if scale > 1:
        return np.repeat(np.repeat(x, int(scale), 0), int(scale), 1)

    if scale < 1:
        skip = round(1/scale)
        return x[::skip,::skip]

    return x

@torch.no_grad
def _render(self: "Benchmark", file: os.PathLike | str, fps: int = 60, scale: int | float = 1, progress=True, w_cols=0.65):
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
        if self._performance_mode:
            raise RuntimeError(f'Images were not created for {self.__class__.__name__} because benchmark mode is enabled')
        raise NotImplementedError(f'Solution plotting is not implemented for {self.__class__.__name__}')

    with OpenCVRenderer(file, fps = fps, scale=1) as renderer:
        lowest_loss = float('inf')

        for step, loss in enumerate(_maybe_progress(list(self.logger['train loss'].values()), enable=progress)):
            # add current and best image
            images: dict[str, np.ndarray | torch.Tensor] = {}

            # add reference image
            for k, image in self._reference_images.items():
                images[k] = image

            # check if new params are better
            if loss <= lowest_loss:
                lowest_loss = loss

                # set to new best images
                for key in lowest_images:
                    if key in logger_images:
                        lowest_images[key] = logger_images[key][step]

            # add logger images
            for key, value in logger_images.items():
                images[key] = value[step]

            # add best images
            for key, image in lowest_images.items():
                images[f"{key} - best"] = image

            # make a collage
            collage, ncols = _make_collage({k: _rescale(make_hw3(tonumpy(v)), scale) for k,v in images.items()}, titles=self._show_titles_on_video, w_cols=w_cols)

            title = f"train loss: {str(format_number(loss,  5)).ljust(7, '0')[:7]}"
            if "test loss" in self.logger:
                test_loss = self.logger.closest("test loss", step)
                title = f"{title}; test loss: {str(format_number(test_loss, 5)).ljust(7, '0')[:7]}"

            renderer.write(_add_title(collage, title, size_per_px=0.04/ncols, wrap=False))

        path = os.path.abspath(renderer.outfile)
        if progress and str(file).lower().endswith(".gif"):
            print(GIF_POST_PBAR_MESSAGE)

    return path