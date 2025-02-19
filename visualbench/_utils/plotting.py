import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
import torch
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig
from myai.plt_tools._types import _K_Line2D
from myai.python_tools import Progress
from myai.torch_tools import pad_to_shape
from myai.transforms import force_hw3, force_hwc, normalize, znormalize
from myai.video import OpenCVRenderer, render_frames

from .utils import _check_image

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _plot_loss(bench: "Benchmark", ylim: Literal['auto'] | Sequence[float] | None = 'auto', yscale = None, x = 'num passes', y = 'loss', fig=None, show=True, **kw: Unpack[_K_Line2D]):
    if fig is None: fig = Fig()

    possible_keys = [f'train {y}', f'test {y}', f'{y}']
    keys_in_logger = [i for i in possible_keys if i in bench.logger]
    if len(keys_in_logger) == 0:
        warnings.warn(f"No key `{y}` found in benchmark {bench.__class__.__name__}")
        return fig

    # automatic ylim from first and min values
    if (ylim == 'auto') and (yscale is None) and y in ('loss', 'train loss', 'test loss'):
        ymin = min([bench.logger.min(i) for i in possible_keys if i in bench.logger])
        ymax = max([bench.logger.first(i) for i in possible_keys if i in bench.logger])
        # expand range a little
        d = ymax - ymin
        ymin -= d*0.05; ymax += d*0.05
        ylim = (ymin, ymax)
    if ylim == 'auto': ylim = None

    if yscale is not None: fig.yscale(yscale)

    # plot train and test losses
    for key in possible_keys:
        if key in bench.logger:
            bench.logger.linechart(x = x, y = key, method = 'shared', fig = fig, axlabels=False, ylim = ylim)

    fig.axlabels('num forward/backward passes' if x == 'num passes' else x, y)
    if show: fig.show()
    return fig


def _plot_trajectory(bench: "Benchmark", fig = None, norm: str | None = 'symlog', show = True):
    """plot parameter trajectory, optionally also plot a loss landscape slice defined by first, middle and last points."""
    if fig is None: fig = Fig()

    first_point = (bench.logger.first('proj1'), bench.logger.first('proj2'))
    last_point = (bench.logger.last('proj1'), bench.logger.last('proj2'))

    fig.scatter(
        x = bench.logger.numpy("proj1"),
        y = bench.logger.numpy("proj2"),
        alpha=0.4,
        s=4,
        c=bench.logger.numpy("train loss"),
        cmap = 'coolwarm',
        norm = norm,
    ).point(*first_point, c='red').point(*last_point, c = 'green').tick_params(labelsize=7)

    if show: fig.show()
    return fig

def _plot_images(bench: "Benchmark", fig=None, show=True):
    if fig is None: fig = Fig()
    # plot images from logger
    first = True
    for k in bench.logger.keys():
        if k.startswith(('image', 'train image', 'test image')):
            # add new subplot except 1st
            if first: first = False
            else: fig.add()
            if k in bench.display_best_keys:
                fig.imshow(bench.logger.last(k), norm=None).axtitle(f"{k} - last")
                fig.add().imshow(bench.logger.get_closest(k, bench.logger.argmin('train loss')), norm=None).axtitle(f"{k} - best")
            else:
                fig.imshow(bench.logger.last(k), norm=None).axtitle(k) # no norm because all images are uint8

    # plot reference images
    for k,v in bench.reference_images.items():
        if first: first = False
        else: fig.add()
        fig.imshow(v, norm = None).axtitle(k)

    if show: fig.show()
    return fig


def _repeat_to_largest(images: list[np.ndarray | torch.Tensor]):
    """for each elemnt of x if both height and width are 2 or more times smaller than largest element repeat them

    x must be hwc"""
    max_h, max_w = np.max([i.shape for i in images], axis = 0)[:-1]
    for i,img in enumerate(images.copy()):
        h,w = img.shape[:-1]
        ratio = min(max_h/h, max_w/w)
        if ratio >= 2:
            if isinstance(img, np.ndarray): images[i] = np.repeat(np.repeat(img, ratio, 0), ratio, 1)
            else: images[i] = img.repeat_interleave(ratio, 0).repeat_interleave(ratio, 1)
    return images


@torch.no_grad
def _render_video(bench: "Benchmark", file: str, fps: int = 60, scale: int | float = 1, progress=True,):
    """renders a video of how current and best solution evolves on each step, if applicable to this benchmark."""

    logger_images = {}
    best_images = {}
    length = bench.logger.num_steps()

    # initialize all keys
    for key, value in bench.logger.items():
        if key.startswith(('image', 'train image', 'test image')):
            logger_images[key] = list(value.values())
            if key in bench.display_best_keys: best_images[key] = logger_images[key][0]

            # not saving an image on each step isn't supported (yet)
            assert len(logger_images[key]) == length, f'{len(logger_images[key]) = }, {length = }'

    for key, value in bench.reference_images.items():
        _check_image(value, f'reference_images[{key}]')

    with OpenCVRenderer(file, fps = fps, scale=scale) as renderer:
        lowest_loss = float('inf')

        for step, loss in enumerate(Progress(bench.logger['train loss'].values(), sec=0.1, enable=progress)):
            # add current and best image
            images: list[np.ndarray | torch.Tensor] = []

            # check if new params are better
            if loss <= lowest_loss:
                lowest_loss = loss

                # set to new best images
                for key in bench.display_best_keys:
                    if key in logger_images:
                        best_images[key] = logger_images[key][step]

            # add logger images
            for key, value in logger_images.items(): images.append(value[step])

            # add best images
            for image in best_images.values(): images.append(image)

            # check before adding reference images because they are static
            if len(images) == 0:
                raise NotImplementedError(f'Solution plotting is not implemented for {bench.__class__.__name__}')

            # add reference image
            for image in bench.reference_images.values(): images.append(image)

            # make a collage
            images = _repeat_to_largest([force_hw3(i) for i in images])
            max_shape = np.max([i.shape for i in images], axis = 0)
            max_shape[:-1] += 2 # add 2 pixel to spatial dims
            stacked = np.stack([pad_to_shape(i, max_shape, mode = 'constant', value = 128, crop = True) for i in images])
            # it is now (image, H, W, 3)
            if len(stacked) == 1: renderer.add_frame(stacked[0])
            # compose them
            else:
                ncols = len(stacked) ** 0.55
                nrows = round(len(stacked) / ncols)
                ncols = round(ncols)
                nrows = max(nrows, 1)
                ncols = max(ncols, 1)
                r = True
                while nrows * ncols < len(stacked):
                    if r: ncols += 1
                    else: ncols += 1
                    r = not r
                n_tiles = nrows * ncols
                if len(stacked) < n_tiles: stacked = np.concatenate([stacked, np.zeros_like(stacked[:n_tiles - len(stacked)])])
                stacked = stacked.reshape(nrows, ncols, *max_shape)
                stacked = np.concatenate(np.concatenate(stacked, 1), 1)
                renderer.add_frame(stacked)

