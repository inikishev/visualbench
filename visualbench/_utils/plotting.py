import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Unpack
from functools import partial
import numpy as np
import torch
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig
from myai.plt_tools._types import _K_Line2D
from myai.python_tools import Progress
from myai.torch_tools import pad_to_shape
from myai.transforms import force_hw3, force_hwc, normalize, znormalize, totensor, tonumpy
from myai.video import OpenCVRenderer, render_frames
import tqdm

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


def _plot_trajectory(bench: "Benchmark", fig = None, norm: str | None = 'symlog', cmap ='coolwarm', show = True):
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
        cmap = cmap,
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

def _evaluate_projected(x, bench: "Benchmark", basis):
    x_full = x @ basis
    torch.nn.utils.vector_to_parameters(x_full, bench.parameters())
    return bench()

def _tqdm(x,enable=True):
    if enable: return tqdm.tqdm(x)
    return x

@torch.no_grad
def grid_search(func, bounds, num, device=None):
    ndim = len(bounds)
    candidates = torch.stack(
        torch.meshgrid(
            *[torch.linspace(*b, steps=num, device=device) for b in bounds],
            indexing='ij'
        )
    ).view(ndim, -1).moveaxis(0, 1) # npoints, ndim

    losses = torch.zeros_like(candidates[:,0]) # npoints
    for i, vec in enumerate(_tqdm(candidates)):
        losses[i] = func(vec)

    return list(candidates.detach().cpu().unbind(0)), losses.detach().cpu().tolist()

def _map_to_bounds(x, bounds):
    return x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
    #return x * (bounds[1] - bounds[0]) + bounds[0]

@torch.no_grad
def accelerated_random_search(func, bounds, fevals, device=None, x0=None, decay=0.5, restart=10):
    bounds = totensor(bounds, device=device)
    ndim = len(bounds)
    if x0 is None: x0 = (bounds[:,0] + 0.5 * (bounds[:,1] - bounds[:,0]))
    #if x0 is None: x0 = (bounds[0] + 0.5 * (bounds[1] - bounds[0]))
    else: x0 = totensor(x0, device=device)

    best_point = x0
    lowest_loss = func(best_point)
    sigma = 1
    steps = 0
    generator = torch.Generator(device).manual_seed(0)

    candidates = []
    losses = []

    for i in _tqdm(list(range(fevals))):
        candidate = best_point + _map_to_bounds(
            torch.empty(ndim, device=device, dtype=x0.dtype).uniform_(0, 1, generator=generator) * sigma,
            bounds
        ).to(x0)
        loss = func(candidate)
        candidates.append(candidate.detach().cpu())
        losses.append(loss.detach().cpu().item())
        if loss < lowest_loss:
            best_point = candidate
            lowest_loss = loss
            sigma = 1
            steps = 0
        else:
            sigma *= decay
            steps += 1

        if steps == restart:
            sigma = 1
            steps = 0

    return candidates, losses

def qr_basis(v1, v2) -> tuple[torch.Tensor, torch.Tensor]:
    A = torch.stack([v1, v2], dim=1)
    Q, _ = torch.linalg.qr(A) # pylint:disable=not-callable

    v1 = Q[:, 0]
    v2 = Q[:, 1]

    return v1, v2

def _make_bounds_square(xlim, ylim):
    xsize = np.abs(xlim[1] - xlim[0])
    ysize = np.abs(ylim[1] - ylim[0])
    diff = np.abs(xsize - ysize)
    if xsize > ysize:
        ylim = [ylim[0] - diff/2, ylim[1] + diff/2]
    else:
        xlim = [xlim[0] - diff/2, xlim[1] + diff/2]

    return xlim, ylim

def _sample_2d_landscape(projected_fn, bounds1, bounds2, gs_num, ars_evals, orig_params, bench: "Benchmark"):
    candidates, losses = grid_search(projected_fn, bounds=[bounds1, bounds2], num=gs_num)

    ars_x0 = candidates[np.argmin(losses).item()].to(orig_params)
    ars_candidates, ars_losses = accelerated_random_search(projected_fn, bounds=[bounds1, bounds2], fevals=ars_evals, x0=ars_x0)
    candidates.extend(ars_candidates) # n, 2
    losses.extend(ars_losses) # n

    # sample maximums
    ars_x0 = candidates[np.argmax(losses).item()].to(orig_params)
    ars_candidates, ars_losses = accelerated_random_search(lambda x: -projected_fn(x), bounds=[bounds1, bounds2], fevals=ars_evals//2, x0=ars_x0)
    candidates.extend(ars_candidates) # n, 2
    losses.extend(-np.asarray(ars_losses)) # n

    candidates = np.asarray(candidates)
    x, y = candidates[:,0], candidates[:, 1]

    # set bounds to area given by removing candidates above maximum loss
    max_loss = bench.logger.max('train loss')
    filtered_candidates = candidates[np.asarray(losses) < max_loss]
    filtered_x, filtered_y = filtered_candidates[:,0], filtered_candidates[:, 1]

    if filtered_x.size > 0: xlim = [filtered_x.min(), filtered_x.max()]
    else: xlim = bounds1
    if filtered_y.size > 0: ylim = [filtered_y.min(), filtered_y.max()]
    else: ylim = bounds2

    # resample if new xlim and ylim are significantly different
    if np.abs(bounds1[1] - bounds1[0]) / np.abs(xlim[1] - xlim[0]) > 1.5 or np.abs(bounds2[1] - bounds2[0]) / np.abs(ylim[1] - ylim[0]) > 1.5:
        x2,y2,losses2,xlim,ylim = _sample_2d_landscape(projected_fn=projected_fn,bounds1=xlim,bounds2=ylim, gs_num=int(gs_num/1.5), ars_evals=int(ars_evals/1.5), orig_params=orig_params, bench=bench)
        x = np.concatenate([x, x2])
        y = np.concatenate([y, y2])
        losses.extend(losses2)

    # keep xlim and ylim square
    xlim, ylim = _make_bounds_square(xlim, ylim)

    return x, y, losses, xlim, ylim

def _plot_landscape_random(
    bench: "Benchmark",
    gs_num=30,
    ars_evals=1000,
    plot_trajectory=True,
    norm: str | None = "symlog",
    mode: Literal["random", "edge"] | None = None,
    middle_mode: Literal["middle", "mean", "min"] = "mean",
    expand: float = 1,
    fig=None,
    show=True,
):
    if fig is None: fig = Fig()

    orig_params = torch.nn.utils.parameters_to_vector(bench.parameters())

    if mode is None:
        if 'params' in bench.logger: mode = 'edge'
        else: mode = 'random'

    if mode == 'random':
        proj1, proj2 = bench._proj1, bench._proj2
        if proj1 is None or proj2 is None:
            raise RuntimeError('log_projections must be true to plot landscape with random projection')

        basis = torch.stack([proj1, proj2], 0).float()
        projected = np.stack([bench.logger.numpy('proj1'), bench.logger.numpy('proj2')]).T

    elif mode == 'edge':
        stacked_params = torch.as_tensor(bench.logger.numpy('params')).to(orig_params)
        first_point = stacked_params[0]
        last_point = stacked_params[-1]
        if middle_mode == 'middle': middle_point = stacked_params[len(stacked_params)//2]
        elif middle_mode == 'min': middle_point = stacked_params[bench.logger.argmin('train loss')]
        elif middle_mode == 'mean': middle_point = stacked_params.mean(0)
        else: raise ValueError(middle_mode)

        if (first_point - middle_point == last_point - middle_point).all():
            raise RuntimeError('either there arent enough points or they form a line')

        basis = torch.stack(qr_basis(first_point - middle_point, last_point - middle_point), 0)

        stacked_params = bench.logger.numpy('params')
        projected = stacked_params @ basis.T.detach().cpu().numpy()

    else:
        raise ValueError(mode)


    min = projected.min(0); max = projected.max(0)

    bounds1 = [min[0], max[0]]
    bounds2 = [min[1], max[1]]

    # expand bounds
    size1 = bounds1[1] - bounds1[0]
    bounds1 = bounds1[0] - size1 * expand, bounds1[1] + size1 * expand
    size2 = bounds2[1] - bounds2[0]
    bounds2 = bounds2[0] - size2 * expand, bounds2[1] + size2 * expand
    bounds1, bounds2 = _make_bounds_square(bounds1, bounds2)

    projected_fn = partial(_evaluate_projected, bench=bench, basis=basis)
    x,y,losses,xlim,ylim = _sample_2d_landscape(projected_fn=projected_fn,bounds1=bounds1,bounds2=bounds2, gs_num=gs_num, ars_evals=ars_evals, orig_params=orig_params, bench=bench)

    fig.xlim(*xlim).ylim(*ylim).pcolormesh(x, y, losses, cmap='coolwarm', norm=norm, contour=True, contour_levels=32, contour_alpha=0.3,xlim=xlim,ylim=ylim).colorbar()
    if plot_trajectory:

        fig.scatter(
            x = projected[:,0],
            y = projected[:,1],
            alpha=0.4,
            s=4,
            c=bench.logger.numpy("train loss"),
            cmap = 'inferno',
            norm = norm,
        ).point(*projected[0], c='red').point(*projected[-1], c = 'green').tick_params(labelsize=7)

    torch.nn.utils.vector_to_parameters(orig_params, bench.parameters())

    if show: fig.show()
    return fig

def _evaluate_sklearn_projected(x, bench: "Benchmark", projector):
    x_full = torch.from_numpy(projector.inverse_transform(tonumpy(x.unsqueeze(0)))).to(x)[0]
    torch.nn.utils.vector_to_parameters(x_full, bench.parameters())
    return bench()

def _plot_landscape_sklearn(bench: "Benchmark", gs_num = 30, ars_evals = 1000, plot_trajectory=True, norm:str|None='symlog', projector: Literal['pca'] | Any = 'pca',expand:float=1, use_diff=False,fig=None, show=True):
    if fig is None: fig = Fig()

    orig_params = torch.nn.utils.parameters_to_vector(bench.parameters())

    if projector == 'pca':
        from sklearn.decomposition import PCA
        projector = PCA(n_components=2)

    params = bench.logger.numpy('params') # n, ndim

    if use_diff: projector.fit(np.gradient(params, axis=0), bench.logger.numpy('train loss'))
    else: projector.fit(params, bench.logger.numpy('train loss'))

    projected = projector.transform(params)
    min = projected.min(0); max = projected.max(0)
    bounds1 = [min[0], max[0]]
    bounds2 = [min[1], max[1]]

    # expand bounds
    size1 = bounds1[1] - bounds1[0]
    bounds1 = bounds1[0] - size1 * expand, bounds1[1] + size1 * expand
    size2 = bounds2[1] - bounds2[0]
    bounds2 = bounds2[0] - size2 * expand, bounds2[1] + size2 * expand
    bounds1, bounds2 = _make_bounds_square(bounds1, bounds2)

    projected_fn = partial(_evaluate_sklearn_projected, bench=bench, projector=projector)
    x,y,losses,xlim,ylim = _sample_2d_landscape(projected_fn=projected_fn,bounds1=bounds1,bounds2=bounds2, gs_num=gs_num, ars_evals=ars_evals, orig_params=orig_params, bench=bench)

    fig.xlim(*xlim).ylim(*ylim).pcolormesh(x, y, losses, cmap='coolwarm', norm=norm, contour=True, contour_levels=32, contour_alpha=0.3,xlim=xlim,ylim=ylim).colorbar()
    if plot_trajectory:
        fig.scatter(
            x = projected[:,0],
            y = projected[:,1],
            alpha=0.4,
            s=4,
            c=bench.logger.numpy("train loss"),
            cmap = 'inferno',
            norm = norm,
        ).point(*projected[0], c='red').point(*projected[-1], c = 'green').tick_params(labelsize=7)

    torch.nn.utils.vector_to_parameters(orig_params, bench.parameters())

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

