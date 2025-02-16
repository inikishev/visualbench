from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
import torch
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig
from myai.plt_tools._types import _K_Line2D
from myai.torch_tools import pad_to_shape
from myai.transforms import force_hw3, force_hwc, normalize, znormalize
from myai.video import OpenCVRenderer, render_frames
from myai.python_tools import Progress

if TYPE_CHECKING:
    from .benchmark import Benchmark

CUDA_IF_AVAILABLE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _make_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, str): return znormalize(imreadtensor(x).float()).contiguous()
    if isinstance(x, torch.Tensor): return x.clone().float().contiguous()
    if isinstance(x, np.ndarray): return torch.from_numpy(x).float().contiguous()
    return torch.from_numpy(np.asarray(x)).float().contiguous()

def __tensor_from_ints(x):
    """if x is int, generates (1, x, x), if tuple of its, generates"""
    return None

def _make_float_hw3_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, int):
        return torch.randn((x, x, 3), generator=torch.Generator().manual_seed(0))
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) for i in x):
        return torch.randn(tuple(list(x) + [3]), generator=torch.Generator().manual_seed(0))
    if isinstance(x, tuple) and len(x) == 3 and all(isinstance(i, int) for i in x):
        if x[0] < x[2]: x = (x[1], x[2], 3)
        else: x = (x[0], x[1], 3)
        return torch.randn(x, generator=torch.Generator().manual_seed(0))
    return force_hw3(_make_float_tensor(x))

def test_make_float_hw3_tensor():
    assert _make_float_hw3_tensor(4).shape == (4,4,3), _make_float_hw3_tensor(4).shape
    assert _make_float_hw3_tensor((5, 6)).shape == (5,6,3), _make_float_hw3_tensor((5,6)).shape
    assert _make_float_hw3_tensor((5, 6, 3)).shape == (5,6,3), _make_float_hw3_tensor((5, 6, 3)).shape
    assert _make_float_hw3_tensor((3, 5, 6)).shape == (5,6,3), _make_float_hw3_tensor((3, 5, 6)).shape

def _make_float_hwc_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, int):
        return torch.randn((x, x, 1), generator=torch.Generator().manual_seed(0))
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) for i in x):
        return torch.randn(tuple(list(x) + [1]), generator=torch.Generator().manual_seed(0))
    if isinstance(x, tuple) and len(x) == 3 and all(isinstance(i, int) for i in x):
        if x[0] < x[2]: x = (x[1], x[2], x[0])
        return torch.randn(x, generator=torch.Generator().manual_seed(0))
    return force_hwc(_make_float_tensor(x))

def test_make_float_hwc_tensor():
    assert _make_float_hwc_tensor(4).shape == (4,4,1), _make_float_hw3_tensor(4).shape
    assert _make_float_hwc_tensor((5, 6)).shape == (5,6,1), _make_float_hw3_tensor((5,6)).shape
    assert _make_float_hwc_tensor((5, 6, 3)).shape == (5,6,3), _make_float_hw3_tensor((5, 6, 3)).shape
    assert _make_float_hwc_tensor((2, 5, 6)).shape == (5,6,2), _make_float_hw3_tensor((3, 5, 6)).shape

def _ensure_float(x) -> float:
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return x.item() # type:ignore
    return float(x)

def _maybe_detach_clone(x: torch.Tensor | np.ndarray):
    if isinstance(x, torch.Tensor): return x.detach().clone()
    if isinstance(x, np.ndarray): return x.copy()
    return x

@torch.no_grad
def _normalize_to_uint8(x:torch.Tensor | np.ndarray | torch.nn.Buffer) -> np.ndarray:
    """normalizes to 0-255 range and converts to numpy uint8 array"""
    if isinstance(x, np.ndarray): return normalize(np.nan_to_num(x), 0, 255).astype(np.uint8)
    return normalize(x.nan_to_num().detach(), 0, 255).cpu().numpy().astype(np.uint8)

def _round_significant(x: float, nsignificant: int):
    if x > 1: return round(x, nsignificant)
    if x == 0: return x

    v = 1
    for n in range(100):
        v /= 10
        if abs(x) > v: return round(x, nsignificant + n)

    return x # x is nan
    #raise RuntimeError(f"wtf {x = }")

def _check_image(image, name=None):
    """checks image also returns squeezed"""
    if isinstance(image, np.ndarray): lib = np
    elif isinstance(image, torch.Tensor): lib = torch
    else: raise TypeError(f"Invalid image {name}, type must be np.ndarray or torch.Tensor, got {type(image)}")
    if image.dtype != lib.uint8: raise TypeError(f"Invalid image {name}, dtype must be uint8 but got {image.dtype}")
    if image.ndim > 3: image = lib.squeeze(image) # type:ignore
    if image.ndim not in (2, 3):
        raise ValueError(f"Invalid image {name}, must be 2D or 3D but got shape {image.shape}")
    return image

def _print_progress(bench: "Benchmark"):
    """print progress every second sets _last_print_time"""
    # if one second passed from last print
    t = bench._cur_time
    assert t is not None

    if t - bench._last_print_time > 1:
        text = f'f{bench._num_forwards}'
        if bench._max_forwards is not None: text = f'{text}/{bench._max_forwards}'

        if bench._num_passes != 0:
            text = f'{text} p{bench._num_passes}'
            if bench._max_passes is not None: text = f'{text}/{bench._max_passes}'

        if bench._num_batches != 0:
            text = f'{text} b{bench._num_batches}'
            if bench._max_batches is not None: text = f'{text}/{bench._max_batches}'

        if bench._num_epochs != 0:
            text = f'{text} e{bench._num_epochs}'
            if bench._max_epochs is not None: text = f'{text}/{bench._max_epochs}'

        if bench._last_train_loss is not None: text = f'{text}; train loss = {_round_significant(bench._last_train_loss, 3)}'
        if bench._last_test_loss is not None: text = f"{text}; test loss = {_round_significant(bench._last_test_loss, 3)}"

        print(text, end = '                          \r')
        bench._last_print_time = t


def _aggregate_test_metrics_(bench: "Benchmark"):
    """Logs the mean of each test metrics and resets test metrics to an empty dict"""
    for metric, values in bench._test_metrics.items():
        if len(values) == 1: bench.log(f'test {metric}', values[0], log_test=False)
        else: bench.log(f'test {metric}', np.nanmean(values, axis = -1), log_test=False)
    bench._test_metrics = {}

def _check_stop_condition(bench: "Benchmark") -> str | None:
    """runs at the beginning of every train forward"""
    if (bench._max_forwards is not None) and (bench._num_forwards >= bench._max_forwards): return 'max steps reached'
    if (bench._max_passes is not None) and (bench._num_passes >= bench._max_passes): return 'max passes reached'
    if (bench._max_epochs is not None) and (bench._num_epochs >= bench._max_epochs): return 'max epochs reached'
    if (bench._max_batches is not None) and (bench._num_batches >= bench._max_batches): return 'max batches reached'
    if (bench._max_seconds is not None) and bench._time_passed >= bench._max_seconds: return "max time reached"
    return None

def _ensure_stop_condition_exists_(bench: "Benchmark") -> None:
    """runs before run"""
    if all(i is None for i in [bench._max_forwards, bench._max_passes, bench._max_batches, bench._max_epochs, bench._max_seconds]):
        raise RuntimeError("The benchmark will run forever because there is no stop condition")

def _check_test_epoch_condition(bench: "Benchmark") -> bool:
    """runs after every train closure evaluation"""
    if bench._dltest is None: return False
    if (bench._test_every_forwards is not None) and (bench._num_forwards % bench._test_every_forwards == 0): return True
    if (bench._test_every_batches is not None) and (bench._num_batches % bench._test_every_batches == 0): return True
    if (bench._test_every_epochs is not None) and (bench._num_epochs % bench._test_every_epochs == 0): return True
    assert bench._cur_time is not None
    if (bench._test_every_seconds is not None) and (bench._cur_time - bench._last_test_time >= bench._test_every_seconds): return True

    return False


@torch.no_grad
def _log_params_and_projections_(bench: "Benchmark") -> None:
    """conditionally logs parameters and projections if that is enabled, all in one function to reuse parameter_to_vector

    this runs before 1st step, and each train forward pass"""
    param_vec = None

    # --------------------------- log parameter vectors -------------------------- #
    if bench._log_params:
        param_vec = torch.nn.utils.parameters_to_vector(bench.parameters()).detach()
        bench.log('params', param_vec.detach().cpu(), log_test=False)

    # ------------------------------ log projections ----------------------------- #
    if bench._log_projections:
        if param_vec is None: param_vec = torch.nn.utils.parameters_to_vector(bench.parameters()).detach()

        # create projections if they are none, one is a bernoulli vector and the other one is the inverse
        if (bench._proj1 is None) or (bench._proj2 is None):
            projections = torch.ones((2, param_vec.numel()), dtype = torch.bool, device = param_vec.device)
            projections[0] = torch.bernoulli(
                projections[0].float(), p = 0.5, generator = bench.rng.torch(param_vec.device),
            ).to(dtype = torch.bool, device = param_vec.device)

            projections[1] = ~projections[0]
            bench._proj1, bench._proj2 = projections

        # log projections
        bench.log('proj1', (param_vec * bench._proj1).mean(), log_test=False)
        bench.log('proj2', (param_vec * bench._proj2).mean(), log_test=False)


def _print_final_report(bench: "Benchmark"):
    text = f'finished in {bench._time_passed:.1f}s., reached'
    if 'test loss' in bench.logger:
        text = f'{text} train loss = {_round_significant(bench.logger.min('train loss'), 3)}, train test = {_round_significant(bench.logger.min('test loss'), 3)}'
    else:
        text = f'{text} loss = {_round_significant(bench.logger.min('train loss'), 3)}'
    print(f'{text}                                      ')


def _plot_loss(bench: "Benchmark", ylim: Literal['auto'] | Sequence[float] | None = 'auto', yscale = None, x = 'num passes', y = 'loss', fig=None, show=True, **kw: Unpack[_K_Line2D]):
    if fig is None: fig = Fig()

    possible_keys = [f'train {y}', f'test {y}', f'{y}']

    # automatic ylim from first and min values
    if (ylim == 'auto') and (yscale is None):
        ymin = min([bench.logger.min(i) for i in possible_keys if i in bench.logger])
        ymax = max([bench.logger.first(i) for i in possible_keys if i in bench.logger])
        # expand range a little
        d = ymax - ymin
        ymin -= d*0.05; ymax += d*0.05
        ylim = (ymin, ymax)

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

