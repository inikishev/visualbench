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

if TYPE_CHECKING:
    from ..benchmark import Benchmark

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


def _make_float_hwc_square_matrix(x: Any) -> torch.Tensor:
    x = _make_float_hwc_tensor(x)
    if x.shape[1] == x.shape[0]: return x
    if x.shape[0] > x.shape[1]:
        print(f"got matrix of shape {x.shape} where it needs to be square so trimming down to {x.shape[1], x.shape[1], x.shape[2]}")
        return x[:x.shape[1]]
    if x.shape[0] < x.shape[1]:
        print(f"got matrix of shape {x.shape} where it needs to be square so trimming down to {x.shape[0], x.shape[0], x.shape[2]}")
        return x[:, :x.shape[0]]
    raise RuntimeError(f"wtf {x.shape}")

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


def sinkhorn(logits, num_iters=10):
    """Applies Sinkhorn normalization to logits to generate a doubly stochastic matrix."""
    log_alpha = logits
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


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
        text = f'{text} train loss = {_round_significant(bench.logger.min("train loss"), 3)}, train test = {_round_significant(bench.logger.min("test loss"), 3)}'
    else:
        text = f'{text} loss = {_round_significant(bench.logger.min("train loss"), 3)}'
    print(f'{text}                                      ')

