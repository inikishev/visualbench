import random
import math
import warnings
import os
import textwrap
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, overload

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import msgspec
import numpy as np
from matplotlib import patches, ticker
from matplotlib.axes import Axes
from matplotlib.scale import SymmetricalLogScale
from scipy.ndimage import gaussian_filter1d

from ..utils.plt_tools import _auto_loss_yrange, legend_, make_axes
from ..utils.python_tools import format_number, to_valid_fname

if TYPE_CHECKING:
    from ..logger import Logger
    from ..runs.run import Run, Sweep, Task

# region reference opts
REFERENCE_OPTS = (
    "GD",
    "torch.NAG(0.95)",
    "torch.Adagrad",
    "torch.RMSprop",
    "torch.Adam",
    "torch.AdamW",
    "torch.LBFGS(strong_wolfe)",
    # "tz.BFGS-Backtracking",
    # "tz.Newton",
    "tz.SOAP",
)
# endregion

# region yscales
_YSCALES: dict[str, Any] = {
    # ------------------------------------ new ----------------------------------- #
    "S - Inverse-16 L1": "log",
    "S - Inverse-16 MSE": "log",
    "S - MoorePenrose-16 L1": "log",
    "S - MatrixLogarithm-16 L1": "log",
    "SS - StochasticRLstsq-10 MSE": "log",
    "S - Drazin-fielder16 L1": "log",
    "S - ChebushevRosenbrock-8": "log",
    "S - Inverse-fielder16 MSE": "log",
    "S - Rosenbrock 384": "log",
    "S - Rotated quadratic 384": dict(value='symlog', linthresh=1e-12),
    "S - Nonsmooth Chebyshev-Rosenbrock 384": "log",
    "S - Rastrigin 384": "log",
    "MLS - MovieLens BS-32 - Matrix Factorization": "log",
    "Visual - NeuralDrawer - ReLU+bn": "log",
    "Visual - NeuralDrawer - ELU": "log",
    "Visual - NeuralDrawer - Sine": "log",
    "Visual - NeuralDrawer - Wide ReLU": "log",

    # ------------------------------------ old ----------------------------------- #
    # ML
    "ML - Olivetti Faces FB - Logistic Regression": dict(value='symlog', linthresh=1e-12),
    "ML - Friedman 1 - Linear Regression - L1": "log",
    "ML - MNIST-1D FB - NeuralODE": "log",
    "ML - Wave PDE - TinyFLS": "log",
    "ML - Wave PDE - FLS": "log",
    "ML - WDBC FB - ElasticNet": "log",
    "MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet": "log",

    # 2D
    "2D - booth": dict(value='symlog', linthresh=1e-8),
    "2D - ill": dict(value='symlog', linthresh=1e-6),
    "2D - star": "log",
    "2D - rosenbrock": dict(value='symlog', linthresh=1e-8),
    "2D - rosenbrock abs": "log",
    "2D - spiral": "log",
    "2D - illppc": "log",
    "2D - oscillating": dict(value='symlog', linthresh=1e-6),
    # "2D simultaneous - rosenbrock-10": "log",
    # "2D simultaneous - rosenbrock": "log",
    # "2D simultaneous - rosenbrock abs": "log",
    # "2D simultaneous - rosenbrock rastrigin": "log",
    # "2D simultaneous - oscillating": "log",

    # Losses
    "ML - Friedman 1 - Linear Regression - L-Infinity": "log",
    "ML - Friedman 1 - Linear Regression - L4": "log",
    "ML - Friedman 1 - Linear Regression - Median": "log",
    "ML - Friedman 1 - Linear Regression - Quartic": "log",

    # Synthetic
    "S - Ill conditioned quadratic": dict(value='symlog', linthresh=1e-12),
    "S - Rosenbrock-384": "log",
    "S - IllConditioned-384": "log",
    "S - LogSumExp": "log",
    "S - Least Squares": "log",
    "S - Inverse - L1": "log",
    "S - Inverse - MSE": "log",
    "S - Matrix idempotent": "log",
    "S - Tropical QR - L1": "log",
    "S - Tropical QR - MSE": "log",

    # synthetic stochastic
    "SS - Stochastic inverse - L1": "log",
    "SS - Stochastic inverse - MSE": "log",
    "SS - Stochastic matrix root - L1": "log",
    "SS - Stochastic matrix root - MSE": "log",
    "SS - Stochastic matrix recovery - L1": "log",
    "SS - Stochastic matrix recovery - MSE": "log",


    # visual
    "Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU": "log",
    "Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ReLU+bn": "log",
    "Visual - PartitionDrawer": "log",
    "Visual - Moons BS-16 - MLP(2-2-2-2-2-2-2-2-1)-ELU": "log",
    "Visual - Colorization": dict(value='symlog', linthresh=1e-6),
    "Visual - Colorization (2nd order)": dict(value='symlog', linthresh=1e-6),
    # "Visual - Colorization (1.3th power)": dict(value='symlog', linthresh=1e-6),
    "Visual - Graph layout optimization": "log",
    "Visual - Sine Approximator - Tanh 7-4": "log",

    # real
    "Real - Human heart dipole": "log",
    "Real - Propane combustion": "log",
    "Real - Style Transfer": "log",
    "Real - Muon coefficients": "log",
}
# endregion

# region train smoothing
_TRAIN_SMOOTHING: dict[str, float] = {
    "SS - Stochastic inverse - L1": 2,
    "SS - Stochastic inverse - MSE": 2,
    "SS - Stochastic matrix recovery - L1": 2,
    "SS - Stochastic matrix recovery - MSE": 2,
    "SS - Stochastic matrix idempotent": 2,
    "SS - Stochastic matrix idempotent (hard)": 2,
    "MLS - Covertype BS-1 - Logistic Regression": 8,
    "MLS - MNIST-1D BS-32 - TinyConvNet": 4,
    "MLS - SynthSeg BS-64 - ConvNet": 2,
    "MLS - MNIST-1D BS-64 - MLP(40-64-96-128-256-10)": 4,
    "MLS - MNIST-1D BS-128 - RNN(2x40)": 4,
    "MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet": 4,
    "SS - StochasticRLstsq-10 MSE": 4,
}
# endregion


_COLORS_MAIN = ("red", "green", "blue")
_COLORS_REFERENCES = ("deepskyblue", "orange", "springgreen", "coral", "lawngreen", "aquamarine", "plum", "pink", "peru")
_COLORS_BEST = ("black", "dimgray", "maroon", "midnightblue", "darkgreen", "rebeccapurple", "darkmagenta", "saddlebrown", "darkslategray")
Scale = None | str | dict[str, Any] | Callable[[Axes], Any]

def _maybe_format_number(x):
    if isinstance(x, (int,float)): return format_number(x, 3)
    return x

def _make_label(run: "Run", best_value: float, hyperparams: str | Sequence[str] | None):
    name = run.run_name
    assert name is not None
    if hyperparams is None: return f"{name}: {format_number(best_value, 5)}"
    if isinstance(hyperparams, str): hyperparams = [hyperparams, ]

    for h in hyperparams:
        if h in run.hyperparams:
            name = f"{name} {h}={_maybe_format_number(run.hyperparams[h])}"

    return f"{name}: {format_number(best_value, 5)}"

def _load_steps_values(logger: "Logger", metric):
    values = logger.numpy(metric)
    step_idxs = np.array(logger.steps(metric))
    num_passes = logger.numpy('num passes').astype(np.uint64)
    steps = num_passes[step_idxs.clip(max=len(num_passes)-1)]
    return steps, values

def _set_scale_(ax: Axes, scale: Scale, which='y'):
    if scale is None: return ax

    if which == 'y':
        if isinstance(scale, str): ax.set_yscale(scale)
        elif isinstance(scale, dict): ax.set_yscale(**scale)
        elif callable(scale): scale(ax)
        else: raise ValueError(f"Invalid yscale {scale}")
        return ax

    if which == 'x':
        if isinstance(scale, str): ax.set_xscale(scale)
        elif isinstance(scale, dict): ax.set_xscale(**scale)
        elif callable(scale): scale(ax)
        else: raise ValueError(f"Invalid yscale {scale}")
        return ax

    raise ValueError(which)

def _is_log_scale(yscale: Scale):
    if yscale is None: return False
    if isinstance(yscale, str): return 'log' in yscale
    if isinstance(yscale, dict):
        yscale = yscale['value']
        return _is_log_scale(yscale)
    if callable(yscale): return False
    raise ValueError(f"Invalid yscale {yscale}")

def _xaxis_settings_(ax:Axes, yscale: Scale):
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if _is_log_scale(yscale):
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    else:
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(which='major', lw=0.5)
    ax.grid(which='minor', lw=0.5, alpha=0.15)

    return ax


def _wrap(s:str | None, maxlen=30) -> Any:
    if s is None: return None
    return '\n'.join(textwrap.wrap(s, maxlen))

# region plot_train_test_values
def plot_train_test_values(
    sweep: "Sweep",
    yscale: Scale = None,
    ax: Axes | None = None,
):
    if len(sweep) == 0: return ax
    if ax is None: ax = plt.gca()

    # --------------------------------- load runs -------------------------------- #
    best_train = sweep.best_runs('train loss', False, 1)[0]
    best_test = sweep.best_runs('test loss', False, 1)[0]

    btr_train_steps, btr_train_values = _load_steps_values(best_train.load_logger(), 'train loss')
    btr_test_steps, btr_test_values = _load_steps_values(best_train.load_logger(), 'test loss')

    bte_train_steps, bte_train_values = _load_steps_values(best_test.load_logger(), 'train loss')
    bte_test_steps, bte_test_values = _load_steps_values(best_test.load_logger(), 'test loss')

    # ---------------------------- determine y limits ---------------------------- #
    ylim = _auto_loss_yrange(btr_train_values, btr_test_values, bte_train_values, bte_test_values, yscale=yscale)
    if ylim is not None: ax.set_ylim(*ylim)

    _set_scale_(ax, yscale)

    # ---------------------------- plot ---------------------------- #
    if sweep.task_name is not None and sweep.task_name in _TRAIN_SMOOTHING:
        smoothing = _TRAIN_SMOOTHING[sweep.task_name]
        if smoothing != 0:
            bte_train_values = gaussian_filter1d(bte_train_values, smoothing, mode='nearest')
            btr_train_values = gaussian_filter1d(btr_train_values, smoothing, mode='nearest')

    ax.plot(bte_train_steps, bte_train_values, label=f"train (best test): {format_number(np.nanmin(bte_train_values), 5)}", c='darkgreen', lw=0.5, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_train_steps, btr_train_values, label=f"train (best train): {format_number(np.nanmin(btr_train_values), 5)}", c='darkred', lw=0.5, alpha=0.5)

    ax.plot(bte_test_steps, bte_test_values, label=f"test (best test): {format_number(np.nanmin(bte_test_values), 5)}", c='lime', lw=1.0, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_test_steps, btr_test_values, label=f"test (best train): {format_number(np.nanmin(btr_test_values), 5)}", c='red', lw=1.0, alpha=0.5)


    # ------------------------------- axes and grid ------------------------------ #
    ax.set_title(_wrap(f'{sweep.run_name} - {sweep.task_name}'), fontsize=9)
    ax.set_ylabel('loss')
    ax.set_xlabel('num forward/backward passes')
    legend_(ax)

    _xaxis_settings_(ax, yscale)
    return ax
# endregion

def _find_different(*d:dict):
    if len(d) == 0: return None
    if len(d) == 1: return _get_1st_key(d[0])
    d0 = d[0]
    d1 = d[1]
    for k, v0 in d0.items():
        v1 = d1[k]
        if v0 != v1: return k
    return _get_1st_key(d[0])

def _get_1st_key(d: dict):
    if len(d) == 0: return None
    return next(iter(d.keys()))


# this ensures same colors
class _RandomRGB:
    def __init__(self):
        self.random = random.Random(0)
    def __call__(self, max_sum=512):
        randrange = self.random.randrange
        rgb = (randrange(0,256), randrange(0,256), randrange(0,256))
        while sum(rgb) > max_sum:
            rgb = (randrange(0,256), randrange(0,256), randrange(0,256))
        return tuple(i/255 for i in rgb)


def _plot_metric(
    ax: Axes,
    runs: "Sequence[Run]",
    metric: str,
    maximize: bool,
    smoothing: float,
    colors: Sequence,
    **plot_kwargs,
):
    gen = _RandomRGB()
    while len(colors) < len(runs):
        # print(f"ADD {len(runs) - len(colors)} MORE COLORS TO {colors}!!!")
        colors = list(colors).copy()
        colors.append(gen())

    for r,c in zip(runs,colors):
        steps, values = _load_steps_values(r.load_logger(), metric)
        best = np.nanmax(values) if maximize else np.nanmin(values)

        if smoothing != 0: values = gaussian_filter1d(values, smoothing, mode='nearest')
        ax.plot(steps, values, label=_wrap(_make_label(r, best, _get_1st_key(r.hyperparams))), color=c, **plot_kwargs)

    return ax

# region plot_values
def plot_values(
    task: "Task",
    metric: str,
    maximize: bool,
    main: str | Sequence[str] | None,
    references: str | Sequence[str] | None,
    n_best: int,
    yscale = None,
    smoothing: float = 0,
    ax: Axes | None = None
):
    if ax is None: ax = plt.gca()

    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    # --------------------------------- load runs -------------------------------- #
    main_runs = [task[r].best_runs(metric, maximize, 1)[0] for r in main]
    best_runs = [r for r in task.best_sweep_runs(metric, maximize, n_best) if r not in main_runs]
    reference_runs = [task[r].best_runs(metric, maximize, 1)[0] for r in references if (r in task.keys())]
    reference_runs = [r for r in reference_runs if r not in main_runs+best_runs]

    # determine y-limit based on first value
    if not maximize:
        all_runs = main_runs + reference_runs + best_runs
        all_values = [r.load_logger().numpy(metric) for r in all_runs]
        ylim = _auto_loss_yrange(*all_values, yscale=yscale)
        if ylim is not None: ax.set_ylim(*ylim)

    _set_scale_(ax, yscale)

    # ----------------------------------- plot ----------------------------------- #
    _plot_metric(ax, reference_runs, metric, maximize, smoothing, _COLORS_REFERENCES, lw=0.5)

    # for best if no references/main are available, use better main colos
    colors_best = _COLORS_BEST
    if len(reference_runs) + len(main_runs) == 0:
        colors_best = _COLORS_MAIN + _COLORS_REFERENCES + _COLORS_BEST
    _plot_metric(ax, best_runs, metric, maximize, smoothing, colors_best, lw=0.5)

    _plot_metric(ax, main_runs, metric, maximize, smoothing, _COLORS_MAIN)

    name = task.task_name
    if len(main) == 1: name = f'{main[0]} - {name}'
    if name is not None: ax.set_title(_wrap(name), fontsize=9)
    ax.set_ylabel(metric)
    ax.set_xlabel('num forward/backward passes')
    legend_(ax)

    # ------------------------------- axes and grid ------------------------------ #
    _xaxis_settings_(ax, yscale)
    return ax
# endregion

def _plot_sweep(
    ax: Axes,
    sweeps: "list[Sweep]",
    metric: str,
    maximize: bool,
    colors: Sequence,
    lw,
    marker_size,
):
    gen = _RandomRGB()
    while len(colors) < len(sweeps):
        colors = list(colors).copy()
        colors.append(gen())

    for s,c in zip(sweeps,colors):
        key = 'max' if maximize else 'min'
        if len(s) == 1:
            v = s[0].stats[metric][key]
            if math.isfinite(v):
                ax.axhline(s[0].stats[metric][key], c=c, lw=lw, ls='--', label=_wrap(s.run_name))
        else:
            hyperparam = _find_different(*(r.hyperparams for r in s))
            if hyperparam is None: continue
            values = [(run.hyperparams[hyperparam], run.stats[metric][key]) for run in s]
            values.sort(key=lambda x: x[0])
            if any(math.isfinite(v) for k,v in values):
                ax.plot(*zip(*values), label=_wrap(s.run_name), c=c, lw=lw)
                ax.scatter(*zip(*values), color=c, s=marker_size, alpha=0.5,)

    return ax

def _sweep_xyaxes(ax: Axes, xscale, yscale):
    # ----------------------------------- xaxis ---------------------------------- #
    if isinstance(xscale, str) and 'log' in xscale:
        ax.xaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    else:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ----------------------------------- yaxis ---------------------------------- #
    if _is_log_scale(yscale):
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    else:
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(which='major', lw=0.5, alpha=0.3)
    ax.grid(which='minor', lw=0.5, alpha=0.15)
    return ax

# region plot_sweeps
def plot_sweeps(
    task: "Task",
    metric: str,
    maximize: bool,
    main: str | Sequence[str] | None,
    references: str | Sequence[str] | None,
    n_best: int,
    xscale: Any = 'log',
    yscale: Scale = None,
    ax: Axes | None = None
):
    if ax is None: ax = plt.gca()

    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    # ------------------ determine y-limit based on first value ------------------ #
    if not maximize:
        best_run = task.best_sweep_runs(metric, maximize, 1)[0]
        values = best_run.load_logger().numpy(metric)
        ylim = _auto_loss_yrange(values, yscale=yscale)
        if ylim is not None: ax.set_ylim(*ylim)

    if xscale is not None: ax.set_xscale(xscale)
    _set_scale_(ax, yscale)

    # --------------------------------- load runs -------------------------------- #
    main_sweeps = [task[r] for r in main]
    names = {s.run_name for s in main_sweeps}
    best_sweeps = [s for s in task.best_sweeps(metric, maximize, n_best) if s.run_name not in names]
    names = names.union(s.run_name for s in best_sweeps)
    reference_sweeps = [task[r] for r in references if r in task and task[r].run_name not in names]

    # ----------------------------------- plot ----------------------------------- #
    _plot_sweep(ax, reference_sweeps, metric, maximize, _COLORS_REFERENCES, 0.5, 5)

    # for best if no references/main are available, use better main colos
    colors_best = _COLORS_BEST
    if len(reference_sweeps) + len(main_sweeps) == 0:
        colors_best = _COLORS_MAIN + _COLORS_REFERENCES + _COLORS_BEST
    _plot_sweep(ax, best_sweeps,  metric, maximize, colors_best, 0.5, 5)

    _plot_sweep(ax, main_sweeps, metric, maximize, _COLORS_MAIN, 1., 10)

    # -------------------------------- ax settings ------------------------------- #
    name = task.task_name
    if len(main) == 1: name = f'{main[0]} - {name}'
    if name is not None: ax.set_title(_wrap(name), fontsize=9)
    ax.set_ylabel(metric)
    ax.set_xlabel("hyperparameter")
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax

# endregion

# region plot_train_test_sweep
def plot_train_test_sweep(
    sweep: "Sweep",
    xscale: Any = 'log',
    yscale: Scale = None,
    ax: Axes | None = None,
):

    if ax is None: ax = plt.gca()
    if len(sweep) == 0: return ax


    # ---------------------------- determine y limits ---------------------------- #
    best_run = sweep.best_runs('test loss', False, 1)[0]
    best_run.load_logger()
    ylim = _auto_loss_yrange(best_run.logger.numpy('train loss'), best_run.logger.numpy('test loss'), yscale=yscale)
    if ylim is not None: ax.set_ylim(*ylim)

    if xscale is not None: ax.set_xscale(xscale)
    _set_scale_(ax, yscale)

    # -------------------------------- plot -------------------------------- #
    hyperparam = None
    if len(sweep) == 1:
        ax.axhline(sweep[0].stats['train loss']['min'], c='red', lw=0.5, ls='--', label='train loss')
        ax.axhline(sweep[0].stats['test loss']['min'], c='blue', lw=1.5, ls='--', label='test loss')

    else:
        hyperparam = _find_different(*(r.hyperparams for r in sweep))
        if hyperparam is None: return ax
        train_values = [(run.hyperparams[hyperparam], run.stats['train loss']['min']) for run in sweep]
        test_values = [(run.hyperparams[hyperparam], run.stats['test loss']['min']) for run in sweep]

        train_values.sort(key=lambda x: x[0])
        test_values.sort(key=lambda x: x[0])

        ax.plot(*zip(*train_values), label='train loss', c='red', lw=0.5)
        ax.scatter(*zip(*train_values), c='red', s=5, alpha=0.5,)

        ax.plot(*zip(*test_values), label='test loss', c='blue', lw=1.5)
        ax.scatter(*zip(*test_values), c='blue', s=15, alpha=0.5,)

    # -------------------------------- ax settings ------------------------------- #
    ax.set_title(_wrap(f'{sweep.run_name} - {sweep.task_name}'), fontsize=9)
    ax.set_ylabel('loss')
    if hyperparam is not None: ax.set_xlabel(hyperparam)
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax
# endregion

# region bar_chart
def bar_chart(
    task: "Task",
    metric: str,
    maximize: bool,
    n=24,
    main=None,
    references = None,
    scale: Scale = None,
    ax: Axes | None = None,
):
    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    if ax is None: ax = plt.gca()
    if len(task) == 0: return ax

    # --------------------------------- load runs -------------------------------- #
    sweeps = task.best_sweeps(metric, maximize, n=n)
    runs = [s.best_runs(metric, maximize, n=1)[0] for s in sweeps]

    # --------------------------- load best keys/values -------------------------- #
    key = 'max' if maximize else 'min'
    runs = [r for r in runs if metric in r.stats][:32]
    keys = [_wrap(r.string(metric), 50) for r in runs]
    values = [r.stats[metric][key] for r in runs]
    colors = ['cornflowerblue' for _ in keys]

    # ------------------------- set main run color to red ------------------------ #
    for ref in references:
        names = [r.run_name for r in runs]
        if ref in names: colors[names.index(ref)] = 'blue'

    for m in main:
        names = [r.run_name for r in runs]
        if m in names: colors[names.index(m)] = 'red'

    # --------------------------------- plotting --------------------------------- #
    ax.grid(which='major', axis='x', lw=0.5)
    ax.grid(which='minor', axis='x', lw=0.5, alpha=0.15)
    _set_scale_(ax, scale, which='x')
    if _is_log_scale(scale):
        ax.xaxis.set_major_locator(ticker.LogLocator(numticks=999))
        ax.xaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    else:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.tick_params(axis='y', labelsize=2)
    ax.barh(keys, values, color=colors)
    return ax

# endregion

# region summary
def summary_df(root:str = "optimizers", include_partial:bool=True):
    from .run import Task
    decoder = msgspec.msgpack.Decoder()
    dirs = [os.path.join(root, d) for d in os.listdir(root)]
    tasks = [Task.load(d, load_loggers=False, decoder=decoder) for d in dirs if os.path.isdir(d)]
    tasks = [t for t in tasks if len(t) != 0]

    tasks_list = []
    for task in tasks:
        log_scale = task.task_name in _YSCALES and _is_log_scale(_YSCALES[task.task_name])
        assert task.target_metrics is not None

        for target_metric, maximize in task.target_metrics.items():

            row_name = task.task_name if len(task.target_metrics) == 1 else f"{task.task_name} - {target_metric}"
            task_dict: dict = dict(name=row_name)

            for sweep in task.values():
                assert sweep.run_name is not None
                assert sweep.target_metrics is not None

                key = 'max' if maximize else 'min'
                best_run = sweep.best_runs(target_metric, maximize, 1)[0]
                value = best_run.stats[target_metric][key]
                if log_scale:
                    assert value >= 0, f"{task.task_name} has value {value} and log scale"
                    value = math.log(value + 1e-12)

                # skip runs that were terminated prematurely (those will appear white)
                if not include_partial:
                    if 'GD' not in task: warnings.warn(f"{task.task_name} has no `GD` run!")
                    else:
                        sgd_logger = task['GD'][0].load_logger()
                        cur_logger = best_run.load_logger()
                        if sgd_logger.list('num passes')[-1] * 0.9 > cur_logger.list('num passes')[-1]: continue

                task_dict[_wrap(sweep.run_name, 50)] = value

            tasks_list.append(task_dict)

    import polars as pl
    df = pl.from_dicts(tasks_list)

    # make name first and sort other cols
    df = df.select(["name"] + sorted(col for col in df.columns if col != 'name'))

    return df


def summary_table(root:str, n=128, ax=None, main=None, references=None):
    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    if ax is None: ax = plt.gca()

    import polars as pl
    df = summary_df(root, include_partial=False)

    # remove 2d
    df = df.filter(pl.col('name').str.contains("2D - ").not_())

    # remove graph layout
    df = df.filter(pl.col('name').str.contains("Visual - Graph layout optimization").not_())

    # transpose so that optimizers are rows
    df = (df
          .select(col for col in df.columns if col != 'name')
          .transpose(include_header=True, header_name='run', column_names = pl.Series(df.select('name')).to_list()))

    # normalize to (0,1)
    vals = sorted(col for col, dtype in df.schema.items() if dtype.is_numeric())

    df = df.with_columns([
        ((pl.col(c) - pl.col(c).min()) / (pl.col(c).max() - pl.col(c).min())).alias(c) for c in vals
    ])


    # sort by sum of ranks and take first n
    rank_expressions = []
    for v in vals:
        weight = 1
        if "Friedman 1" in v: weight = 0.5
        if v.startswith("Real - "): weight = 0.6
        if v.startswith(("S - ", "SS - ")): weight = 0.4
        if v.startswith("Visual - "): weight = 0.2

        rank_expressions.append(
            pl.col(v).fill_null(strategy="max").rank(method='average').log(base=2) * weight
        )

    df = df.sort(by=pl.sum_horizontal(rank_expressions))

    # re-sort (somewhere it de-sorts)
    vals = sorted(col for col, dtype in df.schema.items() if dtype.is_numeric())
    df = df.select(["run"] + vals)

    # remove all rows (opts) except ones who have records + top10, main and references
    top10 = df.select(pl.col("run")).head(10).to_series().to_list()
    holds_record = pl.any_horizontal(pl.col(c) == pl.col(c).min() for c in vals)

    df = df.filter(holds_record | pl.col("run").is_in(main + references + top10))

    # PLOT!!!!!!!
    data = df.head(n).select(vals).to_numpy().transpose() # (rows, cols)

    # in each row set all values below 75th percentile
    # to largest value below 75th percentile
    quantile = np.nanquantile(data, 0.75, axis=1, keepdims=True)

    # find largest value in data below 75th percentile
    values_below = np.where(data < quantile, data, np.nan)
    quantile_max = np.nanmax(values_below, axis=1, keepdims=True)
    data = np.where(data >= quantile, quantile_max, data)

    # normalize filtered data
    data = data - np.nanmin(data, axis=1, keepdims=True)
    data = data / np.nanmax(data, axis=1, keepdims=True).clip(min=1e-16)

    # cmap = plt.get_cmap("coolwarm").copy()
    # cmap.set_bad('black')
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_to_red", ["blue", "red"])
    cmap.set_bad('white')

    ax.pcolormesh(data, cmap=cmap, edgecolors='w', linewidth=2)

    # highlight best values
    for row_idx in range(data.shape[0]):
        row = data[row_idx]
        if np.all(np.isnan(row)): continue
        min_col = np.nanargmin(row)
        rect = patches.Rectangle((int(min_col), row_idx), width=1, height=1, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    # labels
    ax.set_xticks(np.arange(0, data.shape[1], 1)+0.5)
    ax.set_yticks(np.arange(0, data.shape[0], 1)+0.5)
    ax.set_xticklabels(pl.Series(df.select('run')).to_list())
    ax.set_yticklabels(df.columns[1:])
    ax.tick_params(axis='x', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    ax.tick_params(axis='y', labelsize=8)
    return ax
# endregion

def _clean_empty(root):
    for f in os.listdir(root):
        path = os.path.join(root, f)
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
            else:
                _clean_empty(path)

# region render_summary
def render_summary(
    root:str,
    dirname: str,
    main: str | Sequence[str] | None,
    n_best: int = 1,
    references: str | Sequence[str] | None = REFERENCE_OPTS,

    # plotting settings
    axsize=(6,3), dpi=300
):
    from .run import Task
    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    decoder = msgspec.msgpack.Decoder()

    # ----------------------------------- plot ----------------------------------- #
    _clean_empty(root)
    for task_name in os.listdir(root):

        task_path = os.path.join(root, task_name)
        if not os.path.isdir(task_path): continue
        sweeps = os.listdir(task_path)

        # load task if a sweep was done by `main`
        if len(main) == 0 or any(sweep in main for sweep in sweeps):

            task = Task.load(task_path, load_loggers=False, decoder=decoder)
            if len(task) == 0: continue
            assert task.task_name is not None
            assert task.target_metrics is not None
            yscale = _YSCALES.get(task.task_name, None)

            # if there is test loss, plot train/test separately in extra row
            has_test = False
            if len(main) > 0:
                # get 1st non empty sweep and 1st run to see if it has test loss
                run1 = None
                sweep1 = None
                for sweep in task.values():
                    if len(sweep) > 0: sweep1 = sweep
                if sweep1 is not None:
                    run1 = sweep1[0]
                if run1 is not None and 'test loss' in run1.stats:
                    has_test = True

            n_metrics = len(task.target_metrics)
            nrows = n_metrics + has_test
            axes = make_axes(n=nrows*2+n_metrics, ncols=2, axsize=axsize, dpi=dpi)
            axes_iter = iter(axes)

            if has_test:
                sweep = task[main[0]]
                # plot train/test losses of current opt
                ax = next(axes_iter)
                plot_train_test_values(sweep, yscale, ax)

                # plot train/test sweep of current opt
                ax = next(axes_iter)
                plot_train_test_sweep(sweep, xscale='log', yscale=yscale, ax=ax)

            # plot all metrics
            for metric, maximize in task.target_metrics.items():
                # plot values
                ax = next(axes_iter)
                smoothing = 0
                if metric == 'train loss': smoothing = _TRAIN_SMOOTHING.get(task.task_name, 0)
                plot_values(task, metric=metric, maximize=maximize, main=main, references=references, n_best=n_best, yscale=yscale, smoothing=smoothing, ax=ax)

                # plot sweep
                ax = next(axes_iter)
                plot_sweeps(task, metric=metric, maximize=maximize, main=main, references=references, n_best=n_best, xscale='log', yscale=yscale, ax=ax)

            # bars
            for metric, maximize in task.target_metrics.items():
                # plot values
                ax = next(axes_iter)
                bar_chart(task, metric, maximize, main=main, references=references, scale=yscale, ax=ax)
            # ---------------------------------- save ts --------------------------------- #
            # for fn in queue: fn()
            if not os.path.exists(dirname): os.mkdir(dirname)
            plt.savefig(os.path.join(dirname, f"{to_valid_fname(task.task_name)}.png"))
            plt.close()

    # ------------------------------- plot summary ------------------------------- #
    ax = make_axes(1, figsize=(20,20))[0]
    ax = summary_table(root, ax=ax, main=main, references=references)
    plt.colorbar(ax.collections[0])
    # fig: Any = ax.get_figure()
    # fig.set_size_inches(20, 20)
    plt.savefig(os.path.join(dirname, "summary.png"))
    plt.close()
# endregion