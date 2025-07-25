import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import msgspec
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.scale import SymmetricalLogScale
from scipy.ndimage import gaussian_filter1d

from ..utils.plt_tools import _auto_loss_yrange, legend_, make_axes
from ..utils.python_tools import format_number, to_valid_fname

if TYPE_CHECKING:
    from ..logger import Logger
    from ..runs.run import Run, Sweep, Task

REFERENCE_OPTS = ("SGD", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW", "L-BFGS", "BFGS-Backtracking", "Newton")

_YSCALES: dict[str, Any] = {
    # ML
    "ML - Olivetti Faces FB - Logistic Regression": dict(value='symlog', linthresh=1e-12),
    "ML - Friedman 1 - Linear Regression - L1": "log",
    "ML - MNIST-1D FB - NeuralODE": "log",
    "ML - Wave PDE - TinyFLS": "log",
    "ML - Wave PDE - FLS": "log",

    # 2D
    "2D - booth": dict(value='symlog', linthresh=1e-8),
    "2D - ill": dict(value='symlog', linthresh=1e-6),
    "2D - star": "log",
    "2D - rosenbrock": dict(value='symlog', linthresh=1e-8),
    "2D - rosenbrock abs": "log",
    "2D - spiral": "log",
    "2D - illppc": "log",
    "2D - oscillating": dict(value='symlog', linthresh=1e-6),
    "2D simultaneous - rosenbrock-10": "log",
    "2D simultaneous - rosenbrock": "log",
    "2D simultaneous - rosenbrock abs": "log",
    "2D simultaneous - rosenbrock rastrigin": "log",
    "2D simultaneous - oscillating": "log",

    # Losses
    "ML - Friedman 1 - Linear Regression - L-Infinity": "log",
    "ML - Friedman 1 - Linear Regression - L4": "log",
    "ML - Friedman 1 - Linear Regression - Median": "log",
    "ML - Friedman 1 - Linear Regression - Quartic": "log",

    # Synthetic
    "S - Ill conditioned quadratic": dict(value='symlog', linthresh=1e-12),
    "S - Rosenbrock": "log",
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
    "Visual - Colorization": dict(value='symlog', linthresh=1e-12),
    "Visual - Colorization (2nd order)": dict(value='symlog', linthresh=1e-12),
    "Visual - Graph layout optimization": "log",
    "Visual - Style Transfer": "log",
    "Visual - Sine Approximator - Tanh 7-4": "log",
    "Visual - Muon coefficients": "log",

    # real
    "Real - Human heart dipole": "log",
    "Real - Propane combustion": "log",
}

_TRAIN_SMOOTHING: dict[str, float] = {
    "SS - Stochastic inverse - L1": 2,
    "SS - Stochastic inverse - MSE": 2,
    "SS - Stochastic matrix recovery - L1": 2,
    "SS - Stochastic matrix recovery - MSE": 2,
    "SS - Stochastic matrix idempotent": 2,
    "SS - Stochastic matrix idempotent (hard)": 2,
    "MLS - Covertype Online - Logistic Regression": 2
}



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
    ax.plot(bte_train_steps, bte_train_values, label=f"te - train: {format_number(np.nanmin(bte_train_values), 5)}", c='darkgreen', lw=0.5, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_train_steps, btr_train_values, label=f"tr - train: {format_number(np.nanmin(btr_train_values), 5)}", c='darkred', lw=0.5, alpha=0.5)

    ax.plot(bte_test_steps, bte_test_values, label=f"te - test: {format_number(np.nanmin(bte_test_values), 5)}", c='lime', lw=1.0, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_test_steps, btr_test_values, label=f"tr - test: {format_number(np.nanmin(btr_test_values), 5)}", c='red', lw=1.0, alpha=0.5)


    # ------------------------------- axes and grid ------------------------------ #
    ax.set_title(f'{sweep.run_name} - {sweep.task_name}')
    ax.set_ylabel('loss')
    ax.set_xlabel('num forward/backward passes')
    legend_(ax)

    _xaxis_settings_(ax, yscale)
    return ax


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

def _plot_metric(
    ax: Axes,
    runs: "Sequence[Run]",
    metric: str,
    maximize: bool,
    smoothing: float,
    colors: Sequence,
    **plot_kwargs,
):
    while len(colors) < len(runs):
        print(f"ADD {len(runs) - len(colors)} MORE COLORS TO {colors}!!!")
        colors = list(colors).copy()
        colors.append("pink")

    for r,c in zip(runs,colors):
        steps, values = _load_steps_values(r.load_logger(), metric)
        best = np.nanmax(values) if maximize else np.nanmin(values)

        if smoothing != 0: values = gaussian_filter1d(values, smoothing, mode='nearest')
        ax.plot(steps, values, label=_make_label(r, best, _get_1st_key(r.hyperparams)), c=c, **plot_kwargs)

    return ax

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
    _plot_metric(ax, best_runs, metric, maximize, smoothing, _COLORS_BEST, lw=0.5)
    _plot_metric(ax, main_runs, metric, maximize, smoothing, _COLORS_MAIN)

    name = task.task_name
    if len(main) == 1: name = f'{main[0]} - {name}'
    if name is not None: ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel('num forward/backward passes')
    legend_(ax)

    # ------------------------------- axes and grid ------------------------------ #
    _xaxis_settings_(ax, yscale)
    return ax


def _plot_sweep(
    ax: Axes,
    sweeps: "list[Sweep]",
    metric: str,
    maximize: bool,
    colors: Sequence,
    lw,
    marker_size,
):
    while len(colors) < len(sweeps):
        colors = list(colors).copy()
        colors.append("pink")

    for s,c in zip(sweeps,colors):
        key = 'max' if maximize else 'min'
        if len(s) == 1:
            ax.axhline(s[0].stats[metric][key], c=c, lw=lw, ls='--', label=s.run_name)
        else:
            hyperparam = _find_different(*(r.hyperparams for r in s))
            if hyperparam is None: continue
            values = [(run.hyperparams[hyperparam], run.stats[metric][key]) for run in s]
            values.sort(key=lambda x: x[0])
            ax.plot(*zip(*values), label=s.run_name, c=c, lw=lw)
            ax.scatter(*zip(*values), c=c, s=marker_size, alpha=0.5,)

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
    _plot_sweep(ax, best_sweeps,  metric, maximize, _COLORS_BEST, 0.5, 5)
    _plot_sweep(ax, main_sweeps, metric, maximize, _COLORS_MAIN, 1., 10)

    # -------------------------------- ax settings ------------------------------- #
    name = task.task_name
    if len(main) == 1: name = f'{main[0]} - {name}'
    if name is not None: ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel("hyperparameter")
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax

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
    ax.set_title(f'{sweep.run_name} - {sweep.task_name}')
    ax.set_ylabel('loss')
    if hyperparam is not None: ax.set_xlabel(hyperparam)
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax


def bar_chart(
    task: "Task",
    metric: str,
    maximize: bool,
    n=32,
    references = None,
    scale: Scale = None,
    ax: Axes | None = None,
):
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
    keys = [r.string(metric) for r in runs]
    values = [r.stats[metric][key] for r in runs]
    colors = ['cornflowerblue' for _ in keys]

    # ------------------------- set main run color to red ------------------------ #
    for ref in references:
        names = [r.run_name for r in runs]
        if ref in names: colors[names.index(ref)] = 'red'

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

    ax.tick_params(axis='y', labelsize=7)
    ax.barh(keys, values, color=colors)
    return ax


def _clean_empty(root):
    for f in os.listdir(root):
        path = os.path.join(root, f)
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
            else:
                _clean_empty(path)


def render_summary(
    root:str,
    dirname: str,
    main: str | Sequence[str] | None,
    n_best: int = 1,
    references: str | Sequence[str] | None = REFERENCE_OPTS,

    # plotting settings
    axsize=(12,6), dpi=100
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
            axes = make_axes(n=nrows*2+n_metrics, nrows=nrows+1, ncols=2, axsize=axsize, dpi=dpi)
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
                bar_chart(task, metric, maximize, references=references, scale=yscale, ax=ax)
            # ---------------------------------- save ts --------------------------------- #
            # for fn in queue: fn()
            if not os.path.exists(dirname): os.mkdir(dirname)
            plt.savefig(os.path.join(dirname, f"{to_valid_fname(task.task_name)}.png"))
            plt.close()