import os
from collections.abc import Sequence, Callable
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

REFERENCE_OPTS = ("SGD", "NAG", "Adagrad", "RMSprop", "Adam", "AdamW", "L-BFGS", "BFGS", "Newton")

_YSCALES: dict[str, Any] = {
    # ML
    "ML - Graph layout optimization": "log",
    # "MLP(10-10-10-10-10) - MNIST-1D full-batch": "log",
    # "MLP(40-40-40-40-10) - MNIST-1D full-batch": "log",
    "ML - Style Transfer": "log",
    "ML - Olivetti Faces - Logistic Regression": dict(value='symlog', linthresh=1e-12),
    "ML - Friedman 1 - Linear Regression - L1": "log",
    "ML - MNIST-1D FB - NeuralODE": "log",

    # Losses
    # "L-infinity loss linear regression - Friedman 1": "log",
    "ML - Friedman 1 - Linear Regression - Quartic": "log",


    # Synthetic
    "S - Ill conditioned quadratic": dict(value='symlog', linthresh=1e-12),
    "S - Colorization": dict(value='symlog', linthresh=1e-12),
    "S - Colorization (2nd order)": dict(value='symlog', linthresh=1e-12),
    "S - Rosenbrock": "log",
    "S - LogSumExp": "log",
    "S - Inverse - L1": "log",
    "S - Inverse - L2": "log",
    "S - Matrix idempotent": "log",
    "S - Normal scalar curvature": "log",
    "S - Kato problem": "log",

    # synthetic stochastic
    "SS - Stochastic inverse - L1": "log",
    "SS - Stochastic inverse - L2": "log",
    "SS - Stochastic matrix recovery": "log",
}

_TRAIN_SMOOTHING: dict[str, float] = {

}



_COLORS_MAIN = ("red", "green", "blue")
_COLORS_REFERENCES = ("deepskyblue", "orange", "springgreen", "coral", "lawngreen", "aquamarine", "plum", "pink", "peru")
_COLORS_BEST = ("black", "dimgray", "maroon", "midnightblue", "darkgreen", "rebeccapurple", "darkmagenta", "saddlebrown", "darkslategray")
Yscale = None | str | dict[str, Any] | Callable[[Axes], Any]

def _maybe_format_number(x):
    if isinstance(x, (int,float)): return format_number(x, 3)
    return x

def _make_label(run: "Run", best_value: float, hyperparams: Sequence[str] | None):
    name = run.run_name
    assert name is not None
    if hyperparams is None: return name

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

def _set_yscale_(ax: Axes, yscale: Yscale):
    if yscale is None: return ax
    if isinstance(yscale, str): ax.set_yscale(yscale)
    elif isinstance(yscale, dict): ax.set_yscale(**yscale)
    elif callable(yscale): yscale(ax)
    else: raise ValueError(f"Invalid yscale {yscale}")
    return ax

def _is_log_yscale(yscale: Yscale):
    if yscale is None: return False
    if isinstance(yscale, str): return 'log' in yscale
    if isinstance(yscale, dict):
        yscale = yscale['value']
        return _is_log_yscale(yscale)
    if callable(yscale): return False
    raise ValueError(f"Invalid yscale {yscale}")

def _xaxis_settings_(ax:Axes, yscale: Yscale):
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if _is_log_yscale(yscale):
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
    yscale: Yscale = None,
    ax: Axes | None = None,
):
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

    _set_yscale_(ax, yscale)

    # ---------------------------- plot ---------------------------- #
    ax.plot(bte_train_steps, bte_train_values, label=f"te - train: {format_number(bte_train_values.min(), 5)}", c='darkgreen', lw=0.5, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_train_steps, btr_train_values, label=f"tr - train: {format_number(btr_train_values.min(), 5)}", c='darkred', lw=0.5, alpha=0.5)

    ax.plot(bte_test_steps, bte_test_values, label=f"te - test: {format_number(bte_test_values.min(), 5)}", c='lime', lw=1.0, alpha=0.5)

    if best_train != best_test:
        ax.plot(btr_test_steps, btr_test_values, label=f"tr - test: {format_number(btr_test_values.min(), 5)}", c='red', lw=1.0, alpha=0.5)


    # ------------------------------- axes and grid ------------------------------ #
    ax.set_title(f'{sweep.run_name} - {sweep.task_name}')
    ax.set_ylabel('loss')
    ax.set_xlabel('num forward/backward passes')
    legend_(ax)

    _xaxis_settings_(ax, yscale)
    return ax


def _plot_metric(
    ax: Axes,
    runs: "Sequence[Run]",
    metric: str,
    maximize: bool,
    hyperparams: Sequence[str] | None,
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
        best = values.max() if maximize else values.min()

        if smoothing != 0: values = gaussian_filter1d(values, smoothing, mode='nearest')
        ax.plot(steps, values, label=_make_label(r, best, hyperparams), c=c, **plot_kwargs)

    return ax

def plot_values(
    task: "Task",
    metric: str,
    maximize: bool,
    main: str | Sequence[str] | None,
    references: str | Sequence[str] | None,
    n_best: int,
    hyperparams: str | Sequence[str] | None,
    yscale = None,
    smoothing: float = 0,
    ax: Axes | None = None
):
    if ax is None: ax = plt.gca()

    if main is None: main = []
    if isinstance(main, str): main = [main]

    if references is None: references = []
    if isinstance(references, str): references = [references]

    if isinstance(hyperparams, str): hyperparams = [hyperparams]

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

    _set_yscale_(ax, yscale)

    # ----------------------------------- plot ----------------------------------- #
    _plot_metric(ax, reference_runs, metric, maximize, hyperparams, smoothing, _COLORS_REFERENCES, lw=0.5)
    _plot_metric(ax, best_runs, metric, maximize, hyperparams, smoothing, _COLORS_BEST, lw=0.5)
    _plot_metric(ax, main_runs, metric, maximize, hyperparams, smoothing, _COLORS_MAIN)

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
    hyperparam: str,
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
    if _is_log_yscale(yscale):
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
    hyperparam: str,
    main: str | Sequence[str] | None,
    references: str | Sequence[str] | None,
    n_best: int,
    xscale: Any = 'log',
    yscale: Yscale = None,
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
    _set_yscale_(ax, yscale)

    # --------------------------------- load runs -------------------------------- #
    main_sweeps = [task[r] for r in main]
    names = {s.run_name for s in main_sweeps}
    best_sweeps = [s for s in task.best_sweeps(metric, maximize, n_best) if s.run_name not in names]
    names = names.union(s.run_name for s in best_sweeps)
    reference_sweeps = [task[r] for r in references if r in task and task[r].run_name not in names]

    # ----------------------------------- plot ----------------------------------- #
    _plot_sweep(ax, reference_sweeps, metric, maximize, hyperparam, _COLORS_REFERENCES, 0.5, 5)
    _plot_sweep(ax, best_sweeps,  metric, maximize, hyperparam, _COLORS_BEST, 0.5, 5)
    _plot_sweep(ax, main_sweeps, metric, maximize, hyperparam, _COLORS_MAIN, 1., 10)

    # -------------------------------- ax settings ------------------------------- #
    name = task.task_name
    if len(main) == 1: name = f'{main[0]} - {name}'
    if name is not None: ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel(hyperparam)
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax

def plot_train_test_sweep(
    sweep: "Sweep",
    hyperparam: str,
    xscale: Any = 'log',
    yscale: Yscale = None,
    ax: Axes | None = None,
):
    if ax is None: ax = plt.gca()

    # ---------------------------- determine y limits ---------------------------- #
    best_run = sweep.best_runs('test loss', False, 1)[0]
    best_run.load_logger()
    ylim = _auto_loss_yrange(best_run.logger.numpy('train loss'), best_run.logger.numpy('test loss'), yscale=yscale)
    if ylim is not None: ax.set_ylim(*ylim)

    if xscale is not None: ax.set_xscale(xscale)
    _set_yscale_(ax, yscale)

    # -------------------------------- plot -------------------------------- #
    if len(sweep) == 1:
        ax.axhline(sweep[0].stats['train loss']['min'], c='red', lw=0.5, ls='--', label='train loss')
        ax.axhline(sweep[0].stats['test loss']['min'], c='blue', lw=1.5, ls='--', label='test loss')

    else:
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
    ax.set_xlabel(hyperparam)
    legend_(ax)
    _sweep_xyaxes(ax, xscale, yscale)

    return ax



def render_summary(
    root:str,
    fname: str,
    main: str | Sequence[str] | None,
    hyperparams: str | Sequence[str] | None,
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

    if isinstance(hyperparams, str): hyperparams = [hyperparams]

    decoder = msgspec.msgpack.Decoder()

    # ------------------------- determine number of axes ------------------------- #
    nrows = 0
    tasks: "list[Task]" = []
    w_test_loss: list[bool] = []
    for task_name in os.listdir(root):

        task_path = os.path.join(root, task_name)
        if not os.path.isdir(task_path): continue
        sweeps = os.listdir(task_path)

        # load task if a sweep was done by `main`
        if len(main) == 0 or any(sweep in main for sweep in sweeps):

            task = Task.load(task_path, load_loggers=False, decoder=decoder)
            if len(task) == 0: continue
            assert task.target_metrics is not None

            tasks.append(task)
            has_test = False
            nrows += len(task.target_metrics)

            # if there is test loss, plot train/test separately in extra row
            if len(main) > 0:
                # get 1st non empty sweep and 1st run to see if it has test loss
                run1 = None
                sweep1 = None
                for sweep in task.values():
                    if len(sweep) > 0: sweep1 = sweep
                if sweep1 is not None:
                    run1 = sweep1[0]
                if run1 is not None and 'test loss' in run1.stats:
                    nrows += 1
                    has_test = True

            w_test_loss.append(has_test)


    # --------------------------------- make axes -------------------------------- #
    if hyperparams is None: nrows = int(nrows / 2)
    axes = make_axes(n=nrows*2, ncols=2, nrows=nrows, axsize=axsize, dpi=dpi)
    axes_iter = iter(axes)

    # sort by task name, tasks with no test loss are first
    zipped = list(zip(tasks, w_test_loss))
    zipped.sort(key=lambda x: (x[1], x[0].task_name if x[0].task_name is not None else 0))

    # ----------------------------------- plot ----------------------------------- #
    for task, has_test in zipped:
        assert task.task_name is not None
        assert task.target_metrics is not None

        yscale = _YSCALES.get(task.task_name, None)
        if has_test:
            sweep = task[main[0]]
            # plot train/test losses of current opt
            ax = next(axes_iter)
            plot_train_test_values(sweep, yscale, ax)

            # plot train/test sweep of current opt
            if hyperparams is not None:
                ax = next(axes_iter)
                plot_train_test_sweep(sweep, hyperparams[0], xscale='log', yscale=yscale, ax=ax)

        # plot all metrics
        for metric, maximize in task.target_metrics.items():
            # plot values
            ax = next(axes_iter)
            smoothing = 0
            if metric == 'train loss': smoothing = _TRAIN_SMOOTHING.get(task.task_name, 0)
            plot_values(task, metric=metric, maximize=maximize, main=main, references=references, n_best=n_best, hyperparams=hyperparams, yscale=yscale, smoothing=smoothing, ax=ax)

            # plot sweep
            if hyperparams is not None:
                ax = next(axes_iter)
                plot_sweeps(task, metric=metric, maximize=maximize, hyperparam=hyperparams[0], main=main, references=references, n_best=n_best, xscale='log', yscale=yscale, ax=ax)

    # ---------------------------------- save ts --------------------------------- #
    # for fn in queue: fn()
    plt.savefig(fname)
    plt.close()



def render_summary_v2(
    root:str,
    dirname: str,
    main: str | Sequence[str] | None,
    hyperparams: str | Sequence[str] | None,
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

    if isinstance(hyperparams, str): hyperparams = [hyperparams]

    decoder = msgspec.msgpack.Decoder()

    # ----------------------------------- plot ----------------------------------- #
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

            nrows = len(task.target_metrics) + has_test
            axes = make_axes(n=nrows*2, nrows=nrows, ncols=2, axsize=axsize, dpi=dpi)
            axes_iter = iter(axes)

            if has_test:
                sweep = task[main[0]]
                # plot train/test losses of current opt
                ax = next(axes_iter)
                plot_train_test_values(sweep, yscale, ax)

                # plot train/test sweep of current opt
                if hyperparams is not None:
                    ax = next(axes_iter)
                    plot_train_test_sweep(sweep, hyperparams[0], xscale='log', yscale=yscale, ax=ax)

            # plot all metrics
            for metric, maximize in task.target_metrics.items():
                # plot values
                ax = next(axes_iter)
                smoothing = 0
                if metric == 'train loss': smoothing = _TRAIN_SMOOTHING.get(task.task_name, 0)
                plot_values(task, metric=metric, maximize=maximize, main=main, references=references, n_best=n_best, hyperparams=hyperparams, yscale=yscale, smoothing=smoothing, ax=ax)

                # plot sweep
                if hyperparams is not None:
                    ax = next(axes_iter)
                    plot_sweeps(task, metric=metric, maximize=maximize, hyperparam=hyperparams[0], main=main, references=references, n_best=n_best, xscale='log', yscale=yscale, ax=ax)

            # ---------------------------------- save ts --------------------------------- #
            # for fn in queue: fn()
            if not os.path.exists(dirname): os.mkdir(dirname)
            plt.savefig(os.path.join(dirname, f"{to_valid_fname(task.task_name)}.png"))
            plt.close()