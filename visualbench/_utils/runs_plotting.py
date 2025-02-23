from typing import Literal, Any
from collections.abc import Sequence
import os

import numpy as np
from myai.logger import DictLogger
from myai.plt_tools import Fig

from .runs import REFERENCE_OPTS, TaskInfo
from .utils import _round_significant

def _get_lr_to_logger(task_name, opt, root) -> dict[str,DictLogger]:
    """returns list of (lr, logger) tuples sorted such that best value is 1st"""
    opt_path = os.path.join(root, task_name, opt)
    return {f.replace('.npz', ''): DictLogger.from_file(os.path.join(opt_path, f)) for f in os.listdir(opt_path)}

def _print_best(task_name, root = 'runs', metric = None):
    task_path = os.path.join(root, task_name)
    if not os.path.exists(os.path.join(root, task_name)): raise FileNotFoundError(os.path.join(root, task_name))
    info = TaskInfo(root, task_name)
    if metric is None: metric = list(info.target_metrics.keys())[0]
    maximize = info.metrics[metric].maximize
    if maximize is None:
        print(f"maximize is None for {task_name} {metric}")
        maximize = False

    opts = {}
    for opt in os.listdir(task_path):
        if opt == 'info.yaml': continue
        opt_path = os.path.join(task_path, opt)
        loggers = [DictLogger.from_file(os.path.join(opt_path, f)) for f in os.listdir(opt_path)]
        if maximize:
            y = [logger.max(metric) for logger in loggers]
            if len(y) > 0: best = np.nanmax(y)
            else: best = -float('inf')
        else:
            y = [logger.min(metric) for logger in loggers]
            if len(y) > 0: best = np.nanmin(y)
            else: best = float('inf')
        opts[opt] = best

    opts_sorted = sorted(list(opts.items()), key = lambda x: x[1], reverse=maximize)
    for opt, best in opts_sorted:
        print(opt, _round_significant(best, 3))

def _get_reference_opts(ref, root, task_name):
    if isinstance(ref, str):
        if ref == 'all':
            ref = os.listdir(os.path.join(root, task_name))
            ref.remove('info.yaml')
        else:
            ref = (ref, )
    return ref

def plot_lr_search_curve(task_name, opts, root='runs', metric = None, ref:str|Sequence[str]|None|Literal['all']=REFERENCE_OPTS, log_scale = False, fig=None, show = True):
    """plots opts, reference opts and best opts"""
    if isinstance(opts, str): opts = (opts, )
    ref = _get_reference_opts(ref, root, task_name)
    # if opts is None: opts = ()
    if fig is None: fig = Fig()

    info = TaskInfo(root, task_name)
    if metric is None: metric = list(info.target_metrics.keys())[0]
    maximize = info.metrics[metric].maximize
    if maximize is None: print(f"maximize is None for {task_name} {metric}")
    plotted = set()

    # plot func
    def _plot_opt(opt_name, display_name, lw, is_main):
        if opt_name in plotted: return
        plotted.add(opt_name)

        if not os.path.exists(os.path.join(root, task_name, opt_name)): return
        lr_logger = list(_get_lr_to_logger(task_name, opt_name, root).items())
        lr_logger.sort(key = lambda x: float(x[0]))

        x = [float(lr) for lr, logger in lr_logger]
        if maximize:
            y = [logger.max(metric) for lr, logger in lr_logger]
            best = np.max(y)
        else:
            y = [logger.min(metric) for lr, logger in lr_logger]
            best = np.min(y)

        kw = {}
        if lw is not None: kw['lw'] = lw
        if is_main: kw['color'] = 'blue'
        if len(x) > 1: fig.linechart(x=x, y=y, label=f'{display_name} - {_round_significant(best, 3)}', marker = '.', **kw)

    # plot opts
    if opts is not None:
        for opt in opts: _plot_opt(opt, opt, lw = None, is_main=True)

    # plot reference opts
    if ref is not None:
        for ref_opt in ref: _plot_opt(ref_opt, ref_opt, 1, is_main=False)

    # plot best opt for reference
    best = info.best_opt(metric)
    _plot_opt(best, f"best: {best}", 1, is_main=False)

    # actual plotting
    if log_scale: fig.yscale('symlog', linthresh = 1e-8)

    # determine ymin and ymax because plt is stupid sometimes
    ymin = info.metrics[metric].min_value
    ymax = info.metrics[metric].max_value if maximize else info.metrics[metric].first
    assert ymax is not None
    d = (ymax - ymin)*0.1
    ymax = ymax + d
    if not log_scale: ymin = ymin-d
    fig.ylim(ymin, ymax)

    fig.xscale('log').preset(xlabel = 'lr', ylabel = metric, legend=True)
    if show: fig.show()
    return fig


def plot_metric(task_name, opts, root='runs', metric = None, opts_all_lrs = True, ref:str|Sequence[str]|None|Literal['all']=REFERENCE_OPTS, log_scale = False, fig=None, show = True):
    """plots opts, reference opts and best opts"""
    if isinstance(opts, str): opts = (opts, )
    ref = _get_reference_opts(ref, root, task_name)
    # if opts is None: opts = ()
    if fig is None: fig = Fig()

    info = TaskInfo(root, task_name)
    if not os.path.exists(info.path): raise FileNotFoundError(info.path)
    if metric is None: metric = list(info.target_metrics.keys())[0]
    maximize = info.metrics[metric].maximize
    if maximize is None: print(f"maximize is None for {task_name} {metric}")
    plotted = set()

    # plot func
    def _plot_opt(opt_name, display_name, all_lrs, is_main,):
        if opt_name in plotted: return
        plotted.add(opt_name)

        if not os.path.exists(os.path.join(root, task_name, opt_name)): return
        lr_logger = list(_get_lr_to_logger(task_name, opt_name, root).items())

        if maximize: lr_logger.sort(key = lambda x: x[1].max(metric), reverse=True)
        else: lr_logger.sort(key = lambda x: x[1].min(metric))


        for i, (lr, logger) in enumerate(lr_logger[:5] if all_lrs else [lr_logger[0]]):
            x, y = logger.shared('num passes', metric)

            best = logger.max(metric) if maximize else logger.min(metric)
            # set linewidth to be smaller on all ones other than main one
            kw = {}
            if i != 0:
                kw['lw'] = 0.5
            if not is_main:
                if opts is not None:
                    if opts_all_lrs:
                        kw['linestyle'] = 'dashed'
                        kw['lw'] = 0.75
            else:
                kw['color'] = 'blue'
            fig.linechart(x=x, y=y, label=f'{display_name} {_round_significant(float(lr), 3)} - {_round_significant(best, 3)}', **kw)


    # plot opts
    if opts is not None:
        for opt in opts: _plot_opt(opt, opt, all_lrs=opts_all_lrs, is_main = True)

    # plot reference opts
    if ref is not None:
        for ref_opt in ref: _plot_opt(ref_opt, ref_opt, all_lrs=False, is_main = False)

    # plot best opt for reference
    best = info.best_opt(metric)
    _plot_opt(best, f"best: {best}", all_lrs=False, is_main = False)

    # actual plotting
    if log_scale: fig.yscale('symlog', linthresh = 1e-8)

    # determine ymin and ymax because plt is stupid sometimes
    ymin = info.metrics[metric].min_value
    ymax = info.metrics[metric].max_value if maximize else info.metrics[metric].first
    assert ymax is not None
    d = (ymax - ymin)*0.1
    ymax = ymax + d
    if not log_scale: ymin = ymin-d
    fig.ylim(ymin, ymax)

    fig.preset(xlabel = 'forwad/backward passes', ylabel = metric, legend=True)
    if show: fig.show()
    return fig

