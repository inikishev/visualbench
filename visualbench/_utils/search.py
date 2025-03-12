import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import bottleneck as bn
import numpy as np
from glio.jupyter_tools import clean_mem
from myai.loaders.yaml import yamlwrite
from myai.logger import DictLogger

from scipy.ndimage import gaussian_filter1d
from .runs import TaskInfo, _clean_empty, _filter_logger_
from .utils import _round_significant

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _double_round(x, round1, round2):
    return _round_significant(_round_significant(x, round1), round2)

class _BinarySearch:
    def __init__(
        self,
        fn: Callable[[float], dict[str, float]],
        target_metrics: dict[str, bool],
        log10_lrs: Sequence[float],
        max_binary_search_steps: int,
        max_expansions: int,
        rounding: int,
        debug: bool,
    ):
        self.fn = fn
        """function that takes in lr, evaluates or loads if already evaluated, and returns dictionary of metrics to minimize"""

        self.target_metrics = target_metrics
        """dict with {metric_name: maximize}"""

        self.lrs = list(log10_lrs)
        """list of log10(lr)"""

        self.max_binary_search_steps = max_binary_search_steps
        self.max_expansions = max_expansions
        self.rounding = rounding
        self._debug = debug

        self.binary_search_steps = 0
        self.expansions = 0

        self._lr_metrics: dict[float, dict[str, float]] = {}
        self.suggested_lrs = []

    def rounded_suggested_lrs(self):
        # rounding with self.rounding+2 fixes rounding error where 0.55 is rounded to 0.55 and 0.55000000001 to 0.56
        return [_double_round(i,self.rounding+2, self.rounding) for i in self.suggested_lrs]

    def rounded_evaluated_lrs(self):
        return [_double_round(i,self.rounding+2, self.rounding) for i in self.evaluated_lrs()]

    def suggest_lr(self, log10_lr):

        if _double_round(log10_lr,self.rounding+2, self.rounding) not in self.rounded_suggested_lrs() + self.rounded_evaluated_lrs():
            self.suggested_lrs.append(_double_round(log10_lr, 7, 5))

    def debug(self,*args,**kwargs):
        """print"""
        if self._debug: print(*args, **kwargs)

    def evaluate_lr(self, log10_lr):
        lr = _double_round(10 ** log10_lr, 7, 5)
        values = self.fn(lr)
        self._lr_metrics[log10_lr] = values

    def lr_to_metrics_by_lr(self) -> OrderedDict[float, dict[str, float]]:
        """dict with {log10_lr: {metric_name: value}}, sorted by lr. If value is maximized, then it is negated to make sure we always minimize"""
        return OrderedDict((k,v) for k,v in sorted(list(self._lr_metrics.items()), key = lambda x: x[0]))

    def lr_to_value_by_lr(self, metric: str) -> dict[float, float]:
        """dict with {log10_lr: value} sorted by lr"""
        return OrderedDict((k,v[metric]) for k,v in sorted(list(self._lr_metrics.items()), key = lambda x: x[0]))

    def lr_to_value_by_value(self, metric: str) -> dict[float, float]:
        """dict with {log10_lr: value} sorted by value"""
        return OrderedDict((k,v[metric]) for k,v in sorted(list(self._lr_metrics.items()), key = lambda x: x[1][metric]))

    def evaluated_lrs(self):
        """list with sorted lrs"""
        return sorted(self._lr_metrics)

    def evaluate_base_lrs(self):
        """step 1 - evaluates base lrs usually 10, 1, 0.1, etc"""
        for lr in self.lrs:
            self.evaluate_lr(lr)

    def binary_search_suggest(self, shift: int) -> Literal['overflow', 'expand', 'binary search']:
        """binary search step on `metric`"""
        self.debug(f'suggesting {shift = }')

        # if shift overflows, stop binary search
        if 2+shift > len(self._lr_metrics): return 'overflow'

        # if any of best lrs are on the edge, perform an expansion step
        expand = False
        for metric in self.target_metrics:
            # get two best log10 lrs for the metric with shift
            best_lrs = list(self.lr_to_value_by_value(metric).keys())[shift:2+shift] # e.g. 0:2
            for lr in best_lrs:

                if lr == self.evaluated_lrs()[0]:
                    if self.expansions < self.max_expansions:
                        self.suggest_lr(lr - 1)
                    else:
                        print(f'EXPANSION LIMIT REACHED {self.lr_to_metrics_by_lr()}')
                    expand = True

                if lr == self.evaluated_lrs()[-1]:
                    if self.expansions < self.max_expansions:
                        self.suggest_lr(lr + 1)
                    else:
                        print(f'EXPANSION LIMIT REACHED {self.lr_to_metrics_by_lr()}')
                    expand = True

        if expand: return 'expand'

        # otherwise binary search - suggest both sides of each best lr
        evaluated_lrs = self.evaluated_lrs()

        # suggest more promising lrs first
        first_lrs = []
        last_lrs = []

        for metric in self.target_metrics:
            # get two best log10 lrs for the metric with shift
            best_lrs = list(self.lr_to_value_by_value(metric).keys())[shift:2+shift] # e.g. 0:2

            for lr in best_lrs:

                # possible todo - when expansion limit reached, test next to edge lr, but its also pretty useless

                left_lr  = evaluated_lrs[evaluated_lrs.index(lr) - 1]
                right_lr = evaluated_lrs[evaluated_lrs.index(lr) + 1]
                left_value = self.lr_to_value_by_lr(metric)[left_lr]
                right_value = self.lr_to_value_by_lr(metric)[right_lr]
                left_distance = abs(left_lr - lr)
                right_distance = abs(right_lr - lr)

                # evaluate further lr first to promote exploration
                # if lrs are spaced equally, evaluate a more promising region (with lower value)
                if (left_distance > right_distance) or (left_distance == right_distance and left_value < right_value):
                    first_lrs.append(lr + (left_lr - lr) / 2)
                    last_lrs.append(lr + (right_lr - lr) / 2)

                if (left_distance < right_distance) or (left_distance == right_distance and left_value > right_value):
                    first_lrs.append(lr + (right_lr - lr) / 2)
                    last_lrs.append(lr + (left_lr - lr) / 2)

        for lr in first_lrs: self.suggest_lr(lr)
        for lr in last_lrs: self.suggest_lr(lr)

        return "binary search"

    def binary_search(self):
        shift = 0

        while True:
            self.suggested_lrs = []

            self.debug('--')
            self.debug(f'{self.binary_search_steps = }; {self.expansions = }')
            for metric in self.target_metrics:
                self.debug(f'current values for {metric}: {self.lr_to_value_by_value(metric)}')
            res = self.binary_search_suggest(shift)
            self.debug(f'"{res}" suggested {self.suggested_lrs}')

            if res == 'overflow': return
            if len(self.suggested_lrs) == 0: shift += 1
            else: shift = 0

            # evaluate suggested log10 lrs
            for lr in self.suggested_lrs:
                if res == 'expand' and self.expansions >= self.max_expansions: continue
                self.evaluate_lr(lr)
                if res == 'binary search': self.binary_search_steps += 1
                if res == 'expand': self.expansions += 1

                # max binary search steps reached
                if self.binary_search_steps >= self.max_binary_search_steps: return


def _only_first_batch_metric(batched_metric, num_batches):
    """
    train loss and num batches are from logger, this takes only 1st train loss from each batch,
    so `L-BFGS` and alike minimizing each batch doesn't give them an advantage.
    """
    is_increase = np.concatenate(([True], num_batches[1:] > num_batches[:-1]))
    modified_values = np.where(is_increase, batched_metric, np.nan)

    filled_series = bn.push(modified_values)
    return filled_series

def _search(
    task_name: str,
    opt_name: str,
    bench: "Benchmark",
    target_metrics: dict[str, bool], # {metric: maximize}; first target metric is targeted by binary search
    optimizer_fn: Callable,
    max_passes: int | None = None,
    max_forwards: int | None = None,
    max_batches: int | None = None,
    max_epochs: int | None = None,
    max_seconds: float | None = None,
    test_every_forwards: int | None = None,
    test_every_batches: int | None = None,
    test_every_epochs: int | None = None,
    test_every_seconds: float | None = None,
    batched: dict[str, bool] | None = None,
    smoothing: dict[str, int | None] | None = None,
    log10_lrs: Sequence[float] | None = (1, 0, -1, -2, -3, -4, -5),
    progress: Literal['full', 'reduced', 'none'] = 'reduced',
    root = 'runs',
    print_achievements = True,

    # lr tuning kwargs
    lr_binary_search_steps = 7, # binary search steps
    max_lr_expansions = 7, # separate count for when best lr is on the edge

    debug=False,
):
    """_summary_

    Args:
        task_name (str): name of the task
        opt_name (str): name of the optimizer
        bench (Benchmark): a benchmark object.
        target_metrics (dict[str, bool]): dictionary `{"metric_name": maximimize}`
        max_passes (int | None, optional): max passes. Defaults to None.
        max_forwards (int | None, optional): max forwards (recommended). Defaults to None.
        max_batches (int | None, optional): max batches. Defaults to None.
        max_epochs (int | None, optional): max epochs. Defaults to None.
        max_seconds (float | None, optional): max seconds. Defaults to None.
        test_every_forwards (int | None, optional): test every forwards. Defaults to None.
        test_every_batches (int | None, optional): test every batches (recommended). Defaults to None.
        test_every_epochs (int | None, optional): test every epochs. Defaults to None.
        test_every_seconds (float | None, optional): test every seconds. Defaults to None.
        batched (dict[str, bool] | None, optional):
            `{"metric_name": batched}`. If metric is batched, takes only first loss from each batch,
            which prevents L-BFGS advantage on minimizing train loss because it minimizes a batch. Defaults to None.
        smoothing (dict[str, int] | None, optional):
            `{"metric_name": smoothing}`. Smoothes the values with gaussian kernel to make batched train metrics
            more accurate. Defaults to None.
        log10_lrs (Sequence[float] | None, optional): list of log10(lr). Defaults to (1, 0, -1, -2, -3, -4, -5).
        progress (str, optional): what to print. Defaults to 'reduced'.
        root (str, optional): runs root. Defaults to 'runs'.
        print_achievements (bool, optional): whether to print when an optimizer set a new record. Defaults to True.
        lr_binary_search_steps (int, optional): max binary search steps. Defaults to 7.
    """
    def _debug(*args,**kwargs):
        if debug: print(*args, **kwargs)

    _debug(f'--- testing {task_name = }; {opt_name = } ---')

    # performance
    bench._make_images = False
    bench._log_params = False
    bench._log_projections = False

    if log10_lrs is None: log10_lrs = (0, ) # optimizers with no lrs

    # create TaskInfo. This creates a new directory for the task
    info = TaskInfo(root=root, task_name=task_name, target_metrics=target_metrics)

    # kwargs for calling `bench.run(**kwargs)`
    kwargs: dict[str, Any] = dict(
        max_passes=max_passes,
        max_forwards=max_forwards,
        max_batches=max_batches,
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        test_every_forwards=test_every_forwards,
        test_every_batches=test_every_batches,
        test_every_epochs=test_every_epochs,
        test_every_seconds=test_every_seconds,
        progress=progress == "full",
    )

    # make optimizer directory
    opt_path = os.path.join(info.path, opt_name)
    if not os.path.exists(opt_path): os.mkdir(opt_path)

    def _test_lr(lr):
        """test lr, save it and return target metrics dict"""

        # if lr already evaluated, load it
        if os.path.exists(os.path.join(opt_path, f'{lr}.npz')):
            logger = DictLogger.from_file(os.path.join(opt_path, f'{lr}.npz'))

        # else evaluate lr and save the logger
        else:
            if progress == 'reduced': print(f'{task_name}: testing {opt_name} {_round_significant(lr, 3)}', end = '                 \r')
            clean_mem()
            bench.reset()
            optimizer = optimizer_fn(bench.parameters(), lr)
            bench.run(optimizer=optimizer,**kwargs)
            _filter_logger_(bench.logger) # filter logger before reporting and saving
            logger = bench.logger

            # postprocess the metric
            for metric, maximize in target_metrics.items():
                vals = None

                # take only 1st value from each batch
                if (batched is not None) and (metric in batched) and batched[metric]:
                    if vals is None: vals = logger.numpy(metric)
                    vals = _only_first_batch_metric(vals, logger.numpy('num batches'))

                # smooth
                if (smoothing is not None) and (metric in smoothing) and (smoothing[metric] is not None):
                    if vals is None: vals = logger.numpy(metric)
                    vals = gaussian_filter1d(vals, smoothing[metric], mode = 'mirror')

                # update metric if necessary
                if vals is not None:
                    bench.logger[f'_ {metric} orig'] = bench.logger[metric]
                    bench.logger.set_array(metric, bench.logger.get_metric_steps(metric), vals)

            info.report(opt_name = opt_name, lr=lr, logger=logger, print_achievements=print_achievements)
            bench.logger.save(os.path.join(opt_path, f'{lr}'))

        # create values dictionary
        values = {}
        for metric, maximize in target_metrics.items():
            if maximize: value = -logger.max(metric) # always minimize
            else: value = logger.min(metric)
            values[metric] = value

        # return values
        return values

    searcher = _BinarySearch(
        fn = _test_lr,
        target_metrics=target_metrics,
        log10_lrs=log10_lrs,
        max_binary_search_steps=lr_binary_search_steps,
        max_expansions=max_lr_expansions,
        rounding = 1,
        debug = debug
    )

    searcher.evaluate_base_lrs()
    if len(log10_lrs) > 1: searcher.binary_search()

    # update info yaml if necessary
    if info._needs_yaml_update: yamlwrite(info.metrics, info.yaml_path)

     # clean empty failed runs
    _clean_empty(info.path)
    _debug()


def _search_for_visualization(
    bench: "Benchmark",
    optimizer_fn: Callable,
    target_metric: str = 'train loss',
    maximize: bool = False,
    max_passes: int | None = None,
    max_forwards: int | None = None,
    max_batches: int | None = None,
    max_epochs: int | None = None,
    max_seconds: float | None = None,
    test_every_forwards: int | None = None,
    test_every_batches: int | None = None,
    test_every_epochs: int | None = None,
    test_every_seconds: float | None = None,
    log10_lrs: Sequence[float] = (5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5),

    # lr tuning kwargs
    lr_binary_search_steps = 11, # binary search steps
    max_lr_expansions = 11, # separate count for when best lr is on the edge

    debug=False,

):
    # performance
    # bench._make_images = False
    # bench._log_params = False
    # bench._log_projections = False

    # kwargs for calling `bench.run(**kwargs)`
    kwargs: dict[str, Any] = dict(
        max_passes=max_passes,
        max_forwards=max_forwards,
        max_batches=max_batches,
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        test_every_forwards=test_every_forwards,
        test_every_batches=test_every_batches,
        test_every_epochs=test_every_epochs,
        test_every_seconds=test_every_seconds,
        progress=False,
    )

    def _evaluate_lr(lr):
        """test lr"""
        bench.reset()
        optimizer = optimizer_fn(bench.parameters(), lr)
        bench.run(optimizer=optimizer,**kwargs)

        # loss area under loss curve up to minimum value, and then minimum value, to penalize erratic paths
        arr = np.nan_to_num(bench.logger.numpy(target_metric), copy=False)
        if maximize: arr = - arr

        argmin = np.argmin(arr)
        min = arr[argmin]

        # loss = (np.sum(arr[:argmin]) + min * (arr.size - argmin)) * (min**(1/3) + 1e-8)
                # area under curve                                  # cubic root of loss

        loss = min + 1e-8 * (np.sum(arr[:argmin]) + min * (arr.size - argmin)) # term for choosing better paths when loss is 0/same

        return {target_metric: loss}

    searcher = _BinarySearch(
        fn = _evaluate_lr,
        target_metrics = {target_metric: maximize},
        log10_lrs=log10_lrs,
        max_binary_search_steps=lr_binary_search_steps,
        max_expansions=max_lr_expansions,
        rounding=1,
        debug = debug,
    )

    searcher.evaluate_base_lrs()
    if len(log10_lrs) > 1: searcher.binary_search()

    best_lr = list(searcher.lr_to_value_by_value(target_metric).keys())[0]
    bench.reset()
    optimizer = optimizer_fn(bench.parameters(), 10 ** best_lr)
    bench.run(optimizer=optimizer,**kwargs)
    bench._info['lr'] = 10 ** best_lr
    return bench