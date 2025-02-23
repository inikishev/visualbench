import os
import shutil
from collections import UserDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal
import optuna
import numpy as np
import torch
from glio.jupyter_tools import clean_mem
from myai.loaders.yaml import yamlread, yamlwrite
from myai.logger import DictLogger
from myai.plt_tools import Fig
from myai.python_tools import split_path

from .utils import _ensure_float, _round_significant

if TYPE_CHECKING:
    from ..benchmark import Benchmark

REFERENCE_OPTS = ("SGD", "AdamW", "PSGDKron", "Muon", "PrecondSchedulePaLMSOAP", "MARS-AdamW")

_DEFAULT_DICT = lambda: {
    "min": {"value": float('inf'),  "run": '',},
    "max": {"value": -float('inf'), "run": '',},
    "first": 'none',
    "maximize": None,
}


class Metric(UserDict):
    @property
    def maximize(self) -> bool | None: return self['maximize']
    @maximize.setter
    def maximize(self, v:bool): self['maximize'] = v

    @property
    def first(self) -> float | None:
        if self['first'] == 'none': return None
        return self['first']
    @first.setter
    def first(self, v:float): self['first'] = _ensure_float(v)

    @property
    def min_value(self) -> float: return self['min']['value']
    @min_value.setter
    def min_value(self, v): self['min']['value'] = _ensure_float(v)

    @property
    def min_run(self) -> str: return self['min']['run']
    @min_run.setter
    def min_run(self, v:str): self['min']['run'] = v

    @property
    def max_value(self) -> float: return self['max']['value']
    @max_value.setter
    def max_value(self, v): self['max']['value'] = _ensure_float(v)

    @property
    def max_run(self) -> str: return self['max']['run']
    @max_run.setter
    def max_run(self, v): self['max']['run'] = v

    def best_run(self) -> str:
        if self.maximize: return self.max_run
        return self.min_run

def _update_metrics_(metrics: dict[str, Metric], logger: DictLogger, run_path: str, print_achievements = True):
    """updates metrics in place with new min and max values if they have been achieved, returns whether an update has been made"""
    needs_update = False

    for metric in logger.keys():
        if metric in ('time', 'num passes'): continue
        if metric not in metrics: metrics[metric] = Metric(_DEFAULT_DICT())
        m: Metric = metrics[metric]

        if m.first is None:
            m.first = logger.first(metric)
            needs_update = True

        # compare with current min and max values
        min = logger.min(metric)
        if np.isfinite(min) and (min < m.min_value or m.min_run == ''):
            filename, opt_name, task_name , *_= list(reversed(split_path(run_path)))
            run_name = os.path.join(opt_name, filename)

            if print_achievements and (m.maximize is not None) and (not m.maximize):
                print(f'{task_name}: {run_name} achieved new lowest {metric} of {_round_significant(min, 3)}, '
                      f'beating {m.min_run} which achieved {_round_significant(m.min_value, 3)}.')

            m.min_value = min
            m.min_run = run_name
            needs_update = True


        max = logger.max(metric)
        if np.isfinite(max) and (max > m.max_value or m.max_run == ''):
            filename, opt_name, task_name , *_= list(reversed(split_path(run_path)))
            run_name = os.path.join(opt_name, filename)


            if print_achievements and (m.maximize is not None) and m.maximize:
                print(f'{task_name}: {run_name} achieved new highest {metric} of {_round_significant(max, 3)}, '
                      f'beating {m.max_run} which achieved{_round_significant(m.max_value, 3)}.')

            m.max_value = max
            m.max_run = run_name
            needs_update = True

    return needs_update




class TaskInfo:
    def __init__(self, root, task_name, target_metrics:dict[str,bool] | None = None):
        self.root = root
        if not os.path.exists(root): os.mkdir(root)

        self.task_name = task_name
        self.path = os.path.join(root, task_name)
        if not os.path.exists(self.path): os.mkdir(self.path)

        self.yaml_path = os.path.join(self.path, 'info.yaml')
        self._needs_yaml_update = False

        if not os.path.exists(self.yaml_path):
            self._needs_yaml_update = True
            self.metrics = {}
            yamlwrite(self.metrics, self.yaml_path)

        else:
            self.metrics = {k: Metric(v) for k,v in yamlread(self.yaml_path).items()}

        if target_metrics is not None:
            self.target_metrics = target_metrics
            for metric, maximize in target_metrics.items():
                if metric not in self.metrics: self.metrics[metric] = Metric(_DEFAULT_DICT())
                self.metrics[metric].maximize = maximize
        else:
            self.target_metrics = {k: v.maximize for k,v in self.metrics.items()}

    def best_logger(self, metric=None) -> DictLogger:
        """loads and returns best logger"""
        if metric is None: metric = list(self.target_metrics.keys())[0]
        return DictLogger.from_file(self.metrics[metric].best_run())

    def best_opt(self, metric=None) -> str:
        """returns name of best optimizer"""
        if metric is None: metric = list(self.target_metrics.keys())[0]
        return os.path.dirname(self.metrics[metric].best_run())

    def report(self, opt_name: str, lr: float, logger: DictLogger, print_achievements=True):
        run_path = os.path.join(self.path, opt_name, f'{lr}.npz')
        needs_update = _update_metrics_(metrics=self.metrics, logger=logger, run_path=run_path, print_achievements=print_achievements)
        if needs_update: self._needs_yaml_update = True

    def rebuild_yaml(self):
        """scans folder and remakes yaml.info from scratch"""
        metrics: dict[str, Metric] = {}

        # get minimize/maximize info from target metrics
        for metric, maximize in self.target_metrics.items():
            if metric not in self.metrics: self.metrics[metric] = Metric(_DEFAULT_DICT())
            if maximize is not None: self.metrics[metric].maximize = maximize

        # go through all loggers and update metrics
        for opt in os.listdir(self.path):
            if opt == 'info.yaml': continue
            opt_path = os.path.join(self.path, opt)

            for run in os.listdir(opt_path):
                run_path = os.path.join(opt_path, run)
                logger = DictLogger.from_file(run_path)

                _update_metrics_(metrics=metrics, logger=logger, run_path=run_path, print_achievements=False)

        self.metrics = metrics
        yamlwrite(self.metrics, self.yaml_path)
        self._needs_yaml_update = False

def rebuild_all_yamls_(root = 'runs'):
    for task_name in os.listdir(root):
        TaskInfo(root=root, task_name=task_name).rebuild_yaml()

def _numel(x):
    if isinstance(x, np.ndarray): return x.size
    if isinstance(x, torch.Tensor): return x.numel()
    raise TypeError(type(x))


def _filter_logger_(logger: DictLogger):
    """removes empty keys and keys whose values have more than 1 element (i.e. images)"""
    for k in list(logger.keys()):
        if len(logger[k]) == 0: del logger[k]
        first = logger.first(k)
        if isinstance(first, (np.ndarray, torch.Tensor)) and _numel(first) > 1: del logger[k]

def _rremove_(l, value):
    ind = len(l) - 1 - l[::-1].index(value)
    del l[ind]

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
    lrs10: Sequence[float] | None = (1, 0, -1, -2, -3, -4, -5),
    progress: Literal['full', 'reduced', 'none'] = 'reduced',
    root = 'runs',
    print_achievements = True,

    # lr tuning kwargs
    lr_binary_search_steps = 7, # binary search steps
    max_lr_expansions = 7, # separate count for when best lr is on the edge
):
    if lrs10 is None: lrs10 = (0, ) # optimizers with no lrs

    info = TaskInfo(root=root, task_name=task_name, target_metrics=target_metrics)
    kwargs: dict[str,Any] = dict(max_passes=max_passes,max_forwards=max_forwards,max_batches=max_batches,max_epochs=max_epochs,max_seconds=max_seconds,test_every_forwards=test_every_forwards,test_every_batches=test_every_batches,test_every_epochs=test_every_epochs,test_every_seconds=test_every_seconds,progress=progress=='full')

    # make optimizer directory
    opt_path = os.path.join(info.path, opt_name)
    # if not os.path.exists(opt_path): raise FileExistsError(f"{opt_path} already exists")
    # os.mkdir(opt_path)
    if not os.path.exists(opt_path): os.mkdir(opt_path)


    # stage 1: test all lrs
    metric = list(target_metrics.keys())[0]
    maximize = target_metrics[metric]
    evaluated_lrs10_by_value: list[tuple[float,float]] = []

    def _test_lr10(lr10):
        lr = 10 ** lr10
        if os.path.exists(os.path.join(opt_path, f'{lr}.npz')):
            logger = DictLogger.from_file(os.path.join(opt_path, f'{lr}.npz'))
        else:
            if progress == 'reduced': print(f'{task_name}: testing {opt_name} {_round_significant(lr, 3)}', end = '                 \r')
            clean_mem()
            bench.reset()
            optimizer = optimizer_fn(bench.parameters(), lr)
            bench.run(optimizer=optimizer,**kwargs)
            _filter_logger_(bench.logger) # filter logger before reporting and saving
            logger = bench.logger
            info.report(opt_name = opt_name, lr=lr, logger=logger, print_achievements=print_achievements)
            bench.logger.save(os.path.join(opt_path, f'{lr}'))

        if maximize: value = -logger.max(metric) # always minimize
        else: value = logger.min(metric)

        evaluated_lrs10_by_value.append((lr10, value))
        return value

    for lr10 in lrs10: _test_lr10(lr10)

    # stage2 - binary search
    shift = 0
    while True:
        if len(lrs10) == 1: break

        evaluated_lrs10_by_value.sort(key = lambda x: x[1])
        evaluated_lrs10_by_lr = sorted(evaluated_lrs10_by_value, key = lambda x: x[0])

        # pick two best lrs
        if 2+shift > len(evaluated_lrs10_by_value): break
        best_lrs10 = evaluated_lrs10_by_value[shift:2+shift]

        lrs10_to_eval = []
        # determine next lrs via either binary search or expansions

        for lr10,val in best_lrs10:
            # check if on edge, if yes break to skip evaluating other lr
            if lr10 == evaluated_lrs10_by_lr[0][0]:
                lrs10_to_eval.append(evaluated_lrs10_by_lr[0][0] - 1)
                break
            elif lr10 == evaluated_lrs10_by_lr[-1][0]:
                lrs10_to_eval.append(evaluated_lrs10_by_lr[-1][0] + 1)
                break

            # else binary search, add both sides
            else:
                left_lr10 = evaluated_lrs10_by_lr[evaluated_lrs10_by_lr.index((lr10, val)) - 1][0]
                lrs10_to_eval.append(lr10 + (left_lr10 - lr10) / 2)

                right_lr10 = evaluated_lrs10_by_lr[evaluated_lrs10_by_lr.index((lr10, val)) + 1][0]
                lrs10_to_eval.append(lr10 + (right_lr10 - lr10) / 2)

        # print()
        # print('lrs', evaluated_lrs10_by_value)
        # print('suggested1', lrs10_to_eval)

        # remove lrs that are too similar within list
        rounded_lrs10_to_eval = [_round_significant(lr10, 1) for lr10 in lrs10_to_eval]
        for lr10 in lrs10_to_eval.copy():
            if rounded_lrs10_to_eval.count(_round_significant(lr10, 1)) > 1:
                _rremove_(lrs10_to_eval, lr10)
                rounded_lrs10_to_eval.remove(_round_significant(lr10, 1))
                assert _round_significant(lr10, 1) in rounded_lrs10_to_eval

        # remove lrs that are too similar to evaluated ones
        rounded_lrs10 = [_round_significant(lr10, 1) for lr10,v in evaluated_lrs10_by_lr]
        for lr10 in lrs10_to_eval.copy():
            if _round_significant(lr10,1) in rounded_lrs10:
                _rremove_(lrs10_to_eval, lr10)

        if len(lrs10_to_eval) == 0:
            shift += 1

        # print('suggested2', lrs10_to_eval)
        # print()

        # evaluate new lrs
        terminate = False
        for lr10 in lrs10_to_eval:
            shift = 0
            if lr10 < evaluated_lrs10_by_lr[0][0] or lr10 > evaluated_lrs10_by_lr[-1][0]: max_lr_expansions -= 1
            else: lr_binary_search_steps -= 1

            _test_lr10(lr10)
            if max_lr_expansions == 0 or lr_binary_search_steps == 0:
                terminate = True
                break

        if terminate: break

    # update info yaml if necessary
    if info._needs_yaml_update: yamlwrite(info.metrics, info.yaml_path)

    # clean empty failed runs
    for dir in os.listdir(info.path):
        if dir == 'info.yaml': continue
        opt_path = os.path.join(info.path, dir)
        if len(os.listdir(opt_path)) == 0:
            shutil.rmtree(opt_path)


