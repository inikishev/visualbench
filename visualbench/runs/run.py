import os
import time
import warnings
from collections import UserDict, UserList
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
import random

import msgspec
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ..logger import Logger
from ..utils.format import tonumpy
from ..utils.python_tools import format_number
from . import mbs

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _txtwrite(file: str, text: str | bytes, mode: str):
    with open(file, mode, encoding='utf8' if isinstance(text, str) else None) as f:
        f.write(text)

def _msgpack_decode(file: str, decoder: msgspec.msgpack.Decoder | None = None):
    decode = decoder.decode if decoder is not None else msgspec.msgpack.decode
    with open(file, 'rb') as f:
        return decode(f.read())

def _numel(x: np.ndarray | torch.Tensor):
    if isinstance(x,np.ndarray): return x.size
    return x.numel()

def _get_stats(logger: "Logger") -> dict[str, dict[str, float]]:
    """Extracts stats that are not arrays"""
    stats: dict[str, dict[str, float]] = {}
    for metric, values in logger.items():
        if len(values) == 0: continue
        if isinstance(values[0], (np.ndarray, torch.Tensor)) and _numel(values[0]) != 1: continue

        stats[metric] = {}
        try:
            stats[metric]['min'] = float(logger.nanmin(metric))
            stats[metric]['max'] = float(logger.nanmax(metric))
        except Exception:
            pass

    return stats

def _unpack_path(path: str):
    """Returns root, task name, run name and id"""
    path = os.path.normpath(path)
    path, id = os.path.split(path)
    path, run_name = os.path.split(path)
    root, task_name = os.path.split(path)
    return root, task_name, run_name, id

def _target_metrics_to_dict(metrics:str | Sequence[str] | dict[str, bool]):
    if isinstance(metrics, str): return {metrics: False}
    if isinstance(metrics, Sequence): return {k:False for k in metrics}
    return metrics

def _maybe_format(x):
    if isinstance(x, float): return format_number(x, 3)
    return x

def _dict_to_str(d: dict):
    return ' '.join([f"{k}={_maybe_format(v)}" for k,v in d.items()])

# region Run
class Run:
    """A finished run"""

    def __init__(
        self,
        hyperparams: dict[str, Any],
        logger: "Logger",
        stats: dict[str, dict[str, float]] | None,
        target_metrics: str | Sequence[str] | dict[str, bool],
        id: Any,
    ):
        self.hyperparams = hyperparams
        self.logger = logger
        self.stats = _get_stats(logger) if stats is None else stats
        self.target_metrics = _target_metrics_to_dict(target_metrics)
        self.id = str(time.time_ns()) if id is None else str(id)

        self.root: str | None = None
        self.task_name: str | None = None
        self.run_name: str | None = None
        self.run_path: str | None = None

    def load_logger(self, lazy=True) -> Logger:
        if lazy and len(self.logger) > 0: return self.logger
        if self.run_path is None: raise RuntimeError("trying to load Logger when self.run_path is None")
        self.logger = Logger.from_file(os.path.join(self.run_path, "logger.npz"))
        return self.logger

    def save(self, folder, encoder: msgspec.msgpack.Encoder | None):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)
        if encoder is None: encoder = msgspec.msgpack.Encoder()

        # save logger
        self.logger.save(os.path.join(folder, "logger.npz"))

        # save hyperparameters
        _txtwrite(os.path.join(folder, "hyperparams.msgpack"), encoder.encode(self.hyperparams), 'wb')

        # save stats
        _txtwrite(os.path.join(folder, "stats.msgpack"), encoder.encode(self.stats), 'wb')

        # save target metrics
        _txtwrite(os.path.join(folder, "target_metrics.msgpack"), encoder.encode(self.target_metrics), 'wb')

        self.root, self.task_name, self.run_name, id = _unpack_path(folder)
        assert id == self.id, f"IDs don't match: {id = }, {self.id = }. {type(id) = }, {type(self.id) = }"
        self.run_path = folder

    @classmethod
    def load(cls, folder, load_logger: bool, decoder: msgspec.msgpack.Decoder | None = None):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        id = os.path.basename(folder)
        logger = Logger.from_file(os.path.join(folder, "logger.npz")) if load_logger else Logger()
        hyperparams = _msgpack_decode(os.path.join(folder, "hyperparams.msgpack"), decoder=decoder)
        stats = _msgpack_decode(os.path.join(folder, "stats.msgpack"), decoder=decoder)
        target_metrics = _msgpack_decode(os.path.join(folder, "target_metrics.msgpack"), decoder=decoder)

        run = cls(hyperparams=hyperparams, logger=logger, stats=stats, target_metrics=target_metrics, id=id)
        run.root, run.task_name, run.run_name, id = _unpack_path(folder)
        assert id == run.id
        run.run_path = folder
        return run

    def string(self, metric) -> str:
        if len(self.hyperparams) == 0: s = self.run_name
        else: s = f"{self.run_name} ({_dict_to_str(self.hyperparams)})"
        if s is None: s = "unknown"

        if metric not in self.target_metrics:
            return s

        maximize = self.target_metrics[metric]
        key = 'max' if maximize else 'min'
        return f'{s} {str(format_number(self.stats[metric][key], 5)).ljust(7)}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, Run): raise TypeError(f"Can't check equality because {type(other)} is not a Run!")
        return str(self.id) == str(other.id)
# endregion

# region Sweep
class Sweep(UserList[Run]):
    """List of runs from one sweep"""
    def __init__(self, runs: Iterable[Run]):
        super().__init__(runs)
        self.root: str | None = None
        self.task_name: str | None = None
        self.run_name: str | None = None
        self.sweep_path: str | None = None
        self.target_metrics: dict[str, bool] | None = None

        self._update_paths()

    def _update_paths(self):
        """takes 1st run in self and copies all attributes to self"""
        if len(self) == 0: return
        run1: Run = self.data[0]
        if run1.run_path is not None:
            self.root = run1.root
            self.task_name = run1.task_name
            self.run_name = run1.run_name
            self.sweep_path = os.path.basename(run1.run_path)
            self.target_metrics = run1.target_metrics

    def save(self, folder, encoder: msgspec.msgpack.Encoder | None):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        for run in self:
            run_path = os.path.join(folder, str(run.id))
            os.mkdir(run_path)
            run.save(run_path, encoder=encoder)

        self._update_paths()

    @classmethod
    def load(cls, sweep_path: str, load_loggers: bool, decoder: msgspec.msgpack.Decoder | None):
        if decoder is None: decoder = msgspec.msgpack.Decoder()

        if not os.path.exists(sweep_path):
            raise NotADirectoryError(f"Sweep path \"{sweep_path}\" doesn't exist")

        runs = []
        for id in os.listdir(sweep_path):
            run = Run.load(os.path.join(sweep_path, id), load_logger=load_loggers, decoder=decoder)
            runs.append(run)

        return cls(runs)

    def best_runs(self, metric: str, maximize: bool, n:int):
        k = 'max' if maximize else 'min'
        sorted_runs = sorted(self, key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[:n]
# endregion

# region Task
class Task(UserDict[str, Sweep]):
    """Dictionary of sweeps per optimizer or whatever in a task (sweep name is the key)"""
    def __init__(self, runs: Mapping[str, Sweep]):
        super().__init__(runs)
        self.root: str | None = None
        self.task_name: str | None = None
        self.task_path: str | None = None
        self.target_metrics: dict[str, bool] | None = None

        self._update_paths()

    def _update_paths(self):
        """takes 1st sweep in self and copies all attributes to self"""
        if len(self) == 0: return
        sweep1: Sweep = list(self.values())[0]
        if sweep1.sweep_path is not None:
            self.root = sweep1.root
            self.task_name = sweep1.task_name
            self.task_path = os.path.basename(sweep1.sweep_path)
            self.target_metrics = sweep1.target_metrics

    @classmethod
    def load(cls, task_path: str, load_loggers: bool, decoder: msgspec.msgpack.Decoder | None):
        if decoder is None: decoder = msgspec.msgpack.Decoder()

        if not os.path.exists(task_path):
            raise NotADirectoryError(f"Task path \"{task_path}\" doesn't exist")

        sweeps = {}
        for sweep_name in os.listdir(task_path):
            sweep = Sweep.load(os.path.join(task_path, sweep_name), load_loggers=load_loggers, decoder=decoder)
            sweeps[sweep_name] = sweep

        return cls(sweeps)

    def n_runs(self):
        return sum(len(v) for v in self.values())

    def best_sweeps(self, metric: str, maximize: bool, n:int):
        key = 'max' if maximize else 'min'
        sorted_sweeps = sorted(self.values(), key=lambda s: s.best_runs(metric, maximize, 1)[0].stats[metric][key])
        return sorted_sweeps[:n]

    def best_sweep_runs(self, metric: str, maximize: bool, n:int) -> list[Run]:
        best_run_per_sweep = {}
        for k,v in self.items():
            # new optimizer empty folder is created first
            # then Task is created to find the best values so far
            # so new optimizer folder will be empty
            best_runs = v.best_runs(metric, maximize, 1)
            if len(best_runs) != 0: best_run_per_sweep[k]=best_runs[0]

        k = 'max' if maximize else 'min'
        sorted_runs = sorted(best_run_per_sweep.values(), key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[:n]
# endregion


def _allclose_dict(d1:dict, d2:dict):
    """all keys from d1 must be in d2, but d2 is allowed to have extra keys, unless d1 is empty"""
    if len(d1) == 0 and len(d2) != 0: return False
    for k, v in d1.items():
        if isinstance(v, float):
            if format_number(v, 5) != format_number(d2[k], 5): return False
        else:
            if v != d2[k]: return False
    return True


def _maybe_format_number(x):
    if isinstance(x, (int,float)): return format_number(x, 3)
    return x

# region Search
class Search:
    def __init__(
        self,
        logger_fn: Callable[..., Logger],
        metrics: str | Sequence[str] | dict[str, bool],

        # for printing and saving
        root: str | None = None,
        task_name: str | None = None,
        run_name: str | None = None,
        print_records: bool = False,
        print_progress: bool = False,
        save: bool = False,
        base_hyperparams: dict[str, Any] | None = None,
        pass_base_hyperparams: bool = False,
        load_existing: bool = True,
    ):
        metrics = _target_metrics_to_dict(metrics)
        if base_hyperparams is None: base_hyperparams = {}

        self.logger_fn = logger_fn
        self.target_metrics = metrics
        self.root = root
        self.task_name = task_name
        self.run_name = run_name
        self.print_progress = print_progress
        self.print_records = print_records
        self.save = save
        self.base_hyperparams = base_hyperparams
        self.pass_base_hyperparams = pass_base_hyperparams
        self.load_existing = load_existing

        self.runs = []
        self.encoder = msgspec.msgpack.Encoder()

        # -------------------------- make dirs if save=True -------------------------- #
        self.task_path = self.sweep_path = None
        if save:
            if root is None: raise RuntimeError("save=True but root is None")
            if task_name is None: raise RuntimeError("save=True but task_name is None")
            if run_name is None: raise RuntimeError("save=True but run_name is None")

            if not os.path.exists(root): os.mkdir(root)

            self.task_path = os.path.join(root, task_name)
            if not os.path.exists(self.task_path): os.mkdir(self.task_path)

            self.sweep_path = os.path.join(self.task_path, run_name)
            if not os.path.exists(self.sweep_path): os.mkdir(self.sweep_path)


        # ----------------------- load task stats for printing ----------------------- #
        self.best_metrics: dict[str, tuple[str, float]] | None = None

        if print_records and self.task_path is not None:
            if os.path.exists(self.task_path):
                self.best_metrics = {}
                task = Task.load(self.task_path, load_loggers=False, decoder=None)
                if task.n_runs() > 0:
                    for metric, maximize in metrics.items():
                        run = task.best_sweep_runs(metric, maximize, 1)[0]
                        assert run.run_name is not None
                        if maximize: self.best_metrics[metric] = (run.run_name, run.stats[metric]['max'])
                        else: self.best_metrics[metric] = (run.run_name, run.stats[metric]['min'])
            else:
                if print_records:
                    warnings.warn(f"{self.task_path} doesn't exist")

        # load existing
        self.existing_runs: dict[frozenset[tuple[str, Any]], list] = {}
        if load_existing and self.sweep_path is not None:
            sweep = Sweep.load(self.sweep_path, load_loggers=False, decoder=None)
            for run in sweep:
                self.runs.append(run)
                hyperparams = frozenset(run.hyperparams.items())
                self.existing_runs[hyperparams] = []
                for metric, maximize in metrics.items():
                    if metric in run.stats:
                        stats = run.stats[metric]
                        maximize = metrics[metric]
                        if maximize: self.existing_runs[hyperparams].append(-stats['max'])
                        else: self.existing_runs[hyperparams].append(stats['min'])


    def objective(self, hyperparameters) -> list[float]:
        # - run -
        all_hyperparams = self.base_hyperparams.copy()
        all_hyperparams.update(hyperparameters)

        # - check if hyperparams have already been evaluated -
        for params, metric_values in self.existing_runs.items():

            # extract hyperparameters that are given in all_hyperparams
            existing_hyperparams = {param:value for param,value in params if param in all_hyperparams}
            if _allclose_dict(existing_hyperparams, all_hyperparams):

                # print
                if self.print_progress and random.random() > 0.9:
                    text = f'LOADED {self.run_name} - "{self.task_name}"'
                    if len(hyperparameters) > 0: text = f"{text}: {_maybe_format_number(next(iter(hyperparameters.values())))}"
                    print(f"{text}                      \r", end='')

                return metric_values

        # print
        if self.print_progress:
            text = f'{self.run_name} - "{self.task_name}"'
            if len(hyperparameters) > 0: text = f"{text}: {_maybe_format_number(next(iter(hyperparameters.values())))}"
            print(f"{text}                      \r", end='')

        # run the benchmark
        if self.pass_base_hyperparams: logger = self.logger_fn(**all_hyperparams)
        else: logger = self.logger_fn(**hyperparameters)

        run = Run(all_hyperparams, logger=logger, stats=None, target_metrics=self.target_metrics, id=None)

        # - save -
        if self.save:
            if self.task_path is None or self.run_name is None:
                raise RuntimeError("Save is True but task_path or run_name is not specified")
            if not os.path.exists(self.task_path):
                raise NotADirectoryError(f"task path \"{self.task_path}\" doesn't exist")
            run_path = os.path.join(self.task_path, self.run_name, str(run.id))
            os.mkdir(run_path)
            run.save(run_path, encoder=self.encoder)

        self.runs.append(run)

        # - aggregate target values -
        values = []
        for metric, maximize in self.target_metrics.items():
            if metric not in run.stats:
                raise RuntimeError(f"{metric} is not in stats - {list(logger.keys())}")

            if maximize: values.append(-run.stats[metric]['max'])
            else: values.append(run.stats[metric]['min'])

            # - print if beat record -
            if self.print_records and self.best_metrics is not None and metric in self.best_metrics:
                best_run_name, best_run_value = self.best_metrics[metric]

                if maximize and run.stats[metric]['max'] > best_run_value:
                    print(f'{self.task_name}: {self.run_name} achieved new highest {metric} of '
                          f'{format_number(run.stats[metric]["max"], 3)}, '
                          f'beating {best_run_name} which achieved {format_number(best_run_value, 3)}.')

                    self.best_metrics[metric] = (str(self.run_name), run.stats[metric]["max"])

                if (not maximize) and run.stats[metric]['min'] < best_run_value:
                    print(f'{self.task_name}: {self.run_name} achieved new lowest {metric} of '
                          f'{format_number(run.stats[metric]["min"], 3)}, '
                          f'beating {best_run_name} which achieved {format_number(best_run_value, 3)}.')

                    self.best_metrics[metric] = (str(self.run_name), run.stats[metric]["min"])

        return values
# endregion

def mbs_search(
    logger_fn: Callable[[float], Logger],
    metrics: str | Sequence[str] | dict[str, bool],
    search_hyperparam: str,
    fixed_hyperparams: dict[str, Any] | None,

    # MBS parameters
    log_scale: bool,
    grid: Iterable[float],
    step: float | None,
    num_candidates,
    num_binary,
    num_expansions,
    rounding,

    # for printing and saving
    root: str | None = None,
    task_name: str | None = None,
    run_name: str | None = None,
    print_records: bool = False,
    print_progress: bool = False,
    save: bool = False,
    load_existing: bool = True,
):
    grid = sorted(list(grid))
    if step is None:
        if len(grid) == 1: step = max(abs(grid[0]), 1)
        else: step = abs(grid[0] - grid[1])

    def hparam_fn(**hyperparameters):
        assert len(hyperparameters) == 1
        hyperparam = hyperparameters[search_hyperparam]
        return logger_fn(hyperparam)

    search = Search(
        logger_fn=hparam_fn,
        metrics=metrics,
        root=root,
        task_name=task_name,
        run_name=run_name,
        print_records=print_records,
        print_progress=print_progress,
        save=save,
        base_hyperparams=fixed_hyperparams,
        load_existing=load_existing,
    )

    def objective(x: float):
        return search.objective({search_hyperparam: x})

    mbs.mbs_minimize(
        objective,
        grid=grid,
        step=step,
        num_candidates=num_candidates,
        num_binary=num_binary,
        num_expansions=num_expansions,
        rounding=rounding,
        log_scale=log_scale,
    )

    return Sweep(search.runs)


def single_run(
    logger_fn: Callable[[float], Logger],
    metrics: str | Sequence[str] | dict[str, bool],
    fixed_hyperparams: dict[str, Any] | None,

    # for printing and saving
    root: str | None = None,
    task_name: str | None = None,
    run_name: str | None = None,
    print_records: bool = False,
    print_progress: bool = False,
    save: bool = False,
    load_existing: bool = True,
):
    def hparam_fn(**hyperparameters):
        return logger_fn(0)

    search = Search(
        logger_fn=hparam_fn,
        metrics=metrics,
        root=root,
        task_name=task_name,
        run_name=run_name,
        print_records=print_records,
        print_progress=print_progress,
        save=save,
        base_hyperparams=fixed_hyperparams,
        load_existing=load_existing,
    )

    search.objective({})
    return Sweep(search.runs)

