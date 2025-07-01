import os
import time
import warnings
from collections import UserDict, UserList
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ..logger import Logger
from ..utils.format import tonumpy
from ..utils.python_tools import round_significant
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

class Run:
    """A finished run"""

    def __init__(
        self,
        hyperparams: dict[str, Any],
        logger: "Logger",
        stats: dict[str, dict[str, float]] | None,
        id: int|str|None,
    ):
        self.hyperparams = hyperparams
        self.logger = logger
        self.stats = _get_stats(logger) if stats is None else stats
        self.id = time.time_ns() if id is None else id

        self.root: str | None = None
        self.task_name: str | None = None
        self.run_name: str | None = None
        self.run_path: str | None = None

    def save(self, folder):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        # save logger
        self.logger.save(os.path.join(folder, "logger.npz"))

        # save hyperparameters
        _txtwrite(os.path.join(folder, "hyperparams.msgpack"), msgspec.msgpack.encode(self.hyperparams), 'wb')

        # save stats
        _txtwrite(os.path.join(folder, "stats.msgpack"), msgspec.msgpack.encode(self.stats), 'wb')

        self.root, self.task_name, self.run_name, id = _unpack_path(folder)
        assert id == self.id
        self.run_path = folder

    @classmethod
    def load(cls, folder, load_logger: bool, decoder: msgspec.msgpack.Decoder | None = None):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        id = os.path.basename(folder)
        logger = Logger.from_file(os.path.join(folder, "logger.npz")) if load_logger else Logger()
        hyperparams = _msgpack_decode(os.path.join(folder, "hyperparams.msgpack"), decoder=decoder)
        stats = _msgpack_decode(os.path.join(folder, "stats.msgpack"), decoder=decoder)

        run = cls(hyperparams=hyperparams, logger=logger, stats=stats, id=id)
        run.root, run.task_name, run.run_name, id = _unpack_path(folder)
        assert id == run.id
        run.run_path = folder
        return run


class Sweep(UserList[Run]):
    """List of runs from one sweep"""
    def __init__(self, runs: Iterable[Run]):
        super().__init__(runs)
        self.root: str | None = None
        self.task_name: str | None = None
        self.run_name: str | None = None
        self.sweep_path: str | None = None

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

    def save(self, folder):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        for run in self:
            run_path = os.path.join(folder, str(run.id))
            os.mkdir(run_path)
            run.save(run_path)

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

    def best_run(self, metric: str, maximize: bool):
        k = 'max' if maximize else 'min'
        sorted_runs = sorted(self, key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[0]

class Task(UserDict[str, Sweep]):
    """Dictionary of sweeps per optimizer or whatever in a task (run name is the key)"""
    def __init__(self, runs: Mapping[str, Sweep]):
        super().__init__(runs)
        self.root: str | None = None
        self.task_name: str | None = None
        self.task_path: str | None = None

        sweep1: Sweep = list(self.values())[0]
        if sweep1.sweep_path is not None:
            self.root = sweep1.root
            self.task_name = sweep1.task_name
            self.task_path = os.path.basename(sweep1.sweep_path)

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

    def best_run(self, metric: str, maximize: bool):
        best_runs = {k: v.best_run(metric, maximize) for k,v in self.items()}
        k = 'max' if maximize else 'min'
        sorted_runs = sorted(best_runs.values(), key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[0]



class Search:
    def __init__(
        self,
        logger_fn: Callable[..., Logger],
        targets: str | Sequence[str] | dict[str, bool],

        # for printing and saving
        root: str | None = None,
        task_name: str | None = None,
        run_name: str | None = None,
        print_records: bool = False,
        save: bool = False,
        base_hyperparams: dict[str, Any] | None = None,
        pass_base_hyperparams: bool = False,
    ):
        if isinstance(targets, str): targets = {targets: False}
        if isinstance(targets, Sequence): targets = {k:False for k in targets}
        if base_hyperparams is None: base_hyperparams = {}

        self.logger_fn = logger_fn
        self.targets = targets
        self.root = root
        self.task_name = task_name
        self.run_name = run_name
        self.print_records = print_records
        self.save = save
        self.base_hyperparams = base_hyperparams
        self.pass_base_hyperparams = pass_base_hyperparams

        self.runs = []

        # -------------------------- make dirs if save=True -------------------------- #
        self.task_path = self.sweep_path = None
        if save:
            if root is None: raise RuntimeError("save=True but root is None")
            if task_name is None: raise RuntimeError("save=True but task_name is None")
            if run_name is None: raise RuntimeError("save=True but run_name is None")

            if not os.path.exists(root): os.mkdir(root)

            self.task_path = os.path.join(root, task_name)
            if not os.path.exists(self.task_path): os.mkdir(self.task_path)

            self.sweep_path = os.path.join(root, run_name)
            if not os.path.exists(self.sweep_path): os.mkdir(self.sweep_path)


        # ----------------------- load task stats for printing ----------------------- #
        self.best_metrics: dict[str, tuple[str, float]] | None = None

        if print_records and self.task_path is not None:
            if os.path.exists(self.task_path):
                self.best_metrics = {}
                task = Task.load(self.task_path, load_loggers=False, decoder=None)
                if len(task) > 0:
                    for target, maximize in targets.items():
                        run = task.best_run(target, maximize)
                        assert run.run_name is not None
                        if maximize: self.best_metrics[target] = (run.run_name, run.stats['metric']['max'])
                        else: self.best_metrics[target] = (run.run_name, run.stats['metric']['min'])
            else:
                if print_records:
                    warnings.warn(f"{self.task_path} doesn't exist")


    def objective(self, hyperparameters) -> list[float]:
        # - run -
        all_hyperparams = self.base_hyperparams.copy()
        all_hyperparams.update(hyperparameters)

        if self.pass_base_hyperparams: logger = self.logger_fn(**all_hyperparams)
        else: logger = self.logger_fn(**hyperparameters)

        run = Run(all_hyperparams, logger=logger, stats=None, id=None)

        # - save -
        if self.save:
            if self.task_path is None or self.run_name is None:
                raise RuntimeError("Save is True but task_path or run_name is not specified")
            if not os.path.exists(self.task_path):
                raise NotADirectoryError(f"task path \"{self.task_path}\" doesn't exist")
            run.save(os.path.join(self.task_path, self.run_name, str(run.id)))

        self.runs.append(run)

        # - aggregate target values -
        values = []
        for target, maximize in self.targets.items():
            if target not in run.stats:
                raise RuntimeError(f"{target} is not in stats - {list(logger.keys())}")

            if maximize: values.append(-run.stats[target]['max'])
            else: values.append(run.stats[target]['min'])

            # - print if beat record -
            if self.print_records and self.best_metrics is not None:
                best_run_name, best_run_value = self.best_metrics[target]

                if maximize and run.stats[target]['max'] > best_run_value:
                    print(f'{self.task_name}: {self.run_name} achieved new highest {target} of '
                          f'{round_significant(run.stats[target]["max"], 3, True)}, '
                          f'beating {best_run_name} which achieved {round_significant(best_run_value, 3, True)}.')

                    self.best_metrics[target] = (str(self.run_name), run.stats[target]["max"])

                if (not maximize) and run.stats[target]['min'] < best_run_value:
                    print(f'{self.task_name}: {self.run_name} achieved new lowest {target} of '
                          f'{round_significant(run.stats[target]["min"], 3, True)}, '
                          f'beating {best_run_name} which achieved {round_significant(best_run_value, 3, True)}.')

                    self.best_metrics[target] = (str(self.run_name), run.stats[target]["min"])

        return values


def mbs_search(
    logger_fn: Callable[[float], Logger],
    targets: str | Sequence[str] | dict[str, bool],
    search_hyperparam: str,
    fixed_hyperparams: dict[str, Any] | None,

    # MBS parameters
    log_scale: bool,
    grid: Iterable[float],
    step: float,
    num_candidates,
    num_binary,
    num_expansions,
    rounding,

    # for printing and saving
    root: str | None = None,
    task_name: str | None = None,
    run_name: str | None = None,
    print_records: bool = False,
    save: bool = False,
):

    def hparam_fn(**hyperparameters):
        assert len(hyperparameters) == 1
        hyperparam = hyperparameters[search_hyperparam]
        return logger_fn(hyperparam)

    search = Search(
        logger_fn=hparam_fn,
        targets=targets,
        root=root,
        task_name=task_name,
        run_name=run_name,
        print_records=print_records,
        save=save,
        base_hyperparams=fixed_hyperparams,
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
    targets: str | Sequence[str] | dict[str, bool],
    fixed_hyperparams: dict[str, Any] | None,

    # for printing and saving
    root: str | None = None,
    task_name: str | None = None,
    run_name: str | None = None,
    print_records: bool = False,
    save: bool = False,
):
    def hparam_fn(**hyperparameters):
        return logger_fn(0)

    search = Search(
        logger_fn=hparam_fn,
        targets=targets,
        root=root,
        task_name=task_name,
        run_name=run_name,
        print_records=print_records,
        save=save,
        base_hyperparams=fixed_hyperparams
    )

    search.objective({})
    return Sweep(search.runs)