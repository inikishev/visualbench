import os
import time
from collections import UserDict, UserList
from collections.abc import Callable, Sequence, Iterable
from typing import TYPE_CHECKING, Any
from warnings import warn

import msgspec
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ..utils.format import tonumpy
from ..logger import Logger
from . import mbs

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _txtwrite(file: str, text: str | bytes, mode: str):
    with open(file, mode, encoding='utf8') as f:
        f.write(text)

def _msgpack_decode(file: str, decoder: msgspec.msgpack.Decoder | None = None):
    decode = decoder.decode if decoder is not None else msgspec.msgpack.decode
    with open(file, 'rb', encoding='utf8') as f:
        return decode(f.read())

def _numel(x: np.ndarray | torch.Tensor):
    if isinstance(x,np.ndarray): return x.size
    return x.numel()

def _get_stats(logger: "Logger") -> dict[str, dict[str, float]]:
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


    def save(self, folder):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        # save logger
        self.logger.save(os.path.join(folder, "logger.npz"))

        # save hyperparameters
        _txtwrite(os.path.join(folder, "hyperparams.msgpack"), msgspec.msgpack.encode(self.hyperparams), 'wb')

        # save stats
        _txtwrite(os.path.join(folder, "stats.msgpack"), msgspec.msgpack.encode(self.stats), 'wb')


    @classmethod
    def load(cls, folder, load_logger: bool, decoder: msgspec.msgpack.Decoder | None = None):
        if not os.path.isdir(folder): raise NotADirectoryError(folder)

        id = os.path.basename(folder)
        logger = Logger.from_file(os.path.join(folder, "logger.npz")) if load_logger else Logger()
        hyperparams = _msgpack_decode(os.path.join(folder, "hyperparams.msgpack"), decoder=decoder)
        stats = _msgpack_decode(os.path.join(folder, "stats.msgpack"), decoder=decoder)

        return cls(hyperparams=hyperparams, logger=logger, stats=stats, id=id)


class Sweep(UserList[Run]):
    """List of runs from one sweep"""
    def __init__(self, run_folder: str, load_loggers: bool, decoder: msgspec.msgpack.Decoder | None):
        super().__init__()
        self.folder = run_folder
        self.run_name = os.path.basename(self.folder)
        self.task_name = os.path.dirname(self.folder)

        if os.path.exists(self.folder):
            if decoder is None: decoder = msgspec.msgpack.Decoder()
            for id in os.listdir(self.folder):
                run = Run.load(os.path.join(self.folder, id), load_logger=load_loggers, decoder=decoder)
                self.append(run)

    def get_by_hyperparams(self, **hyperparams):
        for run in self:
            if run.hyperparams == hyperparams:
                return run
        return None

    def best_run(self, metric: str, maximize: bool):
        k = 'max' if maximize else 'min'
        sorted_runs = sorted(self, key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[0]

class Task(UserDict[str, Sweep]):
    """Dictionary of sweeps per whatever"""
    def __init__(self, task_folder: str, load_loggers: bool):
        super().__init__()
        self.folder = task_folder
        self.task_name = os.path.basename(self.folder)
        self.root = os.path.dirname(self.folder)

        if os.path.exists(self.folder):
            decoder = msgspec.msgpack.Decoder()
            for sweep_name in os.listdir(self.folder):
                sweep = Sweep(os.path.join(self.folder, sweep_name), load_loggers=load_loggers, decoder=decoder)
                self[sweep_name] = sweep

    def best_run(self, metric: str, maximize: bool):
        best_runs = {k: v.best_run(metric, maximize) for k,v in self.items()}
        k = 'max' if maximize else 'min'
        sorted_runs = sorted(self, key=lambda run: run.stats[metric][k], reverse=maximize)
        return sorted_runs[0]


def univariate_search(
    logger_fn: Callable[[float], Logger],
    hyperparam_name: str,
    targets: str | Sequence[str] | dict[str, bool],
    grid: Iterable[float] | Any,
    step: float,
    log_scale: bool,
    root: str | None,
    task_name: str | None,
    run_name: str | None,
    num_candidates: int = 2,
    num_binary: int = 7,
    num_expansions: int = 7,
    rounding: int = 2,
) -> list[Run]:
    grid = mbs._tofloatlist(tonumpy(grid))
    if isinstance(targets, str): targets = {targets: False}
    if isinstance(targets, Sequence): targets = {k:False for k in targets}

    runs: list[Run] = []

    def objective(x:float):
        if log_scale: x = 10**x

        logger = logger_fn(x)
        run = Run(hyperparams={hyperparam_name: x}, logger=logger, stats=None, id=None)
        runs.append(run)

        values = []
        for target, maximize in targets.items():
            if target not in run.stats:
                raise RuntimeError(f"{target} is not in stats - {list(logger.keys())}")

            if maximize: values.append(-run.stats[target]['max'])
            else: values.append(run.stats[target]['min'])

        return values

    mbs.minimize(objective, grid=grid, step=step, num_candidates=num_candidates,
                 num_binary=num_binary, num_expansions=num_expansions, rounding=rounding)

    return runs

