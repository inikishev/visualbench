import operator
import os
import typing
from typing import Any
from collections import UserList
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial

import numpy as np
from myai.loaders.yaml import yamlread
from myai.logger import BaseLogger, Comparison, DictLogger
from myai.plt_tools import Fig
from myai.torch_tools import maybe_ensure_pynumber
from myai.python_tools import get__name__

class Run:
    def __init__(self, logger: BaseLogger, hyperparams: dict, attrs: dict, path: str):
        self.logger = logger
        self.hyperparams = hyperparams
        self.attrs = attrs
        self.path = path
        self.dirname = os.path.basename(self.path)
        self.name = self.attrs['name']

    @classmethod
    def from_dir(cls, dir: str):
        logger = DictLogger.from_file(os.path.join(dir, "logger.npz"))
        hyperparams = yamlread(os.path.join(dir, 'hyperparams.yaml'))
        attrs = yamlread(os.path.join(dir, 'attrs.yaml'))
        name = os.path.basename(dir)
        return cls(logger=logger, hyperparams=hyperparams, attrs=attrs, path=dir)


def _dict_slash_index(d: dict, key: str) -> typing.Any:
    """can access nested dict by `/`."""
    path = key.split('/')
    for k in path:
        if k in d: d = d[k]
        else: d = 'None' # type:ignore
    return d

def _logger_slash_index(l: BaseLogger, key: str) -> typing.Any:
    path = key.split('/')
    if path[0] != 'logger': raise ValueError(key)
    metric = path[1]
    method = path[2]
    return maybe_ensure_pynumber(operator.methodcaller(method, metric)(l))

def _run_slash_index(run: Run, key: str) -> typing.Any:
    if key.startswith('logger/'): return _logger_slash_index(run.logger, key)
    if key.startswith('hyperparams/'): return _dict_slash_index(run.hyperparams, key[len('hyperparams/'):])
    if key.startswith('attrs/'): return _dict_slash_index(run.attrs, key[len('attrs/'):])
    if key == 'name': return run.name
    if key == 'dirname': return run.dirname
    return _dict_slash_index(run.hyperparams, key)

class Summary(UserList[Run]):
    def __init__(self, runs: Iterable[Run]):
        super().__init__(runs)

    @property
    def loggers(self):
        return [run.logger for run in self]

    def print_names(self):
        print('\n'.join(r.name for r in self))

    def to_polars(self, *cols: str | Callable[[Run], Any]):
        import polars as pl
        data = {}
        for col in cols:
            if isinstance(col, str): data[col] = self.get_key_list(col)
            elif callable(col):
                data[get__name__(col)] = [col(r) for r in self]
            else: raise TypeError(type(col))
        return pl.DataFrame(data)

    def get_key_list(self, attr: str):
        return [_run_slash_index(r, attr) for r in self]


    @property
    def comparison(self):
        return Comparison({run.name:run.logger for run in self})

    @classmethod
    def from_runs_dir(cls, dir: str):
        return cls([Run.from_dir(os.path.join(dir, r)) for r in os.listdir(dir) if os.path.isdir(os.path.join(dir, r))])

    def sorted_by_metric(self, metric: str, method: typing.Literal['min', 'max', 'last'], reverse = False):
        caller = operator.methodcaller(method, metric)
        return Summary(sorted(self, key = lambda x:caller(x.logger), reverse=reverse))

    def sorted_by_attr(self, attr: str, reverse = False):
        return self.sorted(lambda x: _run_slash_index(x, attr), reverse=reverse)

    def n_best(self, metric: str, n: int, method: typing.Literal['min', 'max', 'last'], highest: bool):
        return Summary(self.sorted_by_metric(metric, method, reverse = highest)[:n])

    def filter(self, filt: Callable[[Run], bool] | Mapping):
        if callable(filt): return Summary([run for run in self if filt(run)])
        return Summary([run for run in self if all(_run_slash_index(run, k) == v for k, v in filt.items())])

    def sorted(self, fn: Callable[[Run], Any] | str, reverse = False):
        if isinstance(fn, str): fn = partial(_run_slash_index, key = fn)
        return Summary(sorted(self, key = fn, reverse = reverse))

    def map_name(self, *names: Callable | str, sep = ' '):
        """valid names:
            - logger
            - hyperparams
            - attrs
            - name
            - dirname

        anything else is searched in hyperparams
        """
        runs = []
        for run in self:
            run_name = []
            for name in names:
                if callable(name):
                    run_name.append(name(run))
                elif isinstance(name, str):
                    if name.startswith('{') and name.endswith('}'): run_name.append(name[1:-1])
                    else: run_name.append(_run_slash_index(run, name))
                else: raise TypeError(f'cannot map name with type {type(name)}')

            new_attrs = run.attrs.copy()
            new_attrs['name'] = sep.join([str(i) for i in run_name])
            runs.append(Run(run.logger, run.hyperparams, new_attrs, run.path))

        return Summary(runs)

    def plot_grid(
        self,
        hyperparam: str,
        metric: str,
        method: typing.Literal["min", "max", "last"],
        fig=None,
        show=True,
        xlim=None,
        ylim=None,
        log = False,
        **kwargs,
    ):
        caller = operator.methodcaller(method, metric)
        summary = self.sorted_by_attr(hyperparam)
        x = [_dict_slash_index(run.hyperparams, hyperparam) for run in summary]
        y = [caller(run.logger) for run in summary]

        if fig is None: fig = Fig()

        fig.linechart(x, y, **kwargs).axlabels(hyperparam, f'{method} {metric}').ticks().grid()
        if xlim: fig.xlim(xlim)
        if ylim: fig.ylim(ylim)
        if log: fig.xscale('log', base=10)
        if show: fig.show()
        return fig