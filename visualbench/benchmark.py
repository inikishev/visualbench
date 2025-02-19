import itertools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import Any, Literal, Unpack, final

import numpy as np
import torch
from myai.logger import DictLogger
from myai.plt_tools import Fig
from myai.plt_tools._types import _K_Line2D
from myai.rng import RNG
from myai.torch_tools import copy_state_dict

from ._utils import (
    _aggregate_test_metrics_,
    _check_image,
    _check_stop_condition,
    _check_test_epoch_condition,
    _ensure_float,
    _ensure_stop_condition_exists_,
    _log_params_and_projections_,
    _make_float_hw3_tensor,
    _make_float_tensor,
    _maybe_detach_clone,
    _normalize_to_uint8,
    _plot_images,
    _plot_loss,
    _plot_trajectory,
    _print_final_report,
    _print_progress,
    _render_video,
    _search,
    plot_lr_search_curve,
    plot_metric,
)


class StopCondition(Exception): pass



class Benchmark(torch.nn.Module, ABC):
    def __init__(
        self,
        dltrain: Sequence[Any] | None = None,
        dltest: Sequence[Any] | None = None,
        log_params = False,
        log_projections = False,
        seed: int | None = 0,
    ):
        super().__init__()
        self.reference_images: dict[str, np.ndarray[Any, np.dtype[np.uint8]] | torch.IntTensor] = {}
        self.display_best_keys: list[str] = []
        """list of keys to logger images that will be shown twice - best value and last value"""

        self._dltrain: Sequence[Any] | None = dltrain
        self._dltest: Sequence[Any] | None = dltest
        self._log_params: bool = log_params
        self._log_projections: bool = log_projections
        self._seed: int | None = seed
        self._print_progress = True
        self._make_images = False
        self._initial_state_dict = None
        self._reset()

    def _store_initial_state_dict(self):
        """gets called before run or before first function evaluation depending on what is called first, saves deep copy of state dict on cpu"""
        self._initial_state_dict = copy_state_dict(self, device='cpu')

    def _restore_initial_state_dict(self):
        if self._initial_state_dict is None: raise RuntimeError("_initial_state_dict is None")
        self.load_state_dict(copy_state_dict(self._initial_state_dict))

    def reset(self):
        """resets this benchmark to initial state, may be faster than creating new benchmark (this needs to be implemented by benchmarks)"""
        self._reset()
        return self

    def _reset(self):
        self.logger = DictLogger()
        self.rng = RNG(self._seed)

        self._start_time: float | None = None
        self._cur_time: float | None = None

        self._num_forwards: int = 0
        self._num_backwards: int = 0
        self._num_batches: int = 0
        self._num_epochs: int = 0
        self._info = {}

        self._max_forwards: int | None = None
        self._max_passes: int | None = None
        self._max_batches: int | None = None
        self._max_epochs: int | None = None
        self._max_seconds: float | None = None

        self._test_every_forwards: int | None = None
        self._test_every_batches: int | None = None
        self._test_every_epochs: int | None = None
        self._test_every_seconds: float | None = None
        self._last_test_time: float = 0
        self._last_print_time: float = 0
        self._last_train_loss: float | None = None
        self._last_test_loss: float | None = None
        self._test_metrics: dict[str, list[Any]] = {}
        self._previous_difference_values: dict[str, Any] = {}

        self._proj1: torch.Tensor | None = None
        self._proj2: torch.Tensor | None = None

        if self._initial_state_dict is not None: self._restore_initial_state_dict()
        self.train()

    @property
    def _status(self) -> Literal['train','test']: return 'train' if self.training else 'test'
    @property
    def _num_passes(self) -> int: return self._num_forwards + self._num_backwards
    @property
    def _time_passed(self) -> float:
        if (self._start_time is None) or (self._cur_time is None): return 0.
        return self._cur_time - self._start_time

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def parameters(self, recurse=True):
        for n, p in self.named_parameters(recurse=recurse):
            if not n.startswith('_ignore_'): yield p

    def log(self, name: str, value: Any, log_test: bool, to_uint8 = False):
        """log something, it automatically gets detached and moved to cpu

        Args:
            name (str): name of metric
            value (Any): value
            log_test (bool, optional):
                if true something like 'accuracy' becomes 'train accuracy' or 'test accuracy'
                depending on if training or testing, and takes mean of test values after each test epoch.
                Otherwise only logged during training. Defaults to True.
        """
        if isinstance(value, torch.Tensor): value = value.detach().cpu()
        if to_uint8: value = _normalize_to_uint8(value)

        # for consistency images should be in uint8 format and I check them to avoid stupid issues
        if name.startswith('image'): value = _check_image(value, name)

        # logged during training
        if self.training:
            if log_test: name = f"{self._status} {name}"
            self.logger.log(self._num_forwards, name, value)

        # if logged during testing, save to test metrics for aggregation
        else:
            if not log_test: return
            if name in self._test_metrics: self._test_metrics[name].append(value)
            else: self._test_metrics[name] = [value]

    def log_difference(self, name: str, value: Any, to_uint8):
        """basically saves last update to name to visualzie optimziation dynamics, train only"""
        if name not in self._previous_difference_values:
            prev = self._previous_difference_values[name] = _maybe_detach_clone(value)
        else:
            prev = self._previous_difference_values[name]

        self.log(name, value - prev, log_test=False, to_uint8=to_uint8)

    @abstractmethod
    def get_loss(self) -> torch.Tensor:
        """returns loss, if needed batch can be accessed via self.batch (please make sure to move it to self.device),
        can log stuff with self.log. No need to log train and test losses, they are logged automatically
        """

    @final
    def forward(self) -> torch.Tensor:
        """get_loss + logs params, loss and checks stop conditions"""
        # store initial state dict on 1st step
        # this also gets called at the beginning of run to make time more accurate
        # but this is for when function is evaluated manually
        if self._initial_state_dict is None: self._store_initial_state_dict()

        # terminate if stop condition reached
        if self.training:
            msg = _check_stop_condition(self)
            if msg is not None: raise StopCondition(msg)

        # get loss and log it
        loss = self.get_loss()

        cpu_loss = _ensure_float(loss)
        self.log('loss', cpu_loss, log_test=True)

        if self.training:
            self._last_train_loss = cpu_loss

            # start timer right after forward pass before 3rd optimizer step to let things compile and warn up.
            # plus it runs the 1st test epoch
            if self._num_forwards == 2:
                self._start_time = time.time()

        else:
            self._last_test_loss = cpu_loss

        return loss

    def _post_closure(self, backward: bool) -> None:
        """post closure logic that must be called by closure, but that way any custom closure can be used like
        gauss newton one as long as it calls this before returning whatever it returns."""
        self._cur_time = time.time()
        if self.training:
            # log params (conditons are in the method)
            _log_params_and_projections_(self)

            self.log("time", self._time_passed, log_test=False)
            self.log("num passes", self._num_passes, log_test=False)

            # this runs before first num forwards is incremented
            # so it usually on 1st step as 0%x = 0
            # this happens after backward so there are .grad attributes already
            # so the only way this could cause issues is if forward pass calcualtes and uses gradients wrt parameters
            if _check_test_epoch_condition(self): self._test_epoch()

            # increments
            self._num_forwards += 1
            if backward: self._num_backwards += 1

        if self._print_progress: _print_progress(self)

    def closure(self, backward = True):
        loss = self.forward()

        if backward:
            self.zero_grad()
            loss.backward()

        self._post_closure(backward)

        return loss

    def one_step(self, optimizer, batch):
        """one batch or one step"""
        self.batch = batch

        if self.training:
            optimizer.step(self.closure)
            self._num_batches += 1
        else:
            self.closure(False)


    def one_epoch(self, optimizer, dl):
        """one epoch"""
        if dl is None: self.one_step(optimizer, None)
        else:
            for batch in dl: self.one_step(optimizer, batch)

        if self.training: self._num_epochs += 1
        else: _aggregate_test_metrics_(self)

    @torch.inference_mode()
    def _test_epoch(self):
        """this runs after potential backward inside optimizer step
        optimizer might perform another backward on same batch
        so this stores current batch and restores it"""
        self.eval()
        batch_backup = self.batch
        self.one_epoch(None, self._dltest)
        self._last_test_time = time.time()
        self.batch = batch_backup
        self.train()


    def run(
        self,
        optimizer,
        max_passes: int | None = None,
        max_forwards: int | None = None,
        max_batches: int | None = None,
        max_epochs: int | None = None,
        max_seconds: float | None = None,
        test_every_forwards: int | None = None,
        test_every_batches: int | None = None,
        test_every_epochs: int | None = None,
        test_every_seconds: float | None = None,
        progress = True
    ):
        if self._initial_state_dict is None: self._store_initial_state_dict()

        self._max_forwards = max_forwards
        self._max_passes = max_passes
        self._max_batches = max_batches
        self._max_epochs = max_epochs
        self._max_seconds = max_seconds
        self._test_every_forwards = test_every_forwards
        self._test_every_batches = test_every_batches
        self._test_every_epochs = test_every_epochs
        self._test_every_seconds = test_every_seconds
        self._print_progress = progress
        _ensure_stop_condition_exists_(self)

        self.train()

        try:
            for _ in range(max_epochs) if max_epochs is not None else itertools.count():
                self.one_epoch(optimizer, self._dltrain)

        except StopCondition: pass
        finally:
            if self._dltest is not None: self.one_epoch(None, self._dltest)
            if self._print_progress: _print_final_report(self)

        return self

    def add_reference_image(self, key: str, image:np.ndarray | torch.Tensor, to_uint8 = True):
        if to_uint8: image = _normalize_to_uint8(image)
        self.reference_images[key] = image # type:ignore

    def set_display_best(self, key: str, display_best: bool = True):
        if key in self.display_best_keys:
            if display_best: return
            self.display_best_keys.remove(key)
        else:
            if display_best: self.display_best_keys.append(key)

    def plot_loss(self, ylim: Literal['auto'] | Sequence[float] | None = 'auto', yscale = None, x = 'num passes', y = 'loss', fig=None, show=True, **kw: Unpack[_K_Line2D]):
        spec = locals().copy()
        spec.update(spec.pop('kw'))
        del spec['self']
        return _plot_loss(self, **spec)

    def plot_trajectory(self, fig = None, norm: str | None = 'symlog', show = True):
        spec = locals().copy()
        del spec['self']
        return _plot_trajectory(self, **spec)

    def plot_images(self, fig=None, show=True):
        spec = locals().copy()
        del spec['self']
        return _plot_images(self, **spec)

    def plot_summary(self, metrics = (), nrows = None, ncols = None, figsize = None, axsize = None):
        fig = Fig()

        # plot losses
        self.plot_loss(fig=fig, show=False)

        # plot any other metrics
        if isinstance(metrics, str): metrics = (metrics, )
        for metric in metrics: self.plot_loss(y=metric, fig=fig.add(metric), show=False)

        # plot trajectory if it was logged
        if self._proj1 is not None: self.plot_trajectory(fig.add('trajectory'), show=False)

        # check if any images to plot and plot them
        if any(i.startswith(('image', 'train image', 'test image')) for i in self.logger.keys()) or len(self.reference_images) != 0:
            self.plot_images(fig.add(), show = False)

        # show
        if (axsize is None) and (figsize is None): axsize = (8, 4)
        fig.show(nrows=nrows, ncols=ncols, figsize=figsize, axsize=axsize)

    def render_video(self, file: str, fps: int = 60, scale: int | float = 1, progress=True):
        spec = locals().copy()
        del spec['self']
        return _render_video(self, **spec)

    def search(
        self,
        task_name: str,
        opt_name: str,
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
        lr_binary_search_steps = 2, # binary search steps
        max_lr_expansions = 5, # separate count for when best lr is on the edge
        plot=False,
    ):
        # performance settings
        self._log_params = False
        self._log_projections = False
        self._make_images = False

        spec = locals().copy()
        spec['bench'] = spec.pop('self')
        del spec['plot']
        _search(**spec)

        if plot:
            fig = Fig()
            for metric in target_metrics.keys():
                plot_metric(task_name = task_name, metric = metric, opts = opt_name, root=root, fig=fig.add(metric), show=False)
            plot_lr_search_curve(task_name = task_name, opts = opt_name, root=root, fig = fig.add('lrs'), show=False)
            fig.show(axsize = (12, 6))

