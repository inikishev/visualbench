"""moved some giant methods here"""
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING
import time

import numpy as np
import torch

from .python_tools import format_number

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _remove_prefix(metric: str) -> str:
    """removes train or test prefix if it exists"""
    if metric.startswith('train '): return metric.replace('train ', '')
    if metric.startswith('test '): return metric.replace('test ', '')
    return metric

def _print_progress_(self: "Benchmark"):
    """print progress every second sets _last_print_time"""
    if self._print_interval_s is None: return

    # if one second passed from last print
    t = self._current_time
    assert t is not None

    if t - self._last_print_time > self._print_interval_s:
        text = f'f{self.num_forwards}'
        if self._max_forwards is not None: text = f'{text}/{self._max_forwards}'

        if self.num_passes != 0:
            text = f'{text} p{self.num_passes}'
            if self._max_passes is not None: text = f'{text}/{self._max_passes}'

        if self.num_steps != 0:
            text = f'{text} b{self.num_steps}'
            if self._max_steps is not None: text = f'{text}/{self._max_steps}'

        if self.num_epochs != 0:
            text = f'{text} e{self.num_epochs}'
            if self._max_epochs is not None: text = f'{text}/{self._max_epochs}'

        if self._last_train_loss is not None:
            text = f'{text}; train loss = {format_number(self._last_train_loss, 3)}'

        if self._last_test_loss is not None:
            text = f"{text}; test loss = {format_number(self._last_test_loss, 3)}"

        print(text, end = '                          \r')
        self._last_print_time = t


def _print_final_report(self: "Benchmark"):
    if self.seconds_passed is None: text = "finished in a very short time, reached"
    else: text = f'finished in {self.seconds_passed:.1f}s., reached'

    if 'test loss' in self.logger:
        text = f'{text} train loss = {format_number(self.logger.nanmin("train loss"), 3)},'\
        f' test loss = {format_number(self.logger.nanmin("test loss"), 3)}'

    elif 'train loss' in self.logger:
        text = f'{text} loss = {format_number(self.logger.nanmin("train loss"), 3)}'

    else:
        if self.seconds_passed is None: s = "unknown"
        else: s = f"{self.seconds_passed:.1f}"
        text = f'finished in {s} s., made 0 steps, something is wrong'

    print(f'{text}                                      ')


def _grad_params_to_vec(params: Iterable[torch.Tensor]):
    if isinstance(params, torch.Tensor): raise RuntimeError("got a tensor")
    return torch.cat([p.ravel() for p in params if p.requires_grad])

@torch.no_grad
def _log_params_and_projections_(self: "Benchmark") -> None:
    """conditionally logs parameters and projections if that is enabled, all in one function to reuse parameter_to_vector

    this runs before 1st step, and each train forward pass"""
    if self._is_perturbed or self._performance_mode: return
    param_vec = None

    # --------------------------- log parameter vectors -------------------------- #
    if self._log_params is None:
        if param_vec is None: param_vec = _grad_params_to_vec(self.parameters()).detach()
        if param_vec.numel() < 1000: self._log_params = True
        else: self._log_params = False

    if self._log_params:
        param_vec = _grad_params_to_vec(self.parameters()).detach()
        self.logger.log(self.num_forwards, 'params', param_vec.cpu())

    # ------------------------------ log projections ----------------------------- #
    if self._num_projections != 0:
        if param_vec is None: param_vec = _grad_params_to_vec(self.parameters()).detach()

        # create projections if they are none, one is a bernoulli vector and the other one is the inverse
        if self._basis is None:
            basis_vecs = []
            while len(basis_vecs) < self._num_projections:
                projections = torch.ones((2, param_vec.numel()), dtype = torch.bool, device = param_vec.device)
                projections[0] = torch.bernoulli(
                    projections[0].float(), p = 0.5, generator = self.rng.torch(param_vec.device),
                ).to(dtype = torch.bool, device = param_vec.device)

                projections[1] = ~projections[0]
                basis_vecs.extend(projections.unbind(0))

            self._basis = torch.stack(basis_vecs).to(param_vec)

        # log projections
        projected = self._basis @ param_vec
        self.logger.log(self.num_forwards, 'projected', projected.cpu())

@torch.no_grad
def _update_noise_(self: "Benchmark"):
    if self._param_noise_alpha != 0:
        for i,p in enumerate(self.parameters()):
            if not p.requires_grad: continue
            k = f"_param_noise_{i}"
            v = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=self.rng.torch(p.device)) * self._param_noise_alpha
            if not hasattr(self, k): self.register_buffer(k, v)
            else: getattr(self, k).set_(v)

    if self._grad_noise_alpha != 0:
        for i,p in enumerate(self.parameters()):
            if not p.requires_grad: continue
            k = f"_grad_noise_{i}"
            v = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=self.rng.torch(p.device)) * self._grad_noise_alpha
            if not hasattr(self, k): self.register_buffer(k, v)
            else: getattr(self, k).set_(v)

@torch.no_grad
def _add_param_noise_(self: "Benchmark", sub: bool):
    assert self.training and self._is_perturbed
    if self._param_noise_alpha != 0:
        noise = []
        params = tuple(self.parameters())
        for i,p in enumerate(params):
            if not p.requires_grad: continue
            noise.append(getattr(self, f"_param_noise_{i}"))

        if sub: torch._foreach_sub_(params, noise)
        else: torch._foreach_add_(params, noise)

@torch.no_grad
def _add_grad_noise_(self: "Benchmark"):
    assert self.training
    if self._grad_noise_alpha != 0:
        noise = []
        params = tuple(self.parameters())
        grads = []
        for i,p in enumerate(params):
            if p.grad is not None:
                grads.append(p.grad)
                noise.append(getattr(self, f"_grad_noise_{i}"))

        torch._foreach_add_(grads, noise)

def _ensure_stop_criteria_exists_(self) -> None:
    """warns if no stopping criteria is specified (another one is KeyboardInterrupt)"""
    criteria = {self._max_passes, self._max_forwards, self._max_steps, self._max_epochs, self._max_seconds, self._target_loss}
    if all(i is None for i in criteria): warnings.warn("No stopping criteria specified, benchmark will run forever!")



def _should_stop(self: "Benchmark") -> str | None:
    """returns True if any stopping criteria is satisfied"""
    if (self._max_forwards is not None) and (self.num_forwards >= self._max_forwards): return 'max forwards reached'
    if (self._max_passes is not None) and (self.num_passes >= self._max_passes): return 'max passes reached'
    if (self._max_epochs is not None) and (self.num_epochs >= self._max_epochs): return 'max epochs reached'
    if (self._max_steps is not None) and (self.num_steps >= self._max_steps): return 'max steps reached'
    if (self._max_seconds is not None) and self.seconds_passed is not None and self.seconds_passed >= self._max_seconds:
        if self._print_timeout: print(f'timeout {self.seconds_passed:.2f}s. / {self._max_seconds}s.! Did {self.num_passes}/{self._max_passes} passes')
        return "max time reached"

    if self._target_loss is not None:
        if self._dltest is not None:
            if self._last_test_loss is not None and self._last_test_loss <= self._target_loss: return "target test loss reached"
        else:
            if self._last_train_loss is not None and self._last_train_loss <= self._target_loss: return "target train loss reached"

    return None



def _should_run_test_epoch(self: "Benchmark", check_dltest:bool=True) -> bool:
    """runs after every train closure evaluation, returns True if should test"""
    if check_dltest and self._dltest is None:
        return False

    if (self._last_test_pass is not None) and (self.num_passes == self._last_test_pass):
        return False

    if self.num_forwards == 0:
        return True

    if (self._test_every_forwards is not None) and (self.num_forwards % self._test_every_forwards == 0):
        return True

    if (self._test_every_seconds is not None) and (self._current_time - self._last_test_time >= self._test_every_seconds):
        return True

    # those two can't run after every closure, since we might perform multiple closure evaluations
    # per step and per epoch, that would cause multiple redundant test epochs

    # if (self._test_every_steps is not None) and (self.num_steps % self._test_every_steps == 0): return True
    # if (self._test_every_epochs is not None) and (self.num_epochs % self._test_every_epochs == 0): return True

    return False



@torch.no_grad
def _aggregate_test_metrics_(self: "Benchmark"):
    """Log test metric means into the logger and clear self._test_*_metrics"""
    # mean test scalar metrics
    for k,v in self._test_scalar_metrics.items():
        v_mean = np.mean(v)
        if k == self._trial_report_metric: self._trial_report(v_mean.item())
        self.logger.log(self.num_forwards, k, v_mean)
    self._test_scalar_metrics.clear()

    # other metrics like images
    for k,v in self._test_other_metrics.items():
        self.logger.log(self.num_forwards, k, v)
    self._test_other_metrics.clear()

