import itertools
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Callable, Sequence
from typing import Any, Literal, final

import numpy as np
import torch

from . import utils
from .logger import Logger
from .rng import RNG
from .utils import _benchmark_utils, plt_tools, python_tools, torch_tools, _benchmark_plotting, _benchmark_video
from .utils.autograd_counter import AutogradCounter

#class StopCondition(BaseException): pass
class StopCondition(Exception): pass

def _sum_of_squares(x: torch.Tensor):
    return x.pow(2).sum()

class Benchmark(torch.nn.Module, ABC):
    _IS_BENCHMARK = True # for type checking
    num_steps: int

    "same as number of batches"
    def __init__(
        self,
        dltrain: Iterable | None = None,
        dltest: Iterable | None = None,
        param_noise: float = 0,
        grad_noise: float = 0,
        log_params: bool | None = None,
        num_projections: int = 0,
        bounds: tuple[float, float] | None = None,
        make_images: bool = True,
        seed: int | None | RNG = 0,
    ):
        super().__init__()
        self._dltrain: Iterable | None = dltrain
        self._dltest: Iterable | None = dltest
        self._param_noise_alpha: float = param_noise
        self._grad_noise_alpha: float = grad_noise
        self._log_params: bool | None = log_params
        self._num_projections: int = num_projections
        self._seed: int | None | RNG = seed
        self.bounds: tuple[float, float] | None = bounds
        self._make_images: bool = make_images
        self._multiobjective = False
        self._multiobjective_func: Callable | None = None

        self._initial_state_dict = None

        self._reference_images: dict[str, torch.Tensor] = {}
        """images to always include in visualizations"""
        self._image_keys: python_tools.SortedSet[str] = python_tools.SortedSet()
        """keys to display as images"""
        self._image_lowest_keys: python_tools.SortedSet[str] = python_tools.SortedSet()
        """keys to display images corresponding to lowest loss found so far"""
        self._plot_keys: python_tools.SortedSet[str] = python_tools.SortedSet()
        """keys to display line charts for"""

        self._basis: torch.Tensor | None = None
        self._print_interval_s: float | None = 0.1
        self._print_timeout: bool = False
        self._plot_perturbed: bool = False
        self._benchmark_mode: bool = False

        self.reset()

    @torch.no_grad
    def reset(self):
        self.rng: RNG = RNG(self._seed)

        # --------------------------------- trackers --------------------------------- #
        self.num_forwards: int = 0
        self.num_backwards: int = 0
        self.num_extra: float = 0

        self._extra_passes_per_step: float = 0 # anything not counted
        self._post_step_callbacks: "list[Callable[[Benchmark], Any]]" = [] # runs after each optimizer step

        self.num_steps: int = 0
        self.num_epochs: int = 0
        self.start_time: float | None = None
        self._current_time: float = time.time()
        self._last_train_loss: float | None = None
        self._last_test_loss: float | None = None
        self._last_print_time: float = 0
        self.batch = None

        # ------------------------------ stop conditions ----------------------------- #
        self._max_passes: int | None = None
        self._max_forwards: int | None = None
        self._max_steps: int | None = None
        self._max_epochs: int | None = None
        self._max_seconds: float | None = None
        self._target_loss: float | None = None

        # --------------------------- test epoch conditions -------------------------- #
        self._test_every_forwards: int | None = None
        self._test_every_steps: int | None = None
        self._test_every_epochs: int | None = None
        self._test_every_seconds: float | None = None
        self._last_test_time: float = 0


        self.logger = Logger()
        self._test_scalar_metrics = defaultdict(list)
        self._test_other_metrics: dict[str, Any] = {}
        self._previous_images: dict[str, torch.Tensor | np.ndarray] = {} # for logging differences
        self._is_perturbed = False

        # restore original parameters on reset
        if self._initial_state_dict is not None:
            self.load_state_dict(utils.torch_tools.copy_state_dict(self._initial_state_dict, device=self.device), assign=True)

        return self

    @property
    def device(self): return next(iter(self.parameters())).device
    @property
    def dtype(self): return next(iter(self.parameters())).dtype

    @property
    def num_passes(self) -> int:
        return sum([self.num_forwards, self.num_backwards, round(self.num_extra)])

    @property
    def seconds_passed(self):
        if self.start_time is None: return None
        return self._current_time - self.start_time

    @property
    def ndim(self):
        """number of learnable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_noise(self, p: float | None = None, g: float | None = None):
        if p is not None: self._param_noise_alpha = p
        if g is not None: self._grad_noise_alpha = g
        return self

    def set_print_inverval(self, s: float | None = None):
        self._print_interval_s = s
        return self

    def set_log_params(self, enable: bool=True):
        self._log_params = enable
        return self

    def set_num_projections(self, num_projections: int):
        self._num_projections = num_projections
        return self

    def set_plot_perturbed(self, enable: bool = True):
        self._plot_perturbed = enable
        return self

    def set_make_images(self, make_images: bool = False):
        self._make_images = make_images
        return self

    def set_benchmark_mode(self, enable: bool = True):
        self._benchmark_mode = enable
        self._make_images = not enable
        return self

    def set_multiobjective(self, multiobjective: bool = True):
        self._multiobjective = multiobjective
        return self

    def set_multiobjective_func(self, func = _sum_of_squares):
        self._multiobjective_func = func
        return self

    def best_params(self, metric:str = "train loss", maximize:bool=False):
        if maximize: v = self.logger.closest(metric, self.logger.stepmax(metric))
        else: v = self.logger.closest("params", self.logger.stepmin(metric))

        params = [p.detach().clone().cpu() for p in self.parameters()]
        torch.nn.utils.vector_to_parameters(v, params)
        return params

    @torch.no_grad
    def add_reference_image(self, name: str, image, to_uint8: bool, min: float | None = None, max: float | None = None):
        """Add an image to be always displayed, for example the target image for image reconstruction.

        Args:
            name (str): name of the image.
            image (Any): image itself
            to_uint8 (bool, optional):
                if True, image will be normalized and converted to uin8. Otherwise it has to already be in uint8. Defaults to False.
            min (float | None, optional):
                only if to_uint8=True, defines minimal value, if None this is calculated as image.min(). Defaults to None.
            max (float | None, optional):
                only if to_uint8=True, defines minimal value, if None this is calculated as image.max(). Defaults to None.
        """
        if (not to_uint8) and (min is not None or max is not None): raise RuntimeError("min and max are only for to_uint8=True")
        image = utils.format.to_3HW(image)
        if to_uint8: image = utils.format.normalize_to_uint8(image, min=min, max=max)
        elif image.dtype != torch.uint8:
            raise RuntimeError(f"Reference image needs to be in uint8 dtype, or to_uint8 needs to be True, got {image.dtype}")
        self._reference_images[name] = image.cpu()

    @torch.no_grad
    def log(self, metric: str, value: Any, plot: bool = True):
        """
        Log `value` under `metric` key.
        Either "train " or "test " prefix will be added to "metric" automatically unless it is specified manually.

        Note that if value is a scalar (single element tensor, array or python number),
        test value is calculated as mean of values obtained during test epoch.
        Otherwise test value is simply the last value obtained during test epoch.

        Args:
            metric (str): name of the metric
            value (Any): value (anything)
            plot (DisplayType | None, optional): if enabled and if value is a scalar, plots this value on visualizations.
        """
        if self._is_perturbed:
            plot = False
            metric = f'{metric} (perturbed)'

        # note - it is possible to log test metrics in train mode
        # for example when both train and test samples are one large batch passed to model in a single pass
        # thats why it is possible to manually specify metric as "test {name}" for it to be logged as test metric while training
        if plot:
            self._plot_keys.add(_benchmark_utils._remove_prefix(metric))

        if isinstance(value, torch.Tensor): value = value.detach().cpu()
        value = utils.format.maybe_tofloat(value)

        if self.training:
            if not metric.startswith(('train ', 'test ')): metric = f'train {metric}'
            self.logger.log(self.num_forwards, metric, value)

        else:
            if metric.startswith('train'): warnings.warn(f"Logging {metric} in eval() mode (while testing)")
            if not metric.startswith(('train ', 'test ')): metric = f'test {metric}'
            if utils.format.is_scalar(value): self._test_scalar_metrics[metric].append(value)
            else: self._test_other_metrics[metric] = value

    @torch.no_grad
    def log_image(
        self,
        name: str,
        image: np.ndarray | torch.Tensor | Any,
        to_uint8: bool,
        min: float | torch.Tensor | None = None,
        max: float | torch.Tensor | None = None,
        log_difference: bool = False,
        show_best: bool = False,
    ):
        """Log an image which will be displayed in plots and animations, images are only logged in training mode.

        Args:
            name (str): name of the image
            image (np.ndarray | torch.Tensor): image
            to_uint8 (bool): if True, image will be normalized and converted to uin8. Otherwise it has to already be in uint8. Defaults to False.
            min (float | None, optional):
                only if to_uint8=True, defines minimal value, if None this is calculated as image.min(). Defaults to None.
            max (float | None, optional):
                only if to_uint8=True, defines minimal value, if None this is calculated as image.max(). Defaults to None.
            log_difference (bool, optional):
                if True, will also store difference between current and previous image, only while training.
                If parameter noise is enabled, this logs differences between noiseless images.
                Defaults to False.
            show_best (DisplayType | None, optional):
                if enabled, will add a display of the image corresponding to the best loss so far.
        """
        if not self._make_images: warnings.warn(f'logging image {name} with make_images=False')
        if self._benchmark_mode: warnings.warn(f'logging image {name} in BENCHMARK_MODE')
        if self._is_perturbed:
            name = f'{name} (perturbed)'
            log_difference=False; show_best=False

        self._image_keys.add(name)
        if show_best: self._image_lowest_keys.add(name)

        if not to_uint8:
            if image.dtype not in (np.uint8, torch.uint8):
                raise RuntimeError(f"image needs to be in uint8 dtype, or to_uint8 needs to be True, got {image.dtype}")
        if (not to_uint8) and (min is not None or max is not None): raise RuntimeError("min and max are only for to_uint8=True")

        if isinstance(image, torch.Tensor): image = image.detach().cpu().clone()

        # difference
        k = difference = None
        if log_difference:
            k = f'{name} (difference)'

            if name not in self._previous_images: difference = image
            else: difference = self._previous_images[name] - image

            self._previous_images[name] = image
            if to_uint8: difference = utils.format.normalize_to_uint8(difference)

        # value
        if to_uint8:
            image = utils.format.normalize_to_uint8(image, min=min, max=max)

        self.logger.log(self.num_forwards, name, image)

        # log difference after image so that order is better
        if (k is not None) and (difference is not None):
            self._image_keys.add(k)
            self.logger.log(self.num_forwards, k, difference)

    def pre_step(self):
        pass

    @abstractmethod
    def get_loss(self) -> torch.Tensor:
        """"""

    @final
    @torch.no_grad
    def forward(self) -> torch.Tensor:
        # store initial state dict on 1st step
        # this also gets called at the beginning of run to make time more accurate
        # but this is for when function is evaluated manually
        if self._initial_state_dict is None:
            _benchmark_utils._update_noise_(self)
            self._initial_state_dict = torch_tools.copy_state_dict(self.state_dict(), device='cpu')

        # terminate if stop condition reached
        if self.training:
            msg = _benchmark_utils._should_stop(self)
            if msg is not None: raise StopCondition(msg)
            if self._is_perturbed: _benchmark_utils._add_param_noise_(self, sub=False)

        # get loss and log it
        with torch.enable_grad():
            ret = self.get_loss()
            if ret.numel() > 1:
                if self._multiobjective_func is None:
                    raise RuntimeError(f"{self.__class__.__name__} returned multiple values but multiobjective "
                                       "function is not set. Add `self.set_multiobjective_func` to `__init__`.")
                loss = self._multiobjective_func(ret)
            else: loss = ret

        cpu_loss = utils.format.tofloat(loss)
        self.log('loss', cpu_loss)

        if self._is_perturbed:
            _benchmark_utils._add_param_noise_(self, sub=True)
            if self._multiobjective: return ret
            return loss

        if self.training:
            self._last_train_loss = cpu_loss

            # start timer right after forward pass before 3rd optimizer step to let things compile and warm up.
            # plus it runs the 1st test epoch
            if self.num_forwards == 2:
                self.start_time = time.time()

        else:
            self._last_test_loss = cpu_loss

        if self._multiobjective: return ret
        return loss

    @torch.no_grad
    def post_closure(self, backward: bool) -> None:
        """post closure logic that must be called by closure, but that way any custom closure can be used like
        gauss newton one as long as it calls this before returning whatever it returns."""
        if backward:
            _benchmark_utils._add_grad_noise_(self)

        if self._is_perturbed:
            if backward: self.num_backwards += 1 # num_forwards incremented by closure(False)
            self._is_perturbed = False
            self.closure(False)
            self._is_perturbed = True
            return

        self._current_time = time.time()
        if self.training:
            # log params (conditons are in the method)
            _benchmark_utils._log_params_and_projections_(self)

            self.logger.log(self.num_forwards, "seconds", self.seconds_passed if self.seconds_passed is not None else 0)
            self.logger.log(self.num_forwards, "num passes", self.num_passes)
            self.logger.log(self.num_forwards, "num batches", self.num_steps)

            # this runs before first num forwards is incremented
            # so it usually on 1st step as 0%x = 0
            # this happens after backward so there are .grad attributes already
            # so the only way this could cause issues is if forward pass calcualtes and uses gradients wrt parameters
            if _benchmark_utils._should_run_test_epoch(self): self.test_epoch()

            # increments
            self.num_forwards += 1
            if backward: self.num_backwards += 1

        if self._print_interval_s is not None: _benchmark_utils._print_progress_(self)

    def closure(self, backward=True, retain_graph=None, create_graph=False) -> torch.Tensor:

        if backward:
            self.zero_grad()
            loss = self.forward()
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        else:
            loss = self.forward()

        self.post_closure(backward)

        return loss

    def get_x0(self):
        return torch.nn.utils.parameters_to_vector(p for p in self.parameters() if p.requires_grad)

    def loss_at(self, x: Any):
        xt = utils.totensor(x, device=self.device, dtype=self.dtype)
        torch.nn.utils.vector_to_parameters(xt, (p for p in self.parameters() if p.requires_grad))
        return utils.tofloat(self.closure(backward=False))

    def loss_grad_at(self, x:Any):
        xt = utils.totensor(x, device=self.device, dtype=self.dtype)
        torch.nn.utils.vector_to_parameters(xt, (p for p in self.parameters() if p.requires_grad))
        loss = utils.tofloat(self.closure(backward=True))
        grad = torch.cat(
            [p.grad.ravel() if p.grad is not None else torch.zeros_like(p) for p in self.parameters() if p.requires_grad]
        )
        if isinstance(x, torch.Tensor): return loss, grad.to(x)
        return loss, utils.tonumpy(grad)

    def one_step(self, optimizer):
        """one batch or one step"""
        _benchmark_utils._update_noise_(self)
        self.pre_step()

        if self.training:
            if self._param_noise_alpha != 0: self._is_perturbed = True
            else: self._is_perturbed = False

            optimizer.step(self.closure)
            self.num_steps += 1
            self.num_extra += self._extra_passes_per_step
            for cb in self._post_step_callbacks: cb(self)
            self._is_perturbed = False

        else:
            self._is_perturbed = False
            self.closure(False)

    def train_epoch(self, optimizer):
        if self._dltrain is None: self.one_step(optimizer)
        else:
            for batch in self._dltrain:
                self.batch = batch
                self.one_step(optimizer)

        if self.training:
            self.num_epochs += 1

    def test_epoch(self):
        assert self._dltest is not None
        self.eval()
        batch_backup = self.batch

        for batch in self._dltest:
            self.batch = batch
            self.one_step(optimizer=None)

        self._last_test_time = time.time()
        self.batch = batch_backup
        self.train()
        _benchmark_utils._aggregate_test_metrics_(self) # this needs to be called after .train because log checks if training


    def run(
        self,
        optimizer: torch.optim.Optimizer,
        max_passes: int | None = None,
        max_forwards: int | None = None,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        max_seconds: float | None = None,
        test_every_forwards: int | None = None,
        test_every_batches: int | None = None,
        test_every_epochs: int | None = None,
        test_every_seconds: float | None = None,
        target_loss: int | None = None,

        # stuff
        num_extra_passes: float | Callable[[int], float] = 0,
        step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,
    ):
        self._max_passes = max_passes; self._max_forwards = max_forwards
        self._max_steps = max_steps; self._max_epochs = max_epochs
        self._max_seconds = max_seconds; self._target_loss = target_loss
        self._extra_passes_per_step = num_extra_passes if isinstance(num_extra_passes, (int,float)) else num_extra_passes(self.ndim)
        _benchmark_utils._ensure_stop_criteria_exists_(self)

        if callable(step_callbacks): step_callbacks = [step_callbacks, ]
        if step_callbacks is None: step_callbacks = []
        self._post_step_callbacks = list(step_callbacks)

        self._test_every_forwards = test_every_forwards; self._test_every_steps = test_every_batches
        self._test_every_epochs = test_every_epochs; self._test_every_seconds = test_every_seconds

        # make sure to store initial state dict
        if self._initial_state_dict is None:
            _benchmark_utils._update_noise_(self)
            self._initial_state_dict = torch_tools.copy_state_dict(self.state_dict(), device='cpu')

        self.train()

        try:
            for _ in range(max_epochs) if max_epochs is not None else itertools.count():
                self.train_epoch(optimizer)

        except (StopCondition, KeyboardInterrupt):
            if self._dltest is not None: self.test_epoch()
            if self._print_interval_s: _benchmark_utils._print_final_report(self)

        else:
            if self._dltest is not None: self.test_epoch()
            if self._print_interval_s: _benchmark_utils._print_final_report(self)

        return self

    def plot_loss(self, ylim: Literal['auto'] | tuple[float,float] | None = 'auto',
                  yscale=None, smoothing: float | tuple[float,float,float] = 0, ax=None):
        train_loss = test_loss = train_loss_perturbed = None

        train_loss = self.logger['train loss'] if 'train loss' in self.logger else None
        test_loss = self.logger['test loss'] if 'test loss' in self.logger else None
        if self._plot_perturbed and "train loss (perturbed)" in self.logger:
            train_loss_perturbed = self.logger["train loss (perturbed)"]

        losses = {"train loss": train_loss, "test loss": test_loss, "train loss (perturbed)": train_loss_perturbed}

        plt_tools.plot_loss(losses, ylim=ylim, yscale=yscale, smoothing=smoothing, ax=ax)

    def plot_summary(
        self: "Benchmark",
        ylim: tuple[float, float] | Literal["auto"] | None = "auto",
        yscale=None,
        smoothing: float | tuple[float, float, float] = 0,
        axsize: float | tuple[float, float] | None = (8, 4),
        dpi: float | None = None,
        fig=None,
    ):
        _benchmark_plotting.plot_summary(self, ylim=ylim, yscale=yscale, smoothing=smoothing, axsize=axsize, dpi=dpi, fig=fig)

    def render(self, file: str, fps: int = 60, scale: int | float = 1, progress=True):
        _benchmark_video._render(self, file, fps=fps, scale=scale, progress=progress)


