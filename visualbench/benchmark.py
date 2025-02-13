import itertools
import os
import time
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from functools import partial
from typing import Any, TypedDict, Unpack, final, Literal
import warnings

import numpy as np
import torch
from myai.loaders.text import txtwrite
from myai.loaders.yaml import yamlwrite
# I temporarily import it from myai. I need to move this into this lib later
from myai.logger import DictLogger
from myai.plt_tools import Fig, imshow_grid
from myai.python_tools import Composable, Progress
from myai.python_tools import SaveSignature as sig
from myai.python_tools import (get__name__, make_dict_serializeable,
                               maybe_compose)
from myai.rng import RNG
from myai.torch_tools import maybe_ensure_pynumber, pad_to_shape
from myai.transforms import force_hw3, totensor, tonumpy
from myai.video import OpenCVRenderer
from torch.nn.utils import parameters_to_vector

# those are public exports
from .dataset_tools import make_dataset, make_dataset_from_tensor
from .utils import CUDA_IF_AVAILABLE


class _ClosureKwargs(TypedDict, total = False):
    backward: bool
    zero_grad: bool
    retain_graph: bool | None
    create_graph: bool
    enable_grad: bool

def _set_closure_defaults_(kwargs: _ClosureKwargs) -> _ClosureKwargs:
    kwargs.setdefault("backward", True)
    kwargs.setdefault("zero_grad", True)
    kwargs.setdefault('retain_graph', None)
    kwargs.setdefault('create_graph', False)
    kwargs.setdefault('enable_grad', True)
    return kwargs

_TEST_KWARGS: _ClosureKwargs = {'enable_grad': False}

class StopCondition(Exception): pass

class Benchmark(torch.nn.Module):
    def __init__(
        self,
        train_data: Any = None,
        test_data: Any = None,
        train_batch_tfms: Composable | None = None,
        test_batch_tfms: Composable | None | Literal['same'] = None,
        log_params=False,
        log_projections = True,
        save_edge_params=False,
        reference_images: np.ndarray[Any, np.dtype[np.uint8]] | torch.Tensor | Sequence[np.ndarray[Any, np.dtype[np.uint8]] | torch.Tensor] | None = None,
        reference_labels: str | Sequence[str] | None = None,
        # device: torch.types.Device = CUDA_IF_AVAILABLE,
        seed: int | RNG | None = 0,
    ):
        """A benchmark.

        Args:
            train_data (Any, optional):
                Dataloader iterable. For full batch training just pass entire dataset in a length one tuple.
                None if this is not a dataset task. Defaults to None.
            test_data (Any, optional):
                test data. For full batch training just pass entire dataset in a length one tuple.
                None if this is not a dataset task or if this task has no test dataset. Defaults to None.
            train_batch_tmfs (Composable | None, optional):
                transforms to apply to train batches. Defaults to None.
            test_batch_tfms (Composable | None, optional):
                transforms to apply to test batches. Defaults to None.
            log_params (str, optional):
                saves parameter vectors on each step, as well as best parameters so far. Defaults to False.
            log_projections (str, optional):
                saves parameters projected into 2 dimensions on each step. Defaults to True.
            save_edge_params (str, optional):
                stores three copies of params - first, middle, and best.
                only works when `max_steps` is set in `run`. Defaults to False.
            reference_images (str, optional):
                optional reference image in `np.uint8`, will be plotted next to solution.
                can be multiple images. Defaults to None.
            reference_labels (str, optional): labels for reference images. Defaults to None.
            device (str, optional): device. Defaults to 'cuda'.
            seed (int | RNG | None, optional):
                    integer seed, RNG object or None for random seed. Used for random parameter projections. Defaults to 0.
        """
        super().__init__()
        self.seed = seed
        self.reset()

        self.train_data = train_data
        self.test_data = test_data
        self.batch: Any = None

        self.log_params = log_params
        self.log_projections = log_projections
        self.save_edge_params = save_edge_params

        if (reference_images is not None) and isinstance(reference_images, (np.ndarray, torch.Tensor)): reference_images = [reference_images]
        self.reference_images: Sequence[np.ndarray[Any, np.dtype[np.uint8]] | torch.Tensor] | None = reference_images

        if self.reference_images is not None:
            if (reference_labels is not None) and isinstance(reference_labels, str): reference_labels = [reference_labels] * len(self.reference_images)
            if reference_labels is None: reference_labels = ['reference'] * len(self.reference_images)
            if len(reference_labels) != len(self.reference_images):
                raise ValueError(f'reference_labels must be the same length as reference_images, got f{len(reference_labels) = } and {len(self.reference_images) = }')
        self.reference_labels: Sequence[str] | None = reference_labels

        self.train_batch_tfms = maybe_compose(train_batch_tfms)
        if test_batch_tfms == 'same': test_batch_tfms = train_batch_tfms
        self.test_batch_tfms = maybe_compose(test_batch_tfms)

        self._projections = None



    def reset(self, *args, **kwargs):
        """reset to without having to reinitialize dataset good for fast hyperparam testing"""
        self.zero_grad()
        self.start_time = None
        self.time_passed = 0
        self.logger = DictLogger()
        self.hyperparams = {}
        self.name: str | None = None

        self.best_params: torch.Tensor | None = None
        self.initial_params: torch.Tensor | None = None
        self.middle_params: torch.Tensor | None = None

        self.lowest_loss = float('inf')

        self.current_step = 0
        self.num_backwards = 0
        self.current_batch = 0
        self.current_epoch = 0

        self._max_steps = None
        self._max_passes = None
        self._max_epochs = None
        self._max_batches = None
        self._max_time = None
        self._min_loss = None
        self._status = 'init'
        self._last_print_time = 0
        self._last_test_time = 0
        self._last_train_loss = None
        self._last_test_loss = None
        self._test_metrics = {}
        self._test_every_steps = None
        self._test_every_sec = None
        self._improved = False

        self.rng = RNG(self.seed)

    @property
    def device(self):
        return self._first_param_device()

    @property
    def num_passes(self):
        """forward passes + backward passes"""
        return self.current_step + self.num_backwards

    def _first_param_device(self):
        return next(iter(self.parameters())).device

    @torch.no_grad
    def log(self, metric, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()

        if self._status == 'train':
            self.logger.log(self.num_passes, metric, value)

        else:
            if metric in self._test_metrics: self._test_metrics[metric].append(value)
            else: self._test_metrics[metric] = [value]

    def _aggregate_test_metrics(self):
        """Logs the mean of each test metrics and resets test metrics to an empty dict"""
        for metric, values in self._test_metrics.items():
            self.logger.log(self.current_step, metric, np.nanmean(values))
        self._test_metrics = {}

    def _print_progress(self, t: float):
        """print progress every second"""
        # if one second passed from last print

        if t - self._last_print_time > 1:
            text = f's{self.current_step}'
            if self._max_steps is not None: text = f'{text}/{self._max_steps}'
            if self._max_passes is not None:
                text = f'{text} p{self.num_passes}'
            if self.current_batch != 0:
                text = f'{text} b{self.current_batch}'
                if self._max_batches is not None: text = f'{text}/{self._max_batches}'
            if self.current_epoch != 0:
                text = f'{text} e{self.current_epoch}'
                if self._max_epochs is not None: text = f'{text}/{self._max_epochs}'

            if self._last_train_loss is not None and self._status == 'train':
                text = f'{text}; train loss = {self._last_train_loss.item():.3f}'
            if self._last_test_loss is not None:
                text = f"{text}; test loss = {self._last_test_loss.item():.3f}"

            print(text, end = '                 \r')
            self._last_print_time = t

    @torch.no_grad
    def _log_params(self):
        """log two parameter projections, also stores params"""
        param_vec = None

        if self.log_projections:
            param_vec = parameters_to_vector(self.parameters()).detach()
            if self._projections is None:
                self._projections = torch.ones((2, param_vec.numel()), dtype = torch.bool, device = param_vec.device)
                self._projections[0] = torch.bernoulli(
                    self._projections[0].float(),
                    p = 0.5,
                    generator = self.rng.torch(param_vec.device),
                ).to(dtype = torch.bool, device = param_vec.device)
                self._projections[1] = ~self._projections[0]

            self.log('proj1', (param_vec * self._projections[0]).mean().detach().cpu())
            self.log('proj2', (param_vec * self._projections[1]).mean().detach().cpu())

        # save parameter vectors
        if self.log_params:
            if param_vec is None: param_vec = parameters_to_vector(self.parameters())
            self.log('params', param_vec.detach().cpu())

        if self.save_edge_params:
            if param_vec is None: param_vec = parameters_to_vector(self.parameters()).detach().cpu()

            if self._improved:
                self.best_params = param_vec

            if self.middle_params is None:
                if (self._max_batches is not None) and (self.current_step == self._max_batches // 2):
                    self.middle_params = param_vec
                elif (self._max_steps is not None) and (self.current_step == self._max_steps // 2):
                    self.middle_params = param_vec
                elif (self._max_passes is not None) and (self.num_passes == self._max_passes // 2):
                    self.middle_params = param_vec

    def _check_stop_condition(self):
        """terminates training via raising StopCondition when any condition is met"""
        if (self._max_steps is not None) and self.current_step >= self._max_steps:
            raise StopCondition('max steps reached')
        if (self._max_passes is not None) and self.num_passes >= self._max_passes:
            raise StopCondition('max passes reached')
        if (self._max_epochs is not None) and self.current_epoch >= self._max_epochs:
            raise StopCondition('max epochs reached')
        if (self._max_batches is not None) and self.current_batch >= self._max_batches:
            raise StopCondition('max batches reached')
        if (self._min_loss is not None) and (self._last_train_loss is not None) and self._last_train_loss <= self._min_loss:
            raise StopCondition('min loss reached')
        if (self._max_time is not None) and self.time_passed >= self._max_time:
            raise StopCondition("max time reached")

    def _ensure_stop_condition_exists(self):
        if all(i is None for i in (self._max_steps, self._max_passes, self._max_epochs, self._max_batches, self._max_time)):
            raise ValueError('must specify at least one stop condition')

    @torch.no_grad
    def _batch_tfms(self, batch):
        """applies train or test batch transforms to the batch depending on self._status."""
        if self._status == 'train': return self.train_batch_tfms(batch)
        return self.test_batch_tfms(batch)

    def _save_signature(self, obj, name: str):
        """save signature to hyperparameters and return resolved object"""
        if isinstance(obj, sig):
            self.hyperparams.update({name: obj.extra_signature()})
            obj = obj.resolve()
        else:
            self.hyperparams.update({name: get__name__(obj)})
        return obj

    # region plotting
    def _make_solution_image(self, paramvec: torch.Tensor, *args, **kwargs) -> np.ndarray[Any, np.dtype[np.uint8]] | None:
        """make a numpy.uint8 image of the solution generated by `params`, if applicable to this benchmark."""
        return None

    @torch.no_grad
    def plot_solution(self, *args, show=True, fig=None, **kwargs):
        """plot best solution found so far, if applicable to this benchmark."""
        cur_params = torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu()

        # create solution first not to modify Fig if it raises NotImplementedError
        images = []
        labels = []
        if self.best_params is None: raise RuntimeError("No best params")
        sol = self._make_solution_image(self.best_params.to(self.device), *args, **kwargs) # pylint:disable = assignment-from-none
        if sol is not None:
            images.append(sol)
            labels.append('best solution')
        for key, value in self.logger.items():
            if key.startswith(('train image', 'test image')):
                x = value[self.logger.argmin('train loss')]
                images.append(x)
                labels.append(key.replace('train image_', '').replace('test image_', ''))
        if self.reference_images is not None:
            if self.reference_labels is None: raise ValueError("Can't happen")
            images.extend(self.reference_images)
            labels.extend(self.reference_labels)
        if len(images) == 0:
            raise NotImplementedError(f'Solution plotting is not implemented for {self.__class__.__name__}')

        torch.nn.utils.vector_to_parameters(cur_params.to(self.device), self.parameters())
        if fig is None: fig = Fig()
        imshow_grid(images, labels, fig = fig, norm = 'no')
        if show: fig.show()
        return fig

    @torch.no_grad
    def render_video(self, file: str, fps: int = 60, progress=True, *args, **kwargs):
        """renders a video of how current and best solution evolves on each step, if applicable to this benchmark."""
        cur_params = torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu()
        logger_images = {}

        for key, value in self.logger.items():
            if key.startswith(('train image', 'test image')):
                logger_images[key] = list(value.values())

        make_sol_images = 'params' in self.logger
        if make_sol_images:
            param_history = self.logger.get_metric_as_list('params')
            best_param_image = self._make_solution_image(param_history[0].to(self.device)) # pylint:disable = assignment-from-none
        else:
            param_history = []
            best_param_image = None

        with OpenCVRenderer(file, fps = fps) as renderer:
            lowest_loss = float('inf')

            for step, loss in Progress(self.logger['train loss'].items(), sec=0.1, enable=progress):
                # add current and best image
                images = []
                if make_sol_images: images.append(self._make_solution_image(param_history[step].to(self.device)))
                # check if new params are better
                if loss <= lowest_loss:
                    lowest_loss = loss
                    if make_sol_images: best_param_image = images[-1]
                if best_param_image is not None: images.append(best_param_image)

                # add logger images
                for key, value in logger_images.items():
                    images.append(value[step])

                # add reference image
                if self.reference_images is not None:
                    images.extend(self.reference_images)

                if len(images) == 0:
                    raise NotImplementedError(f'Solution plotting is not implemented for {self.__class__.__name__}')

                # make a collage
                images = [force_hw3(i) for i in images]
                max_shape = np.max([i.shape for i in images], axis = 0)
                max_shape[:-1] += 2 # add 2 pixel to spatial dims
                images = np.stack([pad_to_shape(i, max_shape, mode = 'constant', value = 128, crop = True) for i in images])
                # it is now (image, H, W, 3)
                if len(images) == 1: renderer.add_frame(images[0])
                else:
                    ncols = len(images) ** 0.55
                    nrows = round(len(images) / ncols)
                    ncols = round(ncols)
                    nrows = max(nrows, 1)
                    ncols = max(ncols, 1)
                    r = True
                    while nrows * ncols < len(images):
                        if r: ncols += 1
                        else: ncols += 1
                        r = not r
                    n_tiles = nrows * ncols
                    if len(images) < n_tiles: images = np.concatenate([images, np.zeros_like(images[:n_tiles - len(images)])])
                    images = images.reshape(nrows, ncols, *max_shape)
                    images = np.concatenate(np.concatenate(images, 1), 1)
                    renderer.add_frame(images)

        torch.nn.utils.vector_to_parameters(cur_params.to(self.device), self.parameters())

    def plot_loss(self, show=True, fig=None):
        """plot train and test (if applicable to this benchmark) losses per step."""
        fig = self.logger.plot(*(i for i in ['train loss', 'test loss'] if i in self.logger), fig = fig)
        if show: fig.show()
        return fig

    def plot_trajectory(self, log=True, show=True, fig = None):
        """plot parameter trajectory, optionally also plot a loss landscape slice defined by first, middle and last points."""
        if fig is None: fig = Fig()
        fig.scatter(
            x = self.logger.get_metric_as_numpy("proj1"),
            y = self.logger.get_metric_as_numpy("proj2"),
            alpha=0.4,
            s=4,
            c=self.logger.get_metric_as_numpy("train loss"),
            cmap = 'coolwarm',
            norm = 'log' if log else None,
        ).ticks().tick_params(labelsize=7)
        if show: fig.show()
        return fig

    def plot_loss_surface(self, num = 20, cmap: str = 'gray', surface_alpha: float = 1, levels: int = 12, contour_cmap: str = 'binary', contour_lw: float = 0.5, contour_alpha: float = 0.3, norm=None, grid_alpha: float = 0, grid_color: str = 'gray', grid_lw: float = 0.5, show=True, fig = None):
        if fig is None: fig = Fig()

        # define a plane that goes through self.initial_params, self.middle_params, self.best_params
        if self.best_params is not None: x0 = self.best_params
        else: x0 = torch.nn.utils.parameters_to_vector(self.parameters())
        cur_params = torch.nn.utils.parameters_to_vector(self.parameters())

        def func(x, y):
            if self._projections is None: raise ValueError('projections are none')
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x).to(dtype = cur_params.dtype, device = cur_params.device)
                y = torch.as_tensor(y).to(dtype = cur_params.dtype, device = cur_params.device)

            params = x0 + x * self._projections[0] + y * self._projections[1]
            torch.nn.utils.vector_to_parameters(params, self.parameters())
            v = self.get_loss()
            if isinstance(v, tuple): v = v[0]
            return v.detach().cpu().item()

        xrange = (self.logger.min('proj1'), self.logger.max('proj1'))
        yrange = (self.logger.min('proj2'), self.logger.max('proj2'))

        fig.funcplot2d(func, xrange, yrange, num, cmap=cmap,levels=levels,surface_alpha=surface_alpha,contour_cmap=contour_cmap,contour_lw=contour_lw,contour_alpha=contour_alpha,grid_alpha=grid_alpha,grid_color=grid_color,grid_lw=grid_lw,lib= None, dtype=cur_params.dtype, device=cur_params.device, norm = norm).colorbar()

        torch.nn.utils.vector_to_parameters(cur_params, self.parameters())
        if show: fig.show()
        return fig

    def plot_summary(self, nrows=None, show=True, fig = None):
        """plots losses, parameter trajectory and best solution if applicable."""
        if fig is None: fig = Fig()
        self.plot_loss(show = False, fig=fig)
        if 'proj1' in self.logger: self.plot_trajectory(show = False, fig=fig.add('optimization trajectory'))
        try:
            # this calls imshow_grid which adds multiple things to the fig
            # so no add() needed.
            self.plot_solution(show = False, fig=fig)
            if show: fig.show(nrows = 2 if nrows is None else nrows, figsize = (14, 14))
        except NotImplementedError:
            if show: fig.show(nrows = 1 if nrows is None else nrows, figsize = (14, 5))
        return fig
    # endregion

    # region saving
    def save_run_to_dir(self, dir: str):
        """creates dir/logger.npz, dir/hyperparams.yaml, dir/attrs.yaml and dir/model.txt"""
        self.logger.save(os.path.join(dir, 'logger.npz'))
        yamlwrite(
            make_dict_serializeable(self.hyperparams, raw_strings = False, recursive=True),
            os.path.join(dir, 'hyperparams.yaml')
        )
        txtwrite(str(self), os.path.join(dir, 'model.txt'))

        attrs = {k:v for k,v in self.__dict__.copy().items() if (not k.startswith('_')) and isinstance(v, (int,float,str,bool))}
        yamlwrite(attrs, os.path.join(dir, 'attrs.yaml'))

    def save_run(self, root:str):
        """saves run to root/self.name or date if it is none."""
        now = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        if self.name is None:
            name = f'{self.__class__.__name__} {now}'
        else:
            name = f'{self.name} {now}'
        if not os.path.exists(root): os.mkdir(root)
        while os.path.exists(os.path.join(root, name)): name = f'{name}-'
        os.mkdir(os.path.join(root, name))
        self.save_run_to_dir(os.path.join(root, name))
    # endregion

    # region evaluate
    @abstractmethod
    def get_loss(self) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Returns loss or a tuple with loss and metrics dictionary.

        If this is a batch task, passes `self.batch` to the model.
        If required, batch is moved to `self.device`.

        This doesn't call backward on the loss."""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement forward")
        # this method must be overwritten
        # if self.model is None or self.loss_fn is None:
        #     raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a forward")

        # preds = self.model(self.batch[0])
        # loss = self.loss_fn(preds, self.batch[1])
        # accuracy = preds.argmax(1).eq_(targets).float().mean()
        # return loss, {"accuracy": accuracy}
    # endregion

    # region forward
    @final
    def forward(self):
        """closure that can be passed to the optimizer"""
        training = self._status == 'train'
        if training: self._check_stop_condition()

        self._improved = False

        # this is called by `one_batch`, so `self.batch` is assigned
        x = self.get_loss()

        # if x is a tuple, split it into loss and metrics
        if isinstance(x, tuple):
            loss, metrics = x
            # log all metics
            for k, v in metrics.items():
                self.log(f'{self._status} {k}', v)

        else:
            loss = x


        # log loss
        loss_cpu = loss.detach().cpu()
        self.log(f'{self._status} loss', loss_cpu)

        # save loss to lowest_loss (this is only for benchmarks with no test set)
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self._improved = True

        if training:

            self._log_params()
            self._last_train_loss = loss_cpu

            # log time
            cur_time = time.time()
            # start timer after one backward pass to let things compile for free
            if self.start_time is None and self.current_step != 0:
                self.start_time = cur_time
            if self.start_time is not None:
                self.time_passed = cur_time - self.start_time

            self.log('time', self.time_passed)
            self._print_progress(self.time_passed)


            if self.current_step == 0:
                print('first train step...', end = '\r')
                self._first_train_step_time = cur_time
            elif self.current_step == 1:
                print(f'first train step took {(cur_time - self._first_train_step_time):.3f} sec.', end = '\r')

            self.current_step += 1

            # test epoch
            if (self.test_data is not None) and \
                (((self._test_every_steps is not None) and self.current_step % self._test_every_steps == 0) or \
                ((self._test_every_sec is not None) and cur_time - self._last_test_time > self._test_every_sec)):
                self.test_epoch(self.test_data)

        else:
            self._last_test_loss = loss_cpu

        return loss
    # endregion

    def _evaluate_loss(self, params):
        params = totensor(params).to(self.device).ravel()
        torch.nn.utils.vector_to_parameters(params, self.parameters())
        self._status = 'train'
        return self.forward()

    @torch.no_grad
    def get_x0(self):
        return tonumpy(torch.cat([p.ravel() for p in self.parameters() if p.requires_grad]))

    @torch.no_grad
    def evaluate_loss(self, params):
        return maybe_ensure_pynumber(self._evaluate_loss(params))

    def evaluate_loss_and_grad(self, params):
        loss = self._evaluate_loss(params)
        loss.backward()
        grad = torch.cat([(p.grad.ravel() if p.grad is not None else torch.zeros_like(p)) for p in self.parameters() if p.requires_grad])
        return maybe_ensure_pynumber(loss), tonumpy(grad)

    # region closure
    def closure(self, *args: bool, kwargs: _ClosureKwargs):
        """_summary_

        Args:
            kwargs (_ClosureKwargs): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if len(args) > 1: raise ValueError(f'{args = }')

        loss = self.forward()

        if kwargs['enable_grad']: # type:ignore
            if len(args) == 1: backward = args[0]
            else: backward = kwargs['backward'] # type:ignore
            if backward:
                if not loss.requires_grad:
                    warnings.warn("loss didn't require grad!")
                    return loss

                self.num_backwards += 1
                if kwargs['zero_grad']: self.zero_grad() # type:ignore
                loss.backward(retain_graph = kwargs['retain_graph'], create_graph = kwargs["create_graph"]) # type:ignore

        return loss
    # endregion

    # region one_step
    def one_step(self, optimizer, train: bool, kwargs: _ClosureKwargs):
        """one optimizer step (optimizer can evaluate closure multiple times thus performing multiple steps)"""
        if train:
            with torch.enable_grad() if kwargs['enable_grad'] else torch.no_grad(): # type:ignore
                self.train()
                self._status = 'train'
                optimizer.step(partial(self.closure, kwargs = kwargs))
                self.current_batch += 1
        else:
            with torch.inference_mode():
                self.eval()
                self._status = 'test'
                self.closure(False, kwargs = _TEST_KWARGS)

    def one_batch(self, optimizer, batch, train: bool, kwargs: _ClosureKwargs) -> None:
        """one batch for batch learning"""
        self.batch = self._batch_tfms(batch)
        self.one_step(optimizer, train, kwargs)

    def one_epoch(self, optimizer, data, train: bool, kwargs: _ClosureKwargs):
        """one epoch for train/test splits"""
        for batch in data:
            self.one_batch(optimizer, batch, train, kwargs)

        if train:
            self.current_epoch += 1

        else:
            self._aggregate_test_metrics()
    # endregion

    # region test_epoch
    def test_epoch(self, data):
        """one epoch but makes sure to restore previous batch"""
        self._status = 'test'
        batch = self.batch
        self.one_epoch(optimizer = None, data = data, train = False, kwargs = {})
        self.batch = batch
        self._status = 'train'
        self._last_test_time = time.time()
    # endregion

    # region run
    @torch.no_grad
    def run(
        self,
        optimizer,
        max_steps: int | None = None,
        max_passes: int | None = None,
        max_batches: int | None = None,
        max_epochs: int | None = None,
        max_time: float | None = None,
        min_loss: float | None = None,
        test_every_steps: int | None = None,
        test_every_sec: float | None = None,
        name: str | None = None,
        hyperparams: Mapping | None = None,
        **kwargs: Unpack[_ClosureKwargs]
    ):
        optimizer = self._save_signature(optimizer, 'optimizer')
        if name is None and self.name is None:
            name = get__name__(optimizer)
        if name is not None: self.name = name
        if hyperparams is not None: self.hyperparams.update({"hyperparams": hyperparams})

        # self.to(self.device)
        self._max_steps = max_steps
        self._max_passes = max_passes
        self._max_batches = max_batches
        self._max_epochs = max_epochs
        self._max_time = max_time
        self._min_loss = min_loss
        self._test_every_steps = test_every_steps
        self._test_every_sec = test_every_sec
        self._ensure_stop_condition_exists()
        kwargs = _set_closure_defaults_(kwargs)
        self._log_params()
        if self.save_edge_params: self.initial_params = parameters_to_vector(self.parameters())

        if self.test_data is not None:
            # if test_every is None:
                # warnings.warn('test_every is not set, will not test during training!')
            self.test_epoch(self.test_data)

        try:
            # train/test or batch training
            if self.train_data is not None:
                for _ in range(max_epochs) if max_epochs is not None else itertools.count():
                    self.one_epoch(optimizer, self.train_data, train = True, kwargs = kwargs)
            else:
                for _ in range(max_batches) if max_batches is not None else itertools.count():
                    self.one_step(optimizer, train = True, kwargs = kwargs)
        except StopCondition:
            pass
        finally:
            if self.test_data is not None: self.test_epoch(self.test_data)
            self._print_progress(time.time())
    # endregion

    @torch.no_grad
    def run_grid_search(
        self,
        optimizer_cls: type | Callable,
        criterion: str,
        grid_kws: dict[str, Sequence | np.ndarray | torch.Tensor],
        other_kws: dict | None,
        reset_call_kws: dict[str, Callable] | None,
        save_dir: str | None = 'runs',
        maximize = False,
        max_steps: int | None = None,
        max_batches: int | None = None,
        max_epochs: int | None = None,
        max_time: float | None = None,
        min_loss: float | None = None,
        test_every_steps: int | None = None,
        test_every_sec: float | None = None,
        name: str | None = None,
        hyperparams: Mapping | None = None,
        **kwargs: Unpack[_ClosureKwargs]
    ):
        """run a grid search

        Args:
            optimizer_cls (_type_): optimizer constructor.
            criterion (str): logger key to minimize/maximize.
            grid_kws (dict[str, Sequence  |  np.ndarray  |  torch.Tensor]): optimizer kwargs to grid search, value is sequence of values to try.
            other_kws (dict | None): other keywargs to pass to optimizer constructor.
            reset_call_kws (dict[str, Callable] | None):
                kwargs for `reset` method, values must be callables and will be called on each run,
                for example to initialize a new model each run.
            save_dir (str | None, optional): directory to save to, can be None. Defaults to 'runs'.
            maximize (bool, optional): if True maximizes criterion. Defaults to False.
            max_steps (int | None, optional): _description_. Defaults to None.
            max_batches (int | None, optional): _description_. Defaults to None.
            max_epochs (int | None, optional): _description_. Defaults to None.
            max_time (float | None, optional): _description_. Defaults to None.
            min_loss (float | None, optional): _description_. Defaults to None.
            test_every (int | None, optional): _description_. Defaults to None.
            name (str | None, optional): name, uses optimizer name if None. Defaults to None.
            hyperparams (Mapping | None, optional): additional hyperparameters to save. Defaults to None.

        Returns:
            (best grid kwargs, value at best grid kwargs)
        """
        # self.to(self.device)

        if other_kws is None: other_kws = {}
        if reset_call_kws is None: reset_call_kws = {}

        grid = [i.ravel() for i in np.meshgrid(*grid_kws.values(), indexing='ij')]
        grid_vals = {k: grid[i] for i, k in enumerate(grid_kws)}
        best_kwargs = None
        best_value = float('inf')
        if maximize: best_value = -best_value
        for i in range(len(grid[0])):
            self.reset(**{k:v() for k,v in reset_call_kws.items()})
            grid_kwargs = {k: grid_vals[k][i].item() for k in grid_vals}
            opt_kwargs = grid_kwargs.copy()
            opt_kwargs.update(other_kws)
            opt = sig(optimizer_cls, self.parameters(), **opt_kwargs)
            self.run(
                opt,
                max_steps = max_steps,
                max_batches = max_batches,
                max_epochs = max_epochs,
                max_time = max_time,
                min_loss = min_loss,
                test_every_steps = test_every_steps,
                test_every_sec = test_every_sec,
                name = name,
                hyperparams = hyperparams,
                **kwargs
            )
            if save_dir is not None: self.save_run(save_dir)

            if maximize:
                value = self.logger.max(criterion)
                if value > best_value:
                    best_value = value
                    best_kwargs = grid_kwargs
            else:
                value = self.logger.min(criterion)
                if value < best_value:
                    best_value = value
                    best_kwargs = grid_kwargs

        return best_kwargs, best_value

    @torch.no_grad
    def run_optuna_search(
        self,
        optimizer_cls: type | Callable,
        criterion: str,
        n_trials: int,
        kws_maker: Callable[..., dict[str, Any]],
        other_kws: dict | None,
        reset_call_kws: dict[str, Any] | None,
        sampler = None,
        save_dir: str | None = 'runs',
        maximize = False,
        last=False,
        max_steps: int | None = None,
        max_batches: int | None = None,
        max_epochs: int | None = None,
        max_time: float | None = None,
        min_loss: float | None = None,
        test_every_steps: int | None = None,
        test_every_sec: float | None = None,
        name: str | None = None,
        hyperparams: Mapping | None = None,
        **kwargs: Unpack[_ClosureKwargs]
    ):
        """run a grid search

        Args:
            optimizer_cls (_type_): optimizer constructor.
            criterion (str): logger key to minimize/maximize.
            kws_maker (Callable[Trial, dict[str, Any]]): callable that accepts optuna trial and returns dict with kwargs.
            other_kws (dict | None): other keywargs to pass to optimizer constructor.
            reset_call_kws (dict[str, Callable] | None):
                kwargs for `reset` method, values must be callables and will be called on each run,
                for example to initialize a new model each run.
            save_dir (str | None, optional): directory to save to, can be None. Defaults to 'runs'.
            maximize (bool, optional): if True maximizes criterion. Defaults to False.
            max_steps (int | None, optional): _description_. Defaults to None.
            max_batches (int | None, optional): _description_. Defaults to None.
            max_epochs (int | None, optional): _description_. Defaults to None.
            max_time (float | None, optional): _description_. Defaults to None.
            min_loss (float | None, optional): _description_. Defaults to None.
            test_every (int | None, optional): _description_. Defaults to None.
            name (str | None, optional): name, uses optimizer name if None. Defaults to None.
            hyperparams (Mapping | None, optional): additional hyperparameters to save. Defaults to None.

        Returns:
            (best grid kwargs, value at best grid kwargs)
        """
        if other_kws is None: other_kws = {}
        if reset_call_kws is None: reset_call_kws = {}

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARN)
        study = optuna.create_study(sampler = sampler, direction="maximize" if maximize else 'minimize')

        for i in range(n_trials):
            self.reset(**{k:v() for k,v in reset_call_kws.items()})
            trial = study.ask()
            suggested_kwargs = kws_maker(trial)
            opt_kwargs = suggested_kwargs.copy()
            opt_kwargs.update(other_kws)
            opt = sig(optimizer_cls, self.parameters(), **opt_kwargs)
            self.run(
                opt,
                max_steps = max_steps,
                max_batches = max_batches,
                max_epochs = max_epochs,
                max_time = max_time,
                min_loss = min_loss,
                test_every_steps = test_every_steps,
                test_every_sec = test_every_sec,
                name = name,
                hyperparams = hyperparams,
                **kwargs
            )
            if save_dir is not None: self.save_run(save_dir)

            if last: value = self.logger.last(criterion)
            elif maximize: value = self.logger.max(criterion)
            else: value = self.logger.min(criterion)
            if np.isnan(value): value = 1e9
            study.tell(trial, maybe_ensure_pynumber(value)) # type:ignore

        return study
