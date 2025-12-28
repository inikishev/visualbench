# pylint:disable=no-member
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import tonumpy, totensor
from .function_descent import _safe_flatten
from .test_functions import TEST_FUNCTIONS, TestFunction


class LearnAndMinimize(Benchmark):
    """Joinly optimize a neural net to model the function, and minimize the model.

    Model should accept input ``(B, 2)`` and output ``(B, 1)`` or ``(B, )``.

    Args:
        func: func.
        model: model.
        x0: x0. Defaults to None.
        domain: domain. Defaults to None.
        criterion: criterion. Defaults to F.mse_loss.
        pass_all: pass_all. Defaults to True.
        dtype: dtype. Defaults to torch.float32.
        n_points: n_points. Defaults to 128.
        log_scale: log_scale. Defaults to False.
        w_func: w_func. Defaults to 'auto'.
        unpack: unpack. Defaults to True.
    """
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        model: torch.nn.Module,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        criterion = F.mse_loss,
        batch_size: int = 10_000,
        dtype: torch.dtype = torch.float32,
        n_points: int = 128,
        log_scale: bool = False,
        w_func: float | Literal['auto'] = 'auto',
        unpack=True,
    ):
        if isinstance(func,str): f = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = dtype)
        else: f = func

        if isinstance(f, TestFunction):
            if x0 is None: x0 = f.x0()
            if domain is None: domain = f.domain()
            unpack = True

        super().__init__()

        self.func: Callable[..., torch.Tensor] | TestFunction = f # type:ignore
        self.xy = torch.nn.Parameter(totensor(x0, dtype=dtype))

        if domain is not None: self._domain = tonumpy(_safe_flatten(domain))
        else: raise RuntimeError("Domain is required")

        self.model = model
        self.log_scale = log_scale
        self.unpack = unpack
        self.n_points = n_points
        self.w_func = w_func

        x = torch.linspace(self._domain[0], self._domain[1], n_points)
        y = torch.linspace(self._domain[2], self._domain[3], n_points)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        self.mesh = torch.nn.Buffer(torch.stack([X, Y], -1)) # (n, n, 2)
        self.batch_size = batch_size
        self.criterion = criterion

        self.points = []
        self.values = []
        self.set_multiobjective_func(lambda x: torch.maximum(torch.max(x), torch.mean(x)))

    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def _unpacked_func(self, x, y):
        if self.unpack:
            return self.func(x,y)
        else:
            return self.func(torch.stack([x,y])) # type:ignore

    def get_loss(self):

        # add point at current value
        with torch.no_grad():
            xy_value = self._unpacked_func(*self.xy).detach()

        loss_func = self.model(self.xy.unsqueeze(0)).view(-1) # need to have xy with graph here

        self.points.append(self.xy.detach().clone()) # pylint:disable=not-callable
        self.values.append(xy_value.clone())

        # model loss
        points = torch.stack(self.points[-self.batch_size:])
        values_true = torch.stack(self.values[-self.batch_size:])
        values_pred = self.model(points).view(-1)

        loss_model = self.criterion(values_pred.view(-1), values_true.view(-1))

        self.log("xy", self.xy, plot=False)
        self.log("loss - model", loss_model, log_scale=True)
        self.log("loss - func", loss_model, log_scale=True)

        # visualize current decision boundary
        if self._make_images:
            with torch.no_grad():
                mesh_flat = self.mesh.flatten(0, 1) # n^2, 2
                Z_flat: torch.Tensor = self.model(mesh_flat)

                if self.log_scale:
                    Z_flat.add_(1e-10).log_()

                Z = Z_flat.squeeze().view(self.n_points, self.n_points)

                self.log_image("trajectory", Z, to_uint8=True, show_best=True)

        if self.w_func == 'auto':
            self.w_func = (loss_model / loss_func).item()

        return torch.stack([loss_func.view(-1) * self.w_func, loss_model.view(-1)])

