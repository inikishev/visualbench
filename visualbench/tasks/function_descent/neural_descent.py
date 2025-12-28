# pylint:disable=no-member
from collections.abc import Callable, Iterable, Sequence
import numpy as np
import torch

from .test_functions import TestFunction, TEST_FUNCTIONS
from ...utils import totensor, tonumpy

from .function_descent import FunctionDescent


class _ShiftedModel(torch.nn.Module):
    def __init__(self, model, shift):
        super().__init__()
        self.model = model
        self.shift = torch.nn.Buffer(totensor(shift).to(device=next(iter(self.model.parameters())).device))
    def forward(self):
        return self.model() + self.shift

class DecisionSpaceDescent(FunctionDescent):
    """Optimize a model to output coordinates that minimize a function.

    The model should accept no arguments, and output length-2 tensor with x and y coordinates.

    For example the model may be a linear layer with random or fixed inputs
    defined within the model.

    Args:
        model (torch.nn.Module):
            model.
        func (Callable | str):
            function or string name of one of the test functions.
        bounds:
            Only used for 2D functions. Either ``(xmin, xmax, ymin, ymax)``, or ``((xmin, xmax), (ymin, ymax))``.
            This is only used for plotting and defines the extent of what is plotted. If None,
            bounds are determined from minimum and maximum values of coords that have been visited.
        minima (_type_, optional): optinal coords of the minima. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        device (torch.types.Device, optional): device. Defaults to "cuda".
        unpack (bool, optional): if True, function is called as ``func(*x)``, otherwise ``func(x)``. Defaults to True.
    """
    _LOGGER_XY_KEY: str = "train xy"
    _LEARNABLE_XY = False
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        model: torch.nn.Module,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        minima = None,
        dtype: torch.dtype = torch.float32,
        mo_func: Callable | None = None,
        unpack=True,
    ):
        super().__init__(func=func, x0=(0,0), domain=domain, minima=minima, dtype=dtype, mo_func=mo_func, unpack=unpack)
        self.model = model

    @classmethod
    def with_x0(
        cls,
        func,
        model: torch.nn.Module,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
    ):
        if isinstance(func, str):
            func = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = torch.float32)

        if x0 is None and isinstance(func, TestFunction):
            x0 = func.x0()

        # shift the model output so that initial point outputted by the model lands to x0
        if x0 is not None:

            x0 = tonumpy(x0)
            model_out = tonumpy(model())
            model = _ShiftedModel(model, shift=x0-model_out)

        return cls(model=model, func=func, domain=domain)

    def get_loss(self):
        xy = self.model().squeeze()
        if self.unpack:
            loss = self.func(xy[0], xy[1])
        else:
            loss = self.func(xy) # type:ignore

        self.log("xy", xy, plot=False)
        return loss

