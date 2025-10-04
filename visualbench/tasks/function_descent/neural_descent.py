# pylint:disable=no-member
"""2D function descent"""

import os
from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from ...benchmark import Benchmark
from ...utils._benchmark_video import _maybe_progress, GIF_POST_PBAR_MESSAGE
from ...utils.format import tonumpy, totensor
from ...utils.funcplot import funcplot2d
from ...utils.renderer import OpenCVRenderer
from .test_functions import TEST_FUNCTIONS, TestFunction

from .function_descent import _UnpackCall, _safe_flatten, FunctionDescent

class NeuralDescent(FunctionDescent):
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
    def __init__(
        self,
        model: torch.nn.Module,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        minima = None,
        dtype: torch.dtype = torch.float32,
        mo_func: Callable | None = None,
        unpack=True,
    ):
        super().__init__(func=func, x0=(0,0), domain=domain, minima=minima, dtype=dtype, mo_func=mo_func, unpack=unpack)
        self.xy.requires_grad_(False)
        self.model = model

    def get_loss(self):
        xy = self.model()
        if self.unpack:
            loss = self.func(xy[0], xy[1])
        else:
            loss = self.func(xy) # type:ignore

        self.log("xy", xy, plot=False)
        return loss


#
# booth = vb.TEST_FUNCTIONS["booth"]

# class Sus(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1024, 2)

#         # optimize x to initial
#         x0 = vb.totensor(booth.x0())
#         x = torch.randn(1024, requires_grad=True)
#         opt = tz.Optimizer([x], tz.m.LBFGS(), tz.m.Backtracking())
#         for _ in range(100):
#             def closure(backward=True):
#                 loss = (self.linear(x) - x0).pow(2).mean()
#                 if backward:
#                     opt.zero_grad()
#                     loss.backward()
#                 return loss
#             opt.step(closure)

#         self.x = nn.Buffer(x.requires_grad_(False))

#     def forward(self):
#         return self.linear(self.x)

# bench = vb.NeuralDescent(Sus(), 'booth').cuda()

# print(f'{bench.ndim = }')
# # opt = torch.optim.Adam(bench.parameters(), 1e-1)
# opt = tz.Optimizer(bench, tz.m.GGT(), tz.m.LR(1e-2))

# bench.run(opt, 100)
# bench.plot()