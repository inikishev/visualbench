# pylint:disable=no-member
"""2D function descent"""


from collections.abc import Callable, Iterable, Sequence

import torch

from ...benchmark import Benchmark
from ...utils.format import tonumpy, totensor
from .function_descent import _safe_flatten
from .test_functions import TEST_FUNCTIONS, TestFunction


class SimultaneousFunctionDescent(Benchmark):
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        n: int = 128,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        dtype: torch.dtype = torch.float32,
        log_scale:bool=False,
        unpack=True,
    ):
        if isinstance(func,str): f = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = dtype)
        else: f = func

        if isinstance(f, TestFunction):
            if domain is None: domain = f.domain()
            unpack = True

        super().__init__()

        self.func: Callable[..., torch.Tensor] | TestFunction = f # type:ignore

        if domain is not None: self._domain = tonumpy(_safe_flatten(domain))
        else: raise RuntimeError("Domain is required")

        self.unpack = unpack
        self.log_scale = log_scale

        x = torch.linspace(self._domain[0], self._domain[1], n)
        y = torch.linspace(self._domain[2], self._domain[3], n)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        self.P = torch.nn.Parameter(torch.stack([X, Y]))

    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def get_loss(self):
        if self.unpack:
            loss = self.func(*self.P)
        else:
            loss = self.func(self.P) # type:ignore

        if self._make_images:
            with torch.no_grad():
                v_loss = loss.flip(0)
                if self.log_scale: v_loss = (loss+1e-10).log()
                self.log_image("loss", v_loss, to_uint8=True, show_best=True)
                P = self.P.flip(1)
                self.log_image("X", P[0], to_uint8=True, min=self._domain[0], max=self._domain[1])
                self.log_image("Y", P[1], to_uint8=True, min=self._domain[2], max=self._domain[3])

        return loss.mean()
