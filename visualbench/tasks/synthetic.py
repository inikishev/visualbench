"""synthetic funcs"""
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark, _sum_of_squares
from ..data import get_ill_conditioned
from ..utils import algebras, to_CHW, to_square, totensor
from .linalg.linalg_utils import orthogonal


class Sphere(Benchmark):
    """Sphere benchmark (or other function depending on criterion).

    Directly minimizes ``criterion`` between ``target`` and ``init``.

    Renders:
        if target is an image, renders the current solution and the error.

    Args:
        target (Any): if int, used as number of dims, otherwise is the target itself.
        init (Any, optional): initial values same shape as target, if None initializes to zeros. Defaults to None.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
    """
    def __init__(self, target: Any, init=None, criterion = torch.nn.functional.mse_loss,):
        super().__init__()

        # target tensor
        if isinstance(target, int): target = torch.randn(target, dtype=torch.float32, generator=self.rng.torch())
        self.target = torch.nn.Buffer(totensor(target).float())

        # preds tensor
        if init is None: init = torch.zeros_like(self.target)
        self.x = torch.nn.Parameter(totensor(init).float().contiguous())
        self.criterion = criterion

        # reference image for plotting
        if self.target.squeeze().ndim in (2, 3):
            self.add_reference_image('target', self.target.squeeze(), to_uint8=True)
            # enable showing best solution so far

    def get_loss(self):
        # log current recreated image if target is an image
        if self._make_images and len(self._reference_images) != 0:
            with torch.no_grad():
                self.log_image('preds', self.x, to_uint8=True, log_difference=True)
                self.log_image('residual', (self.x-self.target).abs_(), to_uint8=True, log_difference=True)

        # return loss
        return self.criterion(self.x, self.target)

# all other funcs moved to projected

def rosenbrock(x:torch.Tensor, a=1., b=100., pd_fn=torch.square):
    x1 = x[..., :-1]
    x2 = x[..., 1:]

    term1 = x2 - x1**2
    term2 = a - x1

    return (b * pd_fn(term1) + pd_fn(term2)).mean(-1)

class Rosenbrock(Benchmark):
    """banana-shaped function"""
    def __init__(
        self,
        dim=512,
        a=1.0,
        b=100.0,
        pd_fn=torch.square,
        shift: float = 1e-1,
    ):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([-1.2, 1.]).repeat(dim//2).clone())
        self.shift = torch.nn.Buffer(torch.randn(self.x.size(), generator=self.rng.torch()) * shift)
        self.pd_fn = pd_fn
        self.a = a
        self.b = b

    def get_loss(self):
        x = self.x + self.shift
        return rosenbrock(x, self.a, self.b, self.pd_fn)


def chebushev_rosenbrock(x:torch.Tensor, a=1.0, p=100.0, pd_fn=torch.square):
    x1 = x[..., :-1]
    x2 = x[..., 1:]

    term1 = pd_fn(x[..., 0] - 1)
    term2 = pd_fn(x2 - 2*pd_fn(x1) + 1)

    return a * term1 + p * torch.sum(term2, -1)


class ChebushevRosenbrock(Benchmark):
    """Nesterov’s Chebyshev-Rosenbrock Functions."""
    def __init__(self, dim=128, p=10, a=1/4, pd_fn=torch.square, shift:float=1):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([-1.2, 1.]).repeat(dim//2).clone())
        self.shift = torch.nn.Buffer(torch.randn(self.x.size(), generator=self.rng.torch()) * shift)
        self.pd_fn = pd_fn
        self.a = a
        self.p = p

    def get_loss(self):
        x = self.x + self.shift
        return chebushev_rosenbrock(x, self.a, self.p, self.pd_fn)


def rotated_quadratic(x:torch.Tensor, c=1.9999):
    term1 = x.pow(2).sum(-1)
    term2 = x.sum(-1).pow(2)
    w1 = 1.0 - 0.5 * c
    w2 = 0.5 * c

    return w1*term1 + w2*term2

class RotatedQuadratic(Benchmark):
    """The diabolical quadratic with a hessian that looks like this

    ```python
    tensor([[2, c, c, c],
            [c, 2, c, c],
            [c, c, 2, c],
            [c, c, c, 2]])
    ```

    condition number is (2 + c(dim-1)) / (2 - c).

    This is as ill conditioned as you can get. When `c` is closer to 2, it becomes more ill-conditioned.
    """
    def __init__(self, dim=512, c=1.9999):
        super().__init__()
        generator = self.rng.torch()
        self.x = torch.nn.Parameter(torch.randn(dim, generator=generator))
        self.c = c
        self.shift = torch.nn.Buffer(torch.randn(dim, generator=generator, requires_grad=False))

    def get_loss(self):
        x = self.x + self.shift
        return rotated_quadratic(x, self.c)


def rastrigin(x:torch.Tensor, A=10.0):
    term1 = torch.sum(x**2, dim=-1)
    term2 = A * (2 * math.pi * x).cos().sum(-1)

    return A * x.size(-1) + term1 - term2


class Rastrigin(Benchmark):
    """Classic non-convex function with many local minima."""
    def __init__(
        self,
        dim=512,
        A=10.0,
    ):
        super().__init__()
        self.x = nn.Parameter(torch.randn(dim, generator=self.rng.torch()) * 5.12)
        self.A = A
        self.dim = dim
        self.shift = nn.Buffer(torch.randn((dim,), generator=self.rng.torch()))

    def get_loss(self):
        x = self.x + self.shift
        return rastrigin(x, self.A)

def ackley(x:torch.Tensor, a=20.0, b=0.2, c=2*torch.pi):
    dim = x.size(-1)
    sum_sq = x.pow(2).sum(-1)
    term1 = -a * (-b * (sum_sq / dim).sqrt()).exp()

    sum_cos = (c*x).cos().sum(-1)
    term2 = -torch.exp(sum_cos / dim)

    return term1 + term2 + a + math.e



class Ackley(Benchmark):
    """Another classic non-convex function with many local minima."""
    def __init__(
        self,
        dim=512,
        a=20.0,
        b=0.2,
        c=2 * math.pi,
    ):
        super().__init__()
        self.x = nn.Parameter(torch.randn(dim, generator=self.rng.torch()) * 15)
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c
        self.shift = nn.Buffer(torch.randn((dim,), generator=self.rng.torch()))

    def get_loss(self):
        x = self.x + self.shift
        return ackley(x, self.a, self.b, self.c)
