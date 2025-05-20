"""synthetic funcs"""
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark
from ..utils import from_algebra, get_algebra, to_CHW, to_square, totensor


class Sphere(Benchmark):
    """Sphere benchmark (or other function depending on criterion).

    Directly minimizes `criterion` between `target` and `init`.

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
            self.log_image('preds', self.x, to_uint8=True, log_difference=True)

        # return loss
        return self.criterion(self.x, self.target)


class QuadraticForm(Benchmark):
    """Basic convex quadratic objective with linear term.

    Args:
        dim (int, optional): number of dimensions. Defaults to 512.
        eps (int, optional): to make sure matrix is positive definite. Defaults to 1e-8.
        shift (bool | None, optional): shifts loss so that minimal loss is close to 0 (not exactly due to imprecision)
        seed (int, optional): rng seed. Defaults to 0.
    """
    def __init__(self, dim=512, eps=1e-4, algebra=None, shift=None, seed=0):
        super().__init__(seed=seed)
        generator = self.rng.torch()
        self.dim = dim

        H = torch.randn(dim, dim, generator=generator)
        self.H = torch.nn.Buffer((H @ H.mT) + torch.eye(dim)*eps) # positive definite matrix
        self.b = torch.nn.Buffer(torch.randn(dim, generator=generator))
        self.target = torch.nn.Buffer(torch.randn(dim, generator=generator))

        self.x = torch.nn.Parameter(torch.randn(dim, generator=generator))
        self.algebra = get_algebra(algebra)

        self.min_value = None
        if shift is None: shift = (algebra is None) and (dim <= 10_000)
        if shift:
            # shift so that minimal value is 0
            try:
                sol, success = torch.linalg.solve_ex(self.H, -self.b) # pylint:disable=not-callable
                if success: self.min_value = 0.5*(sol @ self.H).dot(sol) + sol.dot(self.b)
            except Exception:
                pass


    def get_loss(self):
        x = self.x
        H = self.H * 0.5 # before algebra

        if self.algebra is not None: x, H = self.algebra.convert(x, H)
        x = x + self.target

        term1 = (x @ H).dot(x) # pyright:ignore[reportArgumentType]
        term2 = x.dot(self.b)

        loss = from_algebra(term1 + term2)
        if self.min_value is not None: return loss - self.min_value
        return loss


class Rosenbrock(Benchmark):
    """multidimensional rosenbrock"""

    def __init__(self, dim=4096, seed=0):
        super().__init__()
        p = torch.full((dim,), -1.5, dtype=torch.float32)
        self.x = torch.nn.Parameter(p + torch.randn(p.shape, dtype=torch.float32, generator=self.rng.torch())*0.05)

    def get_loss(self):
        x = self.x
        x_i = x[:-1]
        x_i_plus_1 = x[1:]

        term1 = (1 - x_i)**2
        term2 = 100 * (x_i_plus_1 - x_i**2)**2
        loss = torch.sum(term1 + term2)

        return loss


class IllConditioned(Benchmark):
    """the diabolical hessian looks like this

    .. code:: py

        tensor([[2, c, c, c],
                [c, 2, c, c],
                [c, c, 2, c],
                [c, c, c, 2]])

    condition number is (2 + c(dim-1)) / (2 - c)
    """
    def __init__(self, dim=512, c=1.9999, seed=0):
        super().__init__(seed=seed)
        generator = self.rng.torch()

        self.x = torch.nn.Parameter(torch.randn(dim, generator=generator))
        self.c = c
        self.b = torch.nn.Buffer(torch.randn(dim, generator=generator, requires_grad=False))


    def get_loss(self):
        x = self.x + self.b; c = self.c

        susq = torch.sum(x**2)
        sum = torch.sum(x)
        term1 = 1.0 - 0.5 * c
        term2 = 0.5 * c

        return term1 * susq + term2 * (sum**2)


