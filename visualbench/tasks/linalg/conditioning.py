import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

import torch
import torch.nn.functional as F
from torch import nn

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils


def _get_A_precond(P, A, inverse, algebra):
    P_inv = None
    if inverse:
        A_precond = algebras.matmul(P, A, algebra=algebra) # pylint:disable=not-callable
    else:
        if algebra is not None:
            P_inv, _ = torch.linalg.inv_ex(P) # pylint:disable=not-callable
            A_precond = algebras.matmul(P_inv, A, algebra=algebra)
        else:
            A_precond, _ = torch.linalg.solve_ex(P, A) # pylint:disable=not-callable

    return A_precond, P_inv

class Preconditioner(Benchmark):
    """optimize a preconditioner P such that P^-1 A has better condition number than A.
    If ``inverse``, optimizes inverse preconditioner, so PA has a better condition number than A"""
    def __init__(self, A, p: Any=2, inverse:bool=True, algebra=None):
        super().__init__()
        self.A = nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))
        *b, m, n = self.A.shape
        self.P = nn.Parameter(linalg_utils.orthogonal((*b, n, max(n, m)), generator=self.rng.torch()))

        self.p = p
        self.algebra = algebras.get_algebra(algebra)
        self.inverse = inverse

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        A = self.A
        P = self.P

        A_precond, P_inv = _get_A_precond(P, A, self.inverse, self.algebra)
        loss = torch.linalg.cond(A_precond, p=self.p).mean() # pylint:disable=not-callable

        if self._make_images:
            self.log_image('P', P, to_uint8=True, log_difference=True)
            if P_inv is not None: self.log_image('P^-1', P_inv, to_uint8=True)
            self.log_image('A preconditioned', A_precond, to_uint8=True)

        return loss

class StochasticPreconditioner(Benchmark):
    """optimize a unit norm preconditioner P to minimize distance between ``P^-1 A v`` and ``P^-1 A v+r``,
    where v is a random vector and r is a small perturbation.
    """
    def __init__(self, A, criterion = F.mse_loss, inverse:bool=True, ord='fro', sigma=1e-2, algebra=None):
        super().__init__()
        self.A = nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))
        b, m, n = self.A.shape
        self.P = nn.Parameter(linalg_utils.beye(size=(b, n, max(n, m))))

        self.sigma = sigma * self.A.abs().mean()
        self.algebra = algebras.get_algebra(algebra)
        self.inverse = inverse
        self.ord = ord
        self.criterion = criterion

        self.add_reference_image('A', self.A, to_uint8=True)

    def pre_step(self):
        b, m, n = self.A.shape
        k = max(m, n)
        kw = {"device":self.A.device, "dtype":self.A.dtype, "generator":self.rng.torch(self.A.device)}
        self.v = torch.randn(k, **kw)
        self.v_p = self.v + torch.randn(k, **kw) * self.sigma

    def get_loss(self):
        A = self.A
        P = self.P / torch.linalg.norm(self.P, ord=self.ord, dim=(-2,-1), keepdim=True) # pylint:disable=not-callable
        A_precond, P_inv = _get_A_precond(P, A, self.inverse, self.algebra)

        Av = algebras.matmul(A_precond, self.v, algebra=self.algebra)
        Av_p = algebras.matmul(A_precond, self.v_p, algebra=self.algebra)

        loss = self.criterion(Av, Av_p) / self.sigma

        with torch.no_grad():
            test_loss = torch.linalg.cond(A_precond).mean() # pylint:disable=not-callable
            self.log('test loss', test_loss)

            if self._make_images:
                self.log_image('P', P, to_uint8=True, log_difference=True)
                if P_inv is not None: self.log_image('P^-1', P_inv, to_uint8=True)
                self.log_image('A preconditioned', A_precond, to_uint8=True)

        return loss

