import warnings
from collections.abc import Callable
from typing import Literal, overload, Any

from torch import nn
import torch

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils


class Preconditioner(Benchmark):
    """find preconditioner P such that P^-1 A has better condition number than A. if inverse, P is inverse preconditioner"""
    def __init__(self, A, p: Any=2, inverse:bool=True, algebra=None):
        super().__init__()
        self.A = nn.Buffer(format.to_CHW(A))
        *b, m, n = self.A.shape
        self.P = nn.Parameter(linalg_utils.orthogonal((*b, m, m), generator=self.rng.torch()))

        self.p = p
        self.algebra = algebras.get_algebra(algebra)
        self.inverse = inverse

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        A = self.A
        P = self.P
        P_inv = None

        if self.inverse:
            A_precond = algebras.matmul(P, A, algebra=self.algebra) # pylint:disable=not-callable
        else:
            if self.algebra is not None:
                P_inv, _ = torch.linalg.inv_ex(P) # pylint:disable=not-callable
                A_precond = algebras.matmul(P_inv, A, algebra=self.algebra)
            else:
                A_precond, _ = torch.linalg.solve_ex(P, A) # pylint:disable=not-callable


        loss = torch.linalg.cond(A_precond, p=self.p).mean() # pylint:disable=not-callable

        if self._make_images:
            self.log_image('P', P, to_uint8=True, log_difference=True)
            if P_inv is not None: self.log_image('P_inv', P_inv, to_uint8=True)
            self.log_image('A preconditioned', A_precond, to_uint8=True)

        return loss

