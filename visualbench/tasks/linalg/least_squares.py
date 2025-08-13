"""synthetic funcs"""
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import to_CHW, algebras


class LeastSquares(Benchmark):
    """Least squares. The objective is to find X such that AX = B.

    Args:
        A (Any, optional): (m, n). Defaults to 512.
        B (Any, optional): (m, ) or (m, k). Defaults to 512.
        criterion (Callable, optional): loss. Defaults to F.mse_loss.
        l1 (float, optional): L1 penalty. Defaults to 0.
        l2 (float, optional): L2 penalty. Defaults to 0.
        linf (float, optional): Linf penalty (penalty for maximum value). Defaults to 0.
        algebra (Any, optional): custom algebra for matmul. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(self, A:Any=512, B:Any=512, criterion = F.mse_loss, l1:float=0, l2:float=0, linf:float=0, algebra=None, seed=0):
        super().__init__(seed=seed)
        generator = self.rng.torch()

        self.A = nn.Buffer(to_CHW(A, generator=generator))
        self.min, self.max = self.A.min().item(), self.A.max().item()
        b, m, n = self.A.shape

        self.B = nn.Buffer(to_CHW(B, generator=generator))

        if self.B.ndim == 1:
            assert self.B.size(0) == m, self.B.shape
            self.k = None
            self.X = nn.Parameter(torch.zeros(b, n))

        else:
            assert self.B.size(-2) == m, self.B.shape
            self.k = self.B.size(-1)
            self.X = nn.Parameter(torch.zeros(b, n, self.k))

        self.criterion = criterion
        self.l1 = l1
        self.l2 = l2
        self.linf = linf

        self.algebra = algebras.get_algebra(algebra)

        if self._make_images:
            self.add_reference_image('A', A, to_uint8=True)
            self.add_reference_image('B', B, to_uint8=True)

    def get_loss(self):
        X = self.X
        if X.ndim == 2: X = X.unsqueeze(-1)
        AX = algebras.matmul(self.A, X, self.algebra)
        if X.ndim == 2: AX = AX.squeeze(-1)

        penalty = 0
        if self.l1 != 0: penalty = penalty + torch.linalg.vector_norm(self.X, ord=1) # pylint:disable=not-callable
        if self.l2 != 0: penalty = penalty + torch.linalg.vector_norm(self.X, ord=2) # pylint:disable=not-callable
        if self.linf != 0: penalty = penalty + torch.linalg.vector_norm(self.X, ord=float('inf')) # pylint:disable=not-callable

        if self._make_images:
            with torch.no_grad():
                self.log_image("X", self.X, to_uint8=True, log_difference=True)
                if self.k is not None:
                    self.log_image("AX", AX, to_uint8=True, show_best=True, min=self.min, max=self.max)
                    self.log_image("residual", (AX - self.B).abs_(), to_uint8=True)

        return self.criterion(AX, self.B) + penalty


