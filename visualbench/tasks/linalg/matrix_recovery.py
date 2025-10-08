import warnings
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils


class StochasticMatrixRecovery(Benchmark):
    """Optimize B to recover A given matrix-vector products Ax with a random vector.
    The objective is `criterion(Ax, Bx)`.

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
    def __init__(self, A:Any=512, batch_size: int = 1, criterion = F.mse_loss, l1:float=0, l2:float=0, linf:float=0, vec=True, algebra=None, sampler = linalg_utils.row_sampler, seed=0):
        super().__init__(seed=seed)
        generator = self.rng.torch()
        self._make_images = False # will be True if A or B are an image.

        if isinstance(A, int): A = torch.randn(1, A, A, generator=generator)
        elif isinstance(A, tuple) and len(A) == 2: A = torch.randn((1, *A), generator=generator)
        else:
            self._make_images = True
            A = format.to_CHW(A, generator=self.rng.torch())
        self.A = nn.Buffer(A)
        self.min = self.A.min().item(); self.max = self.A.max().item()
        self.B = nn.Parameter(torch.zeros_like(self.A))

        self.batch_size = batch_size
        self.vec = vec
        self.criterion = criterion
        self.l1 = l1
        self.l2 = l2
        self.linf = linf
        self.sampler = sampler

        self.algebra = algebras.get_algebra(algebra)

        if self._make_images:
            self.add_reference_image('A', A, to_uint8=True)

    def pre_step(self):
        if self.vec:
            b, n, m = self.A.shape
            self.X = self.sampler((self.batch_size, b, m, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            self.X = self.sampler((self.batch_size, *self.A.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        X = self.X

        AX = algebras.matmul(self.A, X, self.algebra)
        BX = algebras.matmul(self.B, X, self.algebra)

        penalty = 0
        if self.l1 != 0: penalty = penalty + torch.linalg.vector_norm(self.B, ord=1) # pylint:disable=not-callable
        if self.l2 != 0: penalty = penalty + torch.linalg.vector_norm(self.B, ord=2) # pylint:disable=not-callable
        if self.linf != 0: penalty = penalty + torch.linalg.vector_norm(self.B, ord=float('inf')) # pylint:disable=not-callable

        with torch.no_grad():
            test_loss = self.criterion(self.B, self.A)
            self.log('test loss', test_loss + penalty)

            if self._make_images:
                self.log_image("B", self.B, to_uint8=True, log_difference=True, show_best=True, min=self.min, max=self.max)
                self.log_image('residual', (self.B - self.A).abs_(), to_uint8=True)

        return self.criterion(AX, BX) + penalty

