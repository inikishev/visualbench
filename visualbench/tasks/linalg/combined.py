from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor
from .linalg_utils import orthogonal, orthogonal_like, row_sampler, eye_like


class StochasticRLstsq(Benchmark):
    """This objective jointly recovers A and B from matrix-vector products
    and optimizes X to be the solution to recovered A and B.

    This returns 3 losses:

    1. ``criterion(A * va, A_hat * va)``

    2. ``criterion(B * vb, B_hat * vb)``

    3. ``criterion(A_hat * X * vb, B_hat * vb)``

    where ``va`` and ``vb`` are random vectors. By default those random vectors sample a random row of the matrix.
    """
    def __init__(self, A, B, batch_size:int=1, criterion = F.mse_loss, vec=True, algebra=None, seed=0, sampler=row_sampler):
        super().__init__(seed=seed)
        generator = self.rng.torch()

        self.A = nn.Buffer(to_CHW(A, generator=generator)) # (B, H, W)
        self.min, self.max = self.A.min().item(), self.A.max().item()
        self.B = nn.Buffer(to_CHW(B, generator=generator)) # (B, H, W)

        b, n, m = self.A.shape
        b, n, k = self.B.shape
        self.A_hat = nn.Parameter(eye_like(self.A)) # (B, N, M)
        self.B_hat = nn.Parameter(eye_like(self.B)) # (B, N, K)
        self.X = nn.Parameter(torch.zeros(b, m, k)) # (B, M, K)

        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.vec = vec
        self.batch_size = batch_size
        self.sampler = sampler

        if self._make_images:
            self.add_reference_image('A', A, to_uint8=True)
            self.add_reference_image('B', B, to_uint8=True)

        self.set_multiobjective_func(torch.sum)

    def pre_step(self):
        b, n, m = self.A.shape
        b, n, k = self.B.shape
        if self.vec:
            self.V_a = self.sampler((self.batch_size, b, m, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

            self.V_b = self.sampler((self.batch_size, b, k, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            self.V_a = self.sampler((self.batch_size, *self.A.mT.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

            self.V_b = self.sampler((self.batch_size, *self.B.mT.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        A = self.A; B = self.B; X = self.X
        A_hat = self.A_hat; B_hat = self.B_hat

        Av = algebras.matmul(A, self.V_a, algebra=self.algebra)
        A_hat_v = algebras.matmul(A_hat, self.V_a, algebra=self.algebra)
        loss1 = self.criterion(Av, A_hat_v)

        Bv = algebras.matmul(B, self.V_b, algebra=self.algebra)
        B_hat_v = algebras.matmul(B_hat, self.V_b, algebra=self.algebra)
        loss2 = self.criterion(Bv, B_hat_v)

        AX = algebras.matmul(A_hat, X, algebra=self.algebra)
        AX_v = algebras.matmul(AX, self.V_b, algebra=self.algebra)
        loss3 = self.criterion(AX_v, B_hat_v)

        with torch.no_grad():
            test_loss1 = self.criterion(A, A_hat)
            test_loss2 = self.criterion(B, B_hat)
            test_loss3 = self.criterion(A@X, B)
            self.log("test loss", test_loss1 + test_loss2 + test_loss3)
            if self._make_images:
                self.log_image("A_hat", A_hat, to_uint8=True, log_difference=True)
                self.log_image("B_hat", B_hat, to_uint8=True, log_difference=True)
                self.log_image("X", self.X, to_uint8=True, log_difference=True)
                self.log_image("AX", AX, to_uint8=True, show_best=True, min=self.min, max=self.max)
                self.log_image("residual", (AX - self.B_hat).abs_(), to_uint8=True)

        return torch.stack([loss1, loss2, loss3])



