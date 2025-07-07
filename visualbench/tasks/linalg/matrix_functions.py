from typing import Literal

import torch

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor


class MatrixRoot(Benchmark):
    def __init__(self, A, p: int, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.B = torch.nn.Parameter(self.A / nuc)
        self.p = p
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        B = self.B

        powers = []
        if self.algebra is None:
            B_p = torch.linalg.matrix_power(B, n=self.p) # pylint:disable=not-callable
        else:
            B_p = B
            for _ in range(1, self.p):
                B_p = self.algebra.matmul(B, B_p)
                if self._make_images: powers.append(B_p)

        loss = self.criterion(B_p, self.A)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.p}', B_p, to_uint8=True, show_best=True)

        return loss


class StochasticMatrixRoot(Benchmark):
    def __init__(self, A, p: int, batch_size: int = 1, criterion=torch.nn.functional.mse_loss, vec=False, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.B = torch.nn.Parameter(self.A / nuc)
        self.p = p
        self.batch_size = batch_size
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.vec = vec

        self.add_reference_image('A', A, to_uint8=True)

    def pre_step(self):
        if self.vec:
            *b, n, m = self.A.shape
            self.X = torch.randn((self.batch_size, *b, 1, m), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            self.X = torch.randn((self.batch_size, *self.A.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        B = self.B
        X = self.X

        powers = []
        if self.algebra is None:
            B_p = torch.linalg.matrix_power(B, n=self.p) # pylint:disable=not-callable
        else:
            B_p = B
            for _ in range(1, self.p):
                B_p = self.algebra.matmul(B, B_p)
                if self._make_images: powers.append(B_p)

        XA = algebras.matmul(X, self.A, self.algebra)
        XB_p = algebras.matmul(X, B_p, self.algebra)
        loss = self.criterion(XA, XB_p)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.p}', B_p, to_uint8=True, show_best=True)

        return loss

class MatrixLogarithm(Benchmark):
    def __init__(self, A, criterion=torch.nn.functional.mse_loss):
        super().__init__()
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.B = torch.nn.Parameter(torch.zeros_like(self.A))
        self.criterion = criterion

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A_hat = torch.linalg.matrix_exp(self.B) # pylint:disable=not-callable
        loss = self.criterion(A_hat, self.A)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            self.log_image('exp(B)', A_hat, to_uint8=True, show_best=True)

        return loss


class MatrixIdempotent(Benchmark):
    """find an idempotent matrix close to A"""
    def __init__(self, A, n: int, chain:Literal['first', 'last'] | None = 'last', criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.B = torch.nn.Parameter(self.A / nuc)
        self.n = n
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.chain = chain

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A = self.A; B = self.B

        if self.chain == 'last': loss = 0
        else: loss = self.criterion(B, A)

        powers = []
        B_p = B
        for _ in range(1, self.n):
            B_prev = B_p
            B_p = algebras.matmul(B_p, B, self.algebra)
            if self.chain is not None: loss = loss + self.criterion(B_p, B_prev)
            else: loss = loss + self.criterion(B_p, A)

            if self._make_images: powers.append(B_p)

        if self.chain == 'last': loss = loss + self.criterion(B_p, A)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.n}', B_p, to_uint8=True, show_best=True)

        assert isinstance(loss, torch.Tensor)
        return loss


class StochasticMatrixIdempotent(Benchmark):
    def __init__(self, A, n: int, batch_size: int = 1, chain:Literal['first', 'last'] | None = 'last', criterion=torch.nn.functional.mse_loss, vec=False, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.B = torch.nn.Parameter(self.A / nuc)
        self.n = n
        self.chain = chain
        self.batch_size = batch_size
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.vec = vec

        self.add_reference_image('A', A, to_uint8=True)

    def pre_step(self):
        A = self.A
        if self.vec:
            *b, n, m = A.shape
            self.X = torch.randn((self.batch_size, *b, 1, m), device=A.device, dtype=A.dtype, generator=self.rng.torch(A.device))

        else:
            self.X = torch.randn((self.batch_size, *self.A.shape), device=A.device, dtype=A.dtype, generator=self.rng.torch(A.device))

    def get_loss(self):
        A = self.A; B = self.B; X = self.X

        powers = []
        B_p = B
        XA = algebras.matmul(X, A, self.algebra)
        XB = algebras.matmul(X, B_p, self.algebra)
        if self.chain == 'last': loss = 0
        else: loss = self.criterion(XB, XA)

        for _ in range(1, self.n):
            XB_prev = XB
            B_p = algebras.matmul(B_p, B, self.algebra)
            XB = algebras.matmul(X, B_p, self.algebra)
            if self.chain: loss = loss + self.criterion(XB, XB_prev)
            else: loss = loss + self.criterion(XB, XA)

            if self._make_images: powers.append(B_p)

        if self.chain == 'last':
            loss = loss + self.criterion(XB, XA)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.n}', B_p, to_uint8=True, show_best=True)

        assert isinstance(loss, torch.Tensor)
        return loss
