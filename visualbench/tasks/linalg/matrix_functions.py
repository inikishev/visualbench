from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor
from . import linalg_utils


class MatrixRoot(Benchmark):
    """The objective is to find B such that B^P = A, where A is a square matrix."""
    def __init__(self, A, p: int, criterion:Callable=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))
        self.min = self.A.min().item(); self.max = self.A.max().item()

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

            self.log_image(f'B^{self.p}', B_p, to_uint8=True, show_best=True, min=self.min, max=self.max)
            self.log_image('residual', (B_p-self.A).abs_(), to_uint8=True)

        return loss


class StochasticMatrixRoot(Benchmark):
    """The objective is to find B such that (B^P)x = Ax, where A is a square matrix and x is a random vector."""
    def __init__(self, A, p: int, batch_size: int = 1, criterion:Callable=torch.nn.functional.mse_loss, vec=True, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))
        self.min = self.A.min().item(); self.max = self.A.max().item()

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
            b, n, m = self.A.shape
            self.X = torch.randn((self.batch_size, b, 1, m), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

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

        with torch.no_grad():
            test_loss = self.criterion(B_p, self.A)
            self.log('test loss', test_loss)

            if self._make_images:
                self.log_image('B', self.B, to_uint8=True, log_difference=True)

                if len(powers) > 1:
                    for i,p in enumerate(powers[:-1]):
                        self.log_image(f'B^{i+2}', p, to_uint8=True)

                self.log_image(f'B^{self.p}', B_p, to_uint8=True, show_best=True, min=self.min, max=self.max)
                self.log_image('residual', (B_p-self.A).abs_(), to_uint8=True)

        return loss

class MatrixLogarithm(Benchmark):
    """The objective is to find B such that exp(B) = A, where A is a square matrix."""
    def __init__(self, A, criterion:Callable=torch.nn.functional.mse_loss):
        super().__init__()
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))
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
    """The objective is to find B such that B^n = A, and for each n from 2 to n, B^(n-1) = B^n.

    In other words the goal is to find a matrix close to A that doesn't change when multipied by itself.
    This is a very ill-conditioned objective and only SOAP appears to be able to solve it with large ``n``.

    Args:
        A (Any, optional): square matrix.
        n (int): n.
        chain (str | None, optional):
            if None, calculates criterion between B^n and A. If not None, calculates criterion between B^(n-1) = B^n.
            - "first" - B should be equal to A.
            - "last" - B^n should be equal to A (hardest one and the default).
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): algebra like "tropical" or None for elementary. Defaults to None.
        seed (int, optional): random seed. Defaults to 0.

    """
    def __init__(self, A, n: int, chain:Literal['first', 'last'] | None = 'last', criterion:Callable=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.B = torch.nn.Parameter(self.A / nuc)
        self.n = n
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.chain = chain

        self.add_reference_image('A', A, to_uint8=True)
        self.set_multiobjective_func(torch.sum)

    def get_loss(self):
        A = self.A; B = self.B

        if self.chain == 'last': losses = []
        else: losses = [self.criterion(B, A)]

        powers = []
        B_p = B
        for _ in range(1, self.n):
            B_prev = B_p
            B_p = algebras.matmul(B_p, B, self.algebra)
            if self.chain: losses.append(self.criterion(B_p, B_prev))
            else: losses.append(self.criterion(B_p, A))

            if self._make_images: powers.append(B_p)

        if self.chain == 'last': losses.append(self.criterion(B_p, A))

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.n}', B_p, to_uint8=True, show_best=True)

        return torch.stack(losses)


class StochasticMatrixIdempotent(Benchmark):
    """The objective is to find B such that (B^n)x = (A)x, and for each n from 2 to n, (B^(n-1))x = (B^n)x, where x is a random vector.

    In other words the goal is to find a matrix close to A that doesn't change when multipied by itself.
    This is an extremely hard objective.

    Args:
        A (Any, optional): square matrix.
        n (int): n.
        batch_size (int, optional): batch size, defaults to 1.
        chain (str | None, optional):
            if None, calculates criterion between B^n and A. If not None, calculates criterion between B^(n-1) = B^n.
            - "first" - B should be equal to A.
            - "last" - B^n should be equal to A (hardest one and the default).
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        vec (bool, optional): if True, x is a vector, otherwise it is a matrix, making the objective a bit easier.
        algebra (Any, optional): algebra like "tropical" or None for elementary. Defaults to None.
        seed (int, optional): random seed. Defaults to 0.

    """
    def __init__(self, A, n: int, batch_size: int = 1, chain:Literal['first', 'last'] | None = 'last', criterion=torch.nn.functional.mse_loss, vec=True, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))

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
        self.set_multiobjective_func(torch.sum)

    def pre_step(self):
        A = self.A
        if self.vec:
            b, n, m = A.shape
            self.X = torch.randn((self.batch_size, b, 1, m), device=A.device, dtype=A.dtype, generator=self.rng.torch(A.device))

        else:
            self.X = torch.randn((self.batch_size, *self.A.shape), device=A.device, dtype=A.dtype, generator=self.rng.torch(A.device))

    def get_loss(self):
        A = self.A; B = self.B; X = self.X

        powers = []
        B_p = B
        XA = algebras.matmul(X, A, self.algebra)
        XB = algebras.matmul(X, B_p, self.algebra)

        if self.chain == 'last': losses = test_losses = []
        else: losses = test_losses = [self.criterion(XB, XA)]

        for _ in range(1, self.n):
            B_prev = B_p
            XB_prev = XB
            B_p = algebras.matmul(B_p, B, self.algebra)
            XB = algebras.matmul(X, B_p, self.algebra)

            if self.chain:
                losses.append(self.criterion(XB, XB_prev))
                with torch.no_grad(): test_losses.append(self.criterion(B_p, B_prev))
            else:
                losses.append(self.criterion(XB, XA))
                with torch.no_grad(): test_losses.append(self.criterion(B_p, A))

            if self._make_images: powers.append(B_p)

        if self.chain == 'last':
            losses.append(self.criterion(XB, XA))
            with torch.no_grad(): test_losses.append(self.criterion(B_p, A))

        self.log('test loss', torch.stack(test_losses).sum())

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'B^{i+2}', p, to_uint8=True)

            self.log_image(f'B^{self.n}', B_p, to_uint8=True, show_best=True)

        return torch.stack(losses)


class StochasticMatrixSign(Benchmark):
    """given square A, find B such that Bx = Ax / ||Ax||, where x is a random unit vector sampled on each step,
    it doesn't hold that sgn(A)x = Ax / ||Ax|| but I am pretty sure matrix sign minimizes this loss"""
    def __init__(self, A, batch_size=1, criterion = F.mse_loss, algebra=None, sampler=torch.randn):
        super().__init__()

        self.A = torch.nn.Buffer(to_square(A, generator=self.rng.torch()))
        self.sign = torch.nn.Buffer(linalg_utils.matrix_sign_svd(self.A))

        self.batch_size = batch_size
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.sampler = sampler

        self.B = torch.nn.Parameter(self.A.clone())
        b, n, m = self.A.shape
        self.I = torch.nn.Buffer(linalg_utils.beye((b, n, n)))

        self.add_reference_image('A', A, to_uint8=True)
        self.add_reference_image('true sign', self.sign, to_uint8=True)

    def pre_step(self):
        b, n, m = self.A.shape
        self.x = self.sampler((self.batch_size, b, m, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        A = self.A.unsqueeze(0)
        B = self.B.unsqueeze(0)
        x = self.x / torch.linalg.vector_norm(self.x, dim=(-2,-1), keepdim=True).clip(min=1e-12) # pylint:disable=not-callable

        Ax = algebras.matmul(A, x, self.algebra)
        Bx = algebras.matmul(B, x, self.algebra)

        loss = self.criterion(Bx, Ax / torch.linalg.vector_norm(Ax).clip(min=1e-12)) # pylint:disable=not-callable

        with torch.no_grad():
            self.log("test loss", self.criterion(self.B, self.sign))
            if self._make_images:
                self.log_image("B", self.B, to_uint8=True, show_best=True, log_difference=True)

        return loss

