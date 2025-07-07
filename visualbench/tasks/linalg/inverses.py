import torch

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor


class Inverse(Benchmark):
    def __init__(self, A, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.I = torch.nn.Buffer(torch.eye(self.A.size(-1)).expand_as(self.A).clone())
        self.B = torch.nn.Parameter(self.I.clone())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A = self.A; B = self.B

        AB = algebras.matmul(A, B, self.algebra)
        BA = algebras.matmul(B, A, self.algebra)

        loss1 = self.criterion(AB, BA)
        loss2 = self.criterion(AB, self.I)
        loss3 = self.criterion(BA, self.I)

        loss = loss1+loss2+loss3

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            self.log_image('AB', AB, to_uint8=True)
            self.log_image('BA', BA, to_uint8=True)
            if self.algebra is None:
                A_inv_inv = torch.linalg.inv_ex(self.B)[0] # pylint:disable=not-callable
                self.log_image('B inverse', A_inv_inv, to_uint8=True)

        return loss


class StochasticInverse(Benchmark):
    """sample random x, update A_inv such that (x@A)@A_inv = x which converges to true inverse"""
    def __init__(self, A, batch_size = 1, criterion=torch.nn.functional.mse_loss, vec=False, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.B = torch.nn.Parameter(torch.eye(self.A.size(-1)).expand_as(self.A).clone())
        self.vec = vec
        self.batch_size = batch_size
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def pre_step(self):
        if self.vec:
            *b, n, m = self.A.shape
            self.X = torch.randn((self.batch_size, *b, m, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            self.X = torch.randn((self.batch_size, *self.A.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        X = self.X
        A = self.A.unsqueeze(0); B = self.B.unsqueeze(0)

        AX = algebras.matmul(A, X, self.algebra)
        X_hat = algebras.matmul(B, AX, self.algebra)

        loss = self.criterion(X, X_hat)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            if self.algebra is None:
                A_inv_inv = torch.linalg.inv_ex(self.B)[0] # pylint:disable=not-callable
                self.log_image('B inverse', A_inv_inv, to_uint8=True)

        return loss


class MoorePenrose(Benchmark):
    def __init__(self, A, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_CHW(A))
        self.B = torch.nn.Parameter(torch.eye(self.A.size(-1)).expand_as(self.A).clone())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A = self.A; B = self.B

        AB = algebras.matmul(A, B, self.algebra)
        BA = algebras.matmul(B, A, self.algebra)
        ABA = algebras.matmul(AB, A, self.algebra)
        BAB = algebras.matmul(BA, B, self.algebra)

        loss1 = self.criterion(ABA, self.A)
        loss2 = self.criterion(BAB, self.B)
        loss3 = self.criterion(AB, AB.mH)
        loss4 = self.criterion(BA, BA.mH)
        loss = loss1+loss2+loss3+loss4

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            self.log_image('AB', AB, to_uint8=True)
            self.log_image('BA', BA, to_uint8=True)
            self.log_image('ABA', ABA, to_uint8=True)
            self.log_image('BAB', BAB, to_uint8=True)

        return loss
