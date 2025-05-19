import torch

from ...utils import totensor, to_CHW, get_algebra, to_square, from_algebra
from ...benchmark import Benchmark

class Inverse(Benchmark):
    def __init__(self, A, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.I = torch.nn.Buffer(torch.eye(self.A.size(-1)).expand_as(self.A).clone())
        self.B = torch.nn.Parameter(torch.zeros_like(self.A))
        self.criterion = criterion
        self.algebra = get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A = self.A; B = self.B
        if self.algebra is not None: A, B = self.algebra.convert(A, B)

        AB = A @ B
        BA = B @ A

        AB, BA = from_algebra(AB, BA)

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
    """sample random x, update A_inv such that (x@A)@A_inv = x"""
    def __init__(self, A, batch_size = 1, criterion=torch.nn.functional.mse_loss, vec=False, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.B = torch.nn.Parameter(torch.zeros_like(self.A))
        self.vec = vec
        self.batch_size = batch_size
        self.criterion = criterion
        self.algebra = get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        if self.vec:
            *b, n, m = self.A.shape
            x = torch.randn((self.batch_size, *b, 1, m), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            x = torch.randn((self.batch_size, *self.A.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        A = self.A.unsqueeze(0); B = self.B.unsqueeze(0)
        if self.algebra is not None: x, A, B = self.algebra.convert(x, A, B)

        x_hat = (x @ self.A.unsqueeze(0)) @ self.B.unsqueeze(0)
        x, x_hat = from_algebra(x, x_hat)

        loss = self.criterion(x, x_hat)

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
        self.B = torch.nn.Parameter(torch.zeros_like(self.A))
        self.criterion = criterion
        self.algebra = get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        A = self.A; B = self.B
        if self.algebra is not None: A, B = self.algebra.convert(A, B)

        AB = A @ B
        BA = B @ A
        ABA = AB @ A
        BAB = B @ AB

        AB, BA, ABA, BAB = from_algebra(AB, BA, ABA, BAB)

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
            # if self.algebra is None:
            #     try:
            #         B_pinv = torch.linalg.pinv(self.B) # pylint:disable=not-callable
            #         self.log_image('B pseudoinverse', B_pinv, to_uint8=True)

            #     except torch.linalg.LinAlgError:
            #         self.log_image('B pseudoinverse', torch.zeros_like(self.A), to_uint8=True)

        return loss
