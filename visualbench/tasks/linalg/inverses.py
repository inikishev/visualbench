from collections.abc import Sequence

import torch

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor
from .linalg_utils import eye_like, row_sampler

class Inverse(Benchmark):
    """For a square ``A``, the objective is to find ``B`` such that ``AB = BA = I``."""
    def __init__(self, A, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))
        self.min = self.A.min(); self.max = self.A.max()

        self.I = torch.nn.Buffer(eye_like(self.A))
        self.B = torch.nn.Parameter(self.I.clone())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)
        self.set_multiobjective_func(torch.sum)

    def get_loss(self):
        A = self.A; B = self.B

        AB = algebras.matmul(A, B, self.algebra)
        BA = algebras.matmul(B, A, self.algebra)

        loss1 = self.criterion(AB, BA)
        loss2 = self.criterion(AB, self.I)
        loss3 = self.criterion(BA, self.I)

        # prevents loss close to 0 when matrices are just zeros
        if self.algebra is None and AB.amax().maximum(BA.amax()) < torch.finfo(AB.dtype).eps:
            loss1 = loss1 * torch.inf


        if self._make_images:
            with torch.no_grad():
                self.log_image('B', self.B, to_uint8=True, log_difference=True)
                self.log_image('AB', AB, to_uint8=True)
                self.log_image('BA', BA, to_uint8=True)
                if self.algebra is None:
                    B_inv = torch.linalg.inv_ex(self.B)[0] # pylint:disable=not-callable
                    self.log_image('B inverse', B_inv, to_uint8=True, show_best=True, min=self.min, max=self.max)
                    self.log_image('residual', (B_inv - self.A).abs_(), to_uint8=True)

        return torch.stack([loss1, loss2, loss3])


class StochasticInverse(Benchmark):
    """For a square ``A``, the objective is to find it's inverse ``B``
    by minimizing ``criterion(ABx, x) + criterion(BAx, x)``,
    where ``x`` is a random vector sampled before each step.
    If ``Ax = v`` then ``Bv = BAx = x`` (left inverse), and
    if ``Av = x``, then ``Bx = v`` (right inverse)."""
    def __init__(self, A, batch_size = 1, criterion=torch.nn.functional.mse_loss, sampler=row_sampler, vec=True, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A, generator=self.rng.torch())))
        self.min = self.A.min().item(); self.max = self.A.max().item()

        self.I = torch.nn.Buffer(eye_like(self.A))
        self.B = torch.nn.Parameter(self.I.clone())
        self.vec = vec
        self.batch_size = batch_size
        self.criterion = criterion
        self.sampler = sampler
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)
        self.set_multiobjective_func(torch.sum)

    def pre_step(self):
        if self.vec:
            b, n, n = self.A.shape # pylint:disable=redeclared-assigned-name
            self.x = self.sampler((self.batch_size, b, n, 1), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

        else:
            self.x = self.sampler((self.batch_size, *self.A.shape), device=self.A.device, dtype=self.A.dtype, generator=self.rng.torch(self.A.device))

    def get_loss(self):
        A = self.A.unsqueeze(0); B = self.B.unsqueeze(0); x = self.x

        # Ax=v so Bv=x
        Ax = algebras.matmul(A, x, self.algebra)
        BAx = algebras.matmul(B, Ax, self.algebra)
        loss_left = self.criterion(BAx, x)

        # Bx=v so Av=x
        Bx = algebras.matmul(B, x, self.algebra)
        ABx = algebras.matmul(A, Bx, self.algebra)
        loss_right = self.criterion(ABx, x)

        with torch.no_grad():
            # --------------------------------- test loss -------------------------------- #
            AB = algebras.matmul(self.A, self.B, self.algebra)
            BA = algebras.matmul(self.B, self.A, self.algebra)

            loss1 = self.criterion(AB, BA)
            loss2 = self.criterion(AB, self.I)
            loss3 = self.criterion(BA, self.I)

            # prevents loss close to 0 when matrices are just zeros
            if self.algebra is None and  AB.amax().maximum(BA.amax()) < torch.finfo(AB.dtype).eps:
                loss2 = loss3 = float('inf')
                with torch.enable_grad():
                    loss_left = loss_right = loss_left*torch.inf

            self.log('test loss', loss1+loss2+loss3)

            # ---------------------------------- images ---------------------------------- #
            if self._make_images:
                with torch.no_grad():
                    self.log_image('B', self.B, to_uint8=True, log_difference=True)
                    self.log_image('AB', AB, to_uint8=True)
                    self.log_image('BA', BA, to_uint8=True)
                    if self.algebra is None:
                        B_inv = torch.linalg.inv_ex(self.B)[0] # pylint:disable=not-callable
                        self.log_image('B inverse', B_inv, to_uint8=True, show_best=True) # removed min and max due to inexact inverse
                        self.log_image('residual', (B_inv - self.A).abs_(), to_uint8=True)

        return torch.stack([loss_left, loss_right])


class MoorePenrose(Benchmark):
    """Given rectangular ``A``, the Moore-Penrose inverse ``B`` is a matrix that satisfies following criteria:

    1. ``ABA = A``
    2. ``BAB = B``
    3. ``(AB)* = AB`` (``AB`` is Hermitian)
    4. ``(BA)* = BA`` (``BA`` is Hermitian)
    """
    def __init__(self, A, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_CHW(A, generator=self.rng.torch()))
        self.B = torch.nn.Parameter(eye_like(self.A))
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)
        self.set_multiobjective_func(torch.sum)

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

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            self.log_image('AB', AB, to_uint8=True)
            self.log_image('BA', BA, to_uint8=True)
            self.log_image('ABA', ABA, to_uint8=True)
            self.log_image('BAB', BAB, to_uint8=True)

        return torch.stack([loss1,loss2,loss3,loss4])


def matrix_index(A:torch.Tensor) -> int:
    k = 2
    rank = torch.linalg.matrix_rank(A) # pylint:disable=not-callable
    while True:
        rank_k = torch.linalg.matrix_rank(torch.linalg.matrix_power(A, k)) # pylint:disable=not-callable
        if rank_k == rank: return k
        rank = rank_k
        k += 1

def algebraic_matrix_power(A:torch.Tensor, power: int, algebra):
    if algebra is None: return torch.linalg.matrix_power(A, power) # pylint:disable=not-callable

    Ap = A
    for i in range(power-1):
        Ap = algebras.matmul(Ap, A, algebra=algebra)

    return Ap

class Drazin(Benchmark):
    """Given square ``A``, the index of ``A`` is defined as the
    least nonnegative integer ``k`` such that ```rank(A^(k+1)) = rank(A^k)```.
    In other words it is the power where rank stops changing.

    The Drazin inverse ``B`` of ``A`` is a unique matrix that satisfies the following criteria:

    1. ``A^(k+1) B = A^k``
    2. ``BAB = B``
    3. ``AB = BA``

    Note:
        If index is large, due to large powers float64 precision may be necessary to avoid infinities.

    Note:
        Index calculation won't be correct with non-elementary algebras (it still uses normal rank and power).
    """
    def __init__(self, A, ind: int | Sequence[int] | None = None, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_CHW(A, generator=self.rng.torch()))
        self.B = torch.nn.Parameter(eye_like(self.A))
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        if ind is None:
            ind = [matrix_index(M) for M in self.A]

        if isinstance(ind, int): ind = [ind for _ in self.A]
        self.ind = ind

        # pre-compute the powers (each channel is treated separate matrix with separate index)
        self.Ak = torch.nn.Buffer(
            torch.stack([algebraic_matrix_power(M, i, algebra=self.algebra) for M, i in zip(self.A, ind)])
        )
        self.Ak1 = torch.nn.Buffer(algebras.matmul(self.Ak, self.A, algebra=self.algebra))

        self.add_reference_image('A', A, to_uint8=True)
        self.set_multiobjective_func(torch.sum)

    def get_loss(self):
        A = self.A; B = self.B; Ak = self.Ak; Ak1 = self.Ak1

        Ak1B = algebras.matmul(Ak1, B, self.algebra)
        AB = algebras.matmul(A, B, self.algebra)
        BA = algebras.matmul(B, A, self.algebra)
        BAB = algebras.matmul(BA, B, self.algebra)

        loss1 = self.criterion(Ak1B, Ak)
        loss2 = self.criterion(BAB, self.B)
        loss3 = self.criterion(AB, BA)

        if self._make_images:
            self.log_image('B', self.B, to_uint8=True, log_difference=True)
            self.log_image('AB', AB, to_uint8=True)
            self.log_image('BA', BA, to_uint8=True)
            self.log_image('BAB', BAB, to_uint8=True)

        return torch.stack([loss1,loss2,loss3])

