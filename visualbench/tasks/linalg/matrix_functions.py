import torch

from ...benchmark import Benchmark
from ...utils import algebras, to_CHW, to_square, totensor


class MatrixRoot(Benchmark):
    def __init__(self, A, p: int, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))

        # this keeps norm at 1 from applying matrix power
        nuc = torch.linalg.matrix_norm(self.A, 'nuc', keepdim=True) # pylint:disable=not-callable
        self.X = torch.nn.Parameter(self.A / nuc)
        self.p = p
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        X = self.X

        powers = []
        if self.algebra is None:
            Xp = torch.linalg.matrix_power(X, n=self.p) # pylint:disable=not-callable
        else:
            Xp = X
            for _ in range(1, self.p):
                Xp = self.algebra.matmul(X, Xp)
                if self._make_images: powers.append(Xp)

        loss = self.criterion(Xp, self.A)

        if self._make_images:
            self.log_image('X', self.X, to_uint8=True, log_difference=True)

            if len(powers) > 1:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'X^{i+2}', p, to_uint8=True)

            self.log_image(f'X^{self.p}', Xp, to_uint8=True)

        return loss

