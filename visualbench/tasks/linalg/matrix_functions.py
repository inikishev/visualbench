import torch

from ...utils import totensor, to_CHW, get_algebra, to_square, from_algebra
from ...benchmark import Benchmark

class MatrixRoot(Benchmark):
    def __init__(self, A, p: int, criterion=torch.nn.functional.mse_loss, algebra=None, seed=0):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(to_square(to_CHW(A)))
        self.X = torch.nn.Parameter(torch.zeros_like(self.A))
        self.p = p
        self.criterion = criterion
        self.algebra = get_algebra(algebra)

        self.add_reference_image('A', A, to_uint8=True)

    def get_loss(self):
        X = self.X

        powers = []
        if self.algebra is None:
            Xp = torch.linalg.matrix_power(X, n=self.p) # pylint:disable=not-callable
        else:
            for _ in range(1, self.p):
                X = self.algebra.matmul(X,X)
                if self._make_images: powers.append(X)
            Xp = X

        loss = self.criterion(Xp, self.A)

        if self._make_images:
            self.log_image('X', self.X, to_uint8=True, log_difference=True)

            if len(powers) > 2:
                for i,p in enumerate(powers[:-1]):
                    self.log_image(f'X^{i+2}', p, to_uint8=True)

            self.log_image(f'X^{self.p}', Xp, to_uint8=True)

        return loss

