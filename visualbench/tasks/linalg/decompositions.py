from typing import Literal
import warnings
import torch

from ...benchmark import Benchmark
from ...utils import format, algebras
from . import linalg_utils


class QR(Benchmark):
    def __init__(
        self,
        A,
        ortho: linalg_utils.OrthoMode = 1,
        mode: Literal["full", "reduced"] = "reduced",
        criterion=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A))

        self.mode = mode
        self.criterion = criterion
        self.ortho: linalg_utils.OrthoMode = ortho
        self.algebra = algebras.get_algebra(algebra)

        *b, m, n = self.A.shape
        k = min(m, n) if mode == 'reduced' else m
        self.Q = torch.nn.Parameter(torch.zeros(*b, m, k))
        self.R = torch.nn.Parameter(torch.zeros(*b, k, n))

        self.add_reference_image('A', self.A, to_uint8=True)
        try:
            Q, R = torch.linalg.qr(self.A, mode=mode) # pylint:disable=not-callable
            self.add_reference_image('true Q', Q, to_uint8=True)
            self.add_reference_image('true R', R, to_uint8=True)
        except torch.linalg.LinAlgError as e:
            warnings.warn(f'true QR failed for some reason: {e!r}')


    def get_loss(self):
        A = self.A
        Q = self.Q
        R = torch.triu(self.R)

        print(f'{Q.shape = }, {R.shape = }')
        QR_ = algebras.matmul(Q, R, self.algebra)
        loss = self.criterion(QR_, A)

        Q, penalty = linalg_utils.orthonormality_constraint(Q, ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)

        if self._make_images:
            self.log_image("Q", self.Q, to_uint8=True, log_difference=True)
            self.log_image("R", self.R, to_uint8=True, log_difference=True)
            self.log_image("QR", QR_, to_uint8=True, show_best=True)

        return loss + penalty