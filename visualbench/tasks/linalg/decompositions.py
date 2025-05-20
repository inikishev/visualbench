from collections.abc import Callable
from typing import Literal
import warnings
import torch

from ...benchmark import Benchmark
from ...utils import format, algebras
from . import linalg_utils


class QR(Benchmark):
    """Decompose A into orthonormal Q and upper triangular R.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce orthogonality of Q (float penalty or "svd"). Defaults to 1.
        exp_diag (bool, optional): if True, applies exp to R diagonal to make it positive. Defaults to False.
        mode (Literal[str]], optional): "full" or "reduced". Defaults to "reduced".
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        ortho: linalg_utils.OrthoMode = 1,
        exp_diag: bool = False,
        mode: Literal["full", "reduced"] = "reduced",
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):

        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A))

        self.mode = mode
        self.criterion = criterion
        self.ortho: linalg_utils.OrthoMode = ortho
        self.algebra = algebras.get_algebra(algebra)
        self.exp_diag = exp_diag

        *b, m, n = self.A.shape
        k = min(m, n) if mode == 'reduced' else m
        self.Q = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.zeros(*b, m, k), generator=self.rng.torch()))
        self.R = torch.nn.Parameter(torch.zeros(*b, k, n))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                Q, R = torch.linalg.qr(self.A, mode=mode) # pylint:disable=not-callable
                self.add_reference_image('true Q', Q, to_uint8=True)
                self.add_reference_image('true R', R, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'true QR failed for some reason: {e!r}')


    def get_loss(self):
        A = self.A
        Q = self.Q

        if self.exp_diag:
            R = torch.triu(self.R, diagonal=1)
            R = R + self.R.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
        else:
            R = torch.triu(self.R)

        Q, penalty = linalg_utils.orthonormality_constraint(Q, ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)

        QR_ = algebras.matmul(Q, R, self.algebra)
        loss = self.criterion(QR_, A)

        if self._make_images:
            self.log_image("Q", self.Q, to_uint8=True, log_difference=True)
            self.log_image("R", R, to_uint8=True, log_difference=True)
            self.log_image("QR", QR_, to_uint8=True, show_best=True)

        return loss + penalty