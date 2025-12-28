import warnings
from collections.abc import Callable
from typing import Literal

import torch

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils


class QR(Benchmark):
    """Decompose rectangular A into QR, where Q is orthonormal and R is upper triangular, optionally with positive diagonal to make factorization unique.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce orthogonality of Q (float penalty or "qr" or "svd"). Defaults to 1.
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
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))
        self.min, self.max = self.A.min().item(), self.A.max().item()
        self.mode = mode
        self.criterion = criterion
        self.ortho: linalg_utils.OrthoMode = ortho
        self.algebra = algebras.get_algebra(algebra)
        self.exp_diag = exp_diag

        *b, m, n = self.A.shape
        k = min(m, n) if mode == 'reduced' else m
        self.Q = torch.nn.Parameter(linalg_utils.orthogonal((*b, m, k), generator=self.rng.torch()))
        self.R = torch.nn.Parameter(linalg_utils.orthogonal((*b, k, n), generator=self.rng.torch()))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                Q, R = torch.linalg.qr(self.A, mode=mode) # pylint:disable=not-callable
                self.add_reference_image('PyTorch Q', Q, to_uint8=True)
                self.add_reference_image('PyTorch R', R, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch QR failed: {e!r}')


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
            with torch.no_grad():
                self.log_image("Q", Q, to_uint8=True, log_difference=True)
                self.log_image("R", R, to_uint8=True, log_difference=True)
                self.log_image("QR", QR_, to_uint8=True, show_best=True, min=self.min, max=self.max)
                self.log_image("residual", (QR_ - A).abs_(), to_uint8=True)

        return loss + penalty



class SVD(Benchmark):
    """Decompose rectangular A into USV*, where U and V are orthonormal unitary, S is diagonal.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce unitarity of S and D (float penalty or "qr" or "svd"). Defaults to 1.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        ortho: linalg_utils.OrthoMode = 1,
        mode: Literal["full", "reduced"] = "reduced",
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):

        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))

        self.criterion = criterion
        self.ortho: linalg_utils.OrthoMode = ortho
        self.algebra = algebras.get_algebra(algebra)

        *b, self.m, self.n = self.A.shape
        k = min(self.m, self.n) if mode == 'reduced' else self.m

        self.U = torch.nn.Parameter(linalg_utils.orthogonal((*b, self.m, k), generator=self.rng.torch()))
        self.S = torch.nn.Parameter(torch.zeros(*b, k))
        self.V = torch.nn.Parameter(linalg_utils.orthogonal((*b, k, self.n), generator=self.rng.torch()))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                U, S, V = torch.linalg.svd(self.A) # pylint:disable=not-callable
                self.add_reference_image('PyTorch U', U, to_uint8=True)
                self.add_reference_image('PyTorch V', V, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch SVD failed: {e!r}')


    def get_loss(self):
        U = self.U
        S = self.S
        V = self.V

        U, penalty1 = linalg_utils.orthonormality_constraint(U, ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)
        V, penalty2 = linalg_utils.orthonormality_constraint(V, ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)
        V_star = V.mH

        US = algebras.mul(U, S.unsqueeze(-2), self.algebra) # same as U @ S.diag_embed()
        USV = algebras.matmul(US, V_star, self.algebra)

        loss = self.criterion(USV, self.A)

        if self._make_images:
            indices = torch.argsort(S**2, descending=True)
            U_sorted = torch.gather(U, 2, indices.unsqueeze(1).expand(-1, self.m, -1))
            Vh_sorted = torch.gather(V_star, 1, indices.unsqueeze(-1).expand(-1, -1, self.n))

            self.log_image("U", U_sorted, to_uint8=True)
            self.log_image("V", Vh_sorted, to_uint8=True)
            self.log_image("USV*", USV, to_uint8=True, show_best=True)

        return loss + penalty1 + penalty2

class Eigendecomposition(Benchmark):
    """Decompose square A into QΛQ^-1, where Q is a square matrix of eigenvectors (optionally orthonormal),
    Λ is a diagonal matrix with eigenvalues.

    This uses no complex values (but it is least squares-like).

    Args:
        A (Any): something to load and use as a matrix.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        invert_Q: if True, computes Q^-1 as inverse, if False, computes it as Q^T.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        criterion:Callable=torch.nn.functional.mse_loss,
        ortho: linalg_utils.OrthoMode = 1,
        invert_Q: bool = False,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_square(format.to_CHW(A, generator=self.rng.torch())).float())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.invert_Q = invert_Q
        self.ortho: linalg_utils.OrthoMode = ortho

        *b, self.n, self.n = self.A.shape
        self.Q = torch.nn.Parameter(torch.linalg.qr(self.A)[0]) # pylint:disable=not-callable
        self.L = torch.nn.Parameter(torch.zeros(*b, self.n))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                L, Q = torch.linalg.eigh(self.A) # pylint:disable=not-callable
                self.add_reference_image('PyTorch Q', Q.real, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch eigh failed: {e!r}')


    def get_loss(self):
        Q = self.Q
        L = self.L

        Q, penalty = linalg_utils.orthonormality_constraint(Q, self.ortho, self.algebra, self.criterion)

        if self.invert_Q:
            try:
                Q_inv = torch.linalg.inv(Q) # pylint:disable=not-callable
            except torch.linalg.LinAlgError:
                Q_inv = torch.linalg.pinv(Q) # pylint:disable=not-callable
        else:
            Q_inv = Q.mH

        QL = algebras.mul(Q, L.unsqueeze(-2), self.algebra) # same as Q @ L.diag_embed()
        QLQi = algebras.matmul(QL, Q_inv, self.algebra)

        loss = self.criterion(QLQi, self.A) + penalty

        if self._make_images:
            indices = torch.argsort(L**2, descending=True)
            Q_sorted = torch.gather(Q, 2, indices.unsqueeze(1).expand(-1, self.n, -1))

            self.log_image("Q", Q_sorted, to_uint8=True)
            self.log_image("QΛQ^-1", QLQi, to_uint8=True, show_best=True)

        return loss




class Cholesky(Benchmark):
    """Decompose square A in LL*, where L is a lower triangular matrix with real and positive diagonal entries.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce unitarity of S and D (float penalty or "qr" or "svd"). Defaults to 1.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_square(format.to_CHW(A, generator=self.rng.torch())).float())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        self.L = torch.nn.Parameter(torch.zeros_like(self.A)) # pylint:disable=not-callable

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                L = torch.linalg.cholesky_ex(self.A)[0] # pylint:disable=not-callable
                self.add_reference_image('PyTorch L', L, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch Cholesky failed: {e!r}')

    def get_loss(self):
        L = torch.tril(self.L, diagonal=1)
        L = L + self.L.diagonal(dim1=-2, dim2=-1).exp().diag_embed() # make diagonal positive

        LLh = algebras.matmul(L, L.mH, self.algebra)

        loss = self.criterion(LLh, self.A)

        if self._make_images:
            self.log_image("L", L, to_uint8=True)
            self.log_image("LL*", LLh, to_uint8=True, show_best=True)

        return loss



class LDL(Benchmark):
    """Decompose square A into LDL*, where L is a lower unit triangular matrix and D is a diagonal matrix.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce unitarity of S and D (float penalty or "qr" or "svd"). Defaults to 1.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_square(format.to_CHW(A, generator=self.rng.torch())).float())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        *b, self.n, self.n = self.A.shape
        self.L = torch.nn.Parameter(torch.zeros_like(self.A)) # pylint:disable=not-callable
        self.D = torch.nn.Parameter(torch.zeros(*b, self.n))
        self.I = torch.nn.Buffer(linalg_utils.eye_like(self.L))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                # this uses pivoting though
                LD = torch.linalg.ldl_factor_ex(self.A)[0] # pylint:disable=not-callable
                self.add_reference_image('PyTorch LD', LD, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch LDL failed: {e!r}')


    def get_loss(self):
        L = self.L.tril(-1) + self.I # lower unit triangular
        D = self.D

        LD = algebras.mul(L, D.unsqueeze(-2), self.algebra) # same as L @ D.diag_embed()
        LDLh = algebras.matmul(LD, L.mH, self.algebra)

        loss = self.criterion(LDLh, self.A)

        if self._make_images:
            self.log_image("L", L, to_uint8=True)
            self.log_image("LD", LD, to_uint8=True)
            self.log_image("LDL*", LDLh, to_uint8=True, show_best=True)

        return loss



class LU(Benchmark):
    """Decompose rectangular A into LU, where L is a lower triangular matrix and U is an upper triangular matrix. This one has no pivoting.

    Args:
        A (Any): something to load and use as a matrix.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        criterion:Callable=torch.nn.functional.mse_loss,
        mode = 'reduced',
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()).float())
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        *b, m, n = self.A.shape
        k = min(m, n) if mode == 'reduced' else m
        self.L = torch.nn.Parameter(torch.ones(*b, m, k))
        self.U = torch.nn.Parameter(torch.zeros(*b, k, n))
        self.I = torch.nn.Buffer(linalg_utils.eye_like(self.A))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                A = self.A.cuda() # LU without pivoting not implemented on CPU
                P, L, U = torch.linalg.lu(A, pivot=False) # pylint:disable=not-callable
                self.add_reference_image('PyTorch L', L.cpu(), to_uint8=True)
                self.add_reference_image('PyTorch U', U.cpu(), to_uint8=True)
            except Exception as e:
                warnings.warn(f'PyTorch LU failed: {e!r}')


    def get_loss(self):
        L = self.L.tril(-1) + self.I
        U = self.U.triu()

        LU_ = algebras.matmul(L, U, self.algebra)
        loss = self.criterion(LU_, self.A)

        if self._make_images:
            self.log_image("L", L, to_uint8=True, log_difference=True)
            self.log_image("U", U, to_uint8=True, log_difference=True)
            self.log_image("LU", LU_, to_uint8=True, show_best=True)

        return loss



class LUP(Benchmark):
    """Decompose rectangular A into PᵀLU, where L is a lower triangular matrix, U is an upper triangular matrix, P is a pivoting permutation matrix.


    Args:
        A (Any): something to load and use as a matrix.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        sinkhorn_iters: int | None = 5,
        ortho: linalg_utils.OrthoMode = 1,
        binary_weight: float = 1,
        criterion:Callable=torch.nn.functional.mse_loss,
        mode = 'full',
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()).float())
        self.ortho: linalg_utils.OrthoMode = ortho
        self.binary_weight = binary_weight
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.sinkhorn_iters = sinkhorn_iters

        *b, m, n = self.A.shape
        k = min(m, n) if mode == 'reduced' else m
        self.P = torch.nn.Parameter(torch.zeros(*b, m, m))
        self.L = torch.nn.Parameter(torch.ones(*b, m, k))
        self.U = torch.nn.Parameter(torch.zeros(*b, k, n))
        self.I = torch.nn.Buffer(linalg_utils.eye_like(self.A))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                P, L, U = torch.linalg.lu(self.A, pivot=True) # pylint:disable=not-callable
                self.add_reference_image('PyTorch P', P, to_uint8=True)
                self.add_reference_image('PyTorch L', L, to_uint8=True)
                self.add_reference_image('PyTorch U', U, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'PyTorch LU failed: {e!r}')


    def get_loss(self):
        P, penalty = linalg_utils.make_permutation(self.P, iters=self.sinkhorn_iters, binary_weight=self.binary_weight,
                                                   ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)
        L = self.L.tril(-1) + self.I
        U = self.U.triu()

        LU_ = algebras.matmul(L, U, self.algebra)
        #PLU = algebras.matmul(P.mT, LU_, self.algebra)
        PLU = P.mT @ LU_
        loss = self.criterion(PLU, self.A)

        if self._make_images:
            self.log_image("P raw", self.P, to_uint8=True, log_difference=True)
            self.log_image("P", P, to_uint8=True, log_difference=True)
            self.log_image("L", L, to_uint8=True, log_difference=True)
            self.log_image("U", U, to_uint8=True, log_difference=True)
            self.log_image("LU", LU_, to_uint8=True)
            self.log_image("PLU", PLU, to_uint8=True, show_best=True)

        return loss + penalty




class Polar(Benchmark):
    """Decompose square A into UP, where U is unitary and P is positive semi-definite Hermitian.
    In this benchmark P is stored as LL*, where L is a lower triagular matrix with positive diagonal.

    Args:
        A (Any): something to load and use as a matrix.
        ortho (linalg_utils.OrthoMode, optional): how to enforce orthogonality of Q (float penalty or "qr" or "svd"). Defaults to 1.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        ortho: linalg_utils.OrthoMode = 1,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_square(format.to_CHW(A, generator=self.rng.torch())).float())
        self.criterion = criterion
        self.ortho: linalg_utils.OrthoMode = ortho
        self.algebra = algebras.get_algebra(algebra)

        self.U = torch.nn.Parameter(linalg_utils.orthogonal_like(self.A, generator=self.rng.torch()))
        self.L = torch.nn.Parameter(torch.zeros_like(self.A))

        self.add_reference_image('A', self.A, to_uint8=True)
        if algebra is None:
            try:
                U, P = linalg_utils.polar(self.A) # pylint:disable=not-callable
                self.add_reference_image('PyTorch U', U, to_uint8=True)
                self.add_reference_image('PyTorch P', P, to_uint8=True)
            except torch.linalg.LinAlgError as e:
                warnings.warn(f'Polar via PyTorch SVD failed: {e!r}')

    def get_loss(self):
        L = torch.tril(self.L, diagonal=1)
        L = L + self.L.diagonal(dim1=-2, dim2=-1).exp().diag_embed() # make diagonal positive

        U, penalty = linalg_utils.orthonormality_constraint(self.U, ortho=self.ortho, algebra=self.algebra, criterion=self.criterion)
        P = algebras.matmul(L, L.mH, self.algebra)
        UP = algebras.matmul(U, P, self.algebra)

        loss = self.criterion(UP, self.A)

        if self._make_images:
            self.log_image("U", U, to_uint8=True)
            self.log_image("L", L, to_uint8=True)
            self.log_image("P", P, to_uint8=True)
            self.log_image("UP", UP, to_uint8=True, show_best=True)

        return loss + penalty


def _make_lowrank(A, rank, seed,):
    if isinstance(A, int):
        A = (A, A)

    if isinstance(A, tuple) and (len(A) in (2, 3)) and isinstance(A[0], int):
        A = linalg_utils.make_low_rank_tensor(A, rank=rank, seed=seed)

    return A

class RankFactorization(Benchmark):
    """Decompose rectangular A into CF, where C is (m, rank) and F is (rank, n).

    Args:
        A (Any): something to load and use as a matrix.
        rank (int): rank.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        rank: int,
        true_rank: int | None = None,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)

        if true_rank is None: true_rank = rank
        A = _make_lowrank(A, true_rank, self.rng.seed)
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))

        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        *b, m, n = self.A.shape
        self.C = torch.nn.Parameter(linalg_utils.orthogonal((*b, m, rank), generator=self.rng.torch()))
        self.F = torch.nn.Parameter(linalg_utils.orthogonal((*b, rank, n), generator=self.rng.torch()))

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        C = self.C
        F = self.F

        CF = algebras.matmul(C, F, self.algebra)
        loss = self.criterion(CF, self.A)

        if self._make_images:
            self.log_image("C", C, to_uint8=True)
            self.log_image("F", F, to_uint8=True)
            self.log_image("CF", CF, to_uint8=True, show_best=True)

        return loss



class NNMF(Benchmark):
    """Decompose nonnegtive rectangular V into product of two smaller nonnegative matrices WH.

    Args:
        A (Any): something to load and use as a matrix.
        rank (int): rank.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        rank: int,
        true_rank: int | None = None,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        if true_rank is None: true_rank = rank
        A = _make_lowrank(A, true_rank, self.rng.seed)

        A = format.to_CHW(A, generator=self.rng.torch())
        self.A = torch.nn.Buffer(A - A.amin().clip(max=0))

        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        *b, m, n = self.A.shape

        # make H @ W have unit norm
        W = torch.empty((*b, m, rank)).uniform_(0.1,1, generator=self.rng.torch())
        H = torch.empty((*b, rank, n)).uniform_(0.1,1, generator=self.rng.torch())
        norm = torch.linalg.norm(algebras.matmul(W, H, self.algebra)) # pylint:disable=not-callable
        scale = (1/norm).sqrt()

        self.W = torch.nn.Parameter((W * scale).log())
        self.H = torch.nn.Parameter((H * scale).log())

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        W = self.W.exp()
        H = self.H.exp()

        WH = algebras.matmul(W, H, self.algebra)
        loss = self.criterion(WH, self.A)

        if self._make_images:
            self.log_image("W", W, to_uint8=True,)
            self.log_image("H", H, to_uint8=True,)
            self.log_image("WH", WH, to_uint8=True, show_best=True)

        return loss


def _brute_find_closest_ab(x: int):
    """find two closest integers a,b such that a*b=x VIA BRUTEFORCE"""
    best_ab = None
    best_diff = float('inf')
    for a in range(x):
        for b in range(x):
            if a*b == x:
                diff = abs(a-b)
                if diff < best_diff:
                    best_diff = diff
                    best_ab = (a,b)

    if best_ab is None:
        raise RuntimeError(f"{x} is prime")

    return best_ab


class KroneckerFactorization(Benchmark):
    """Decompose rectangular A into B⊗C, B is m×n, C is p×q and A is pm×qn.

    Args:
        A (Any): something to load and use as a matrix.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_CHW(A, generator=self.rng.torch()))

        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

        *b, s1, s2 = self.A.shape

        m, p = _brute_find_closest_ab(s1)
        n, q = _brute_find_closest_ab(s2)

        self.B = torch.nn.Parameter(linalg_utils.orthogonal((*b, m, n), generator=self.rng.torch()))
        self.C = torch.nn.Parameter(linalg_utils.orthogonal((*b, p, q), generator=self.rng.torch()))

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        B = self.B
        C = self.C

        rec = torch.stack([algebras.kron(b, c, self.algebra) for b,c in zip(B, C)])
        loss = self.criterion(rec, self.A)

        if self._make_images:
            self.log_image("B", B, to_uint8=True, log_difference=True)
            self.log_image("C", C, to_uint8=True, log_difference=True)
            self.log_image("B⊗C", rec, to_uint8=True, show_best=True)

        return loss


