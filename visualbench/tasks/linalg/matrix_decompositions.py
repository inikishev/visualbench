# pylint: disable = not-callable, redefined-outer-name
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from myai.transforms import normalize, totensor
from torch import nn
from torch.nn import functional as F

from ..._utils import (
    _make_float_chw_square_matrix,
    _make_float_chw_tensor,
    _make_float_tensor,
    _normalize_to_uint8,
    sinkhorn,
)
from ...benchmark import Benchmark
from ._linalg_utils import _expand_channels, _square, _zeros, _full01, _full001, _ones, _normal01



# Matches true SVD with randn not with zeros
# randn and 0.01 produce similar results, ones slightly harder
# keep randn?
class SVD(Benchmark):
    """SVD as objective
    Args:
        A (Any): input matrix
        ortho_weight (_type_, optional): orthogonality loss weight. Defaults to 1..
        non_negative_fn (_type_, optional): function to ensure non-negativity, can also try sigmoid or softplus or something. Defaults to _square.
        make_images (bool, optional): saves images for plotting and video rednering. Defaults to True.
    """

    def __init__(
        self,
        A: Any,
        ortho_weight=1.0,
        loss: Callable = F.mse_loss,
        ortho_loss: Callable = F.mse_loss,
        non_negative_fn: Callable = _square,
        init: Callable=torch.randn,
        make_images=True,
    ):
        super().__init__(log_projections = True, seed=0)
        self._make_images = make_images
        self.non_negative_fn = non_negative_fn
        self.ortho_weight = ortho_weight
        self.loss = loss
        self.ortho_loss = ortho_loss

        matrix: torch.Tensor = _make_float_chw_tensor(A)
        self.A = torch.nn.Buffer(matrix.contiguous())

        b, self.m, self.n = matrix.shape
        self.k = min(self.m,self.n)
        self.I_k = torch.nn.Buffer(torch.eye(self.k).unsqueeze(0).repeat_interleave(b, 0).contiguous())

        # Initialize parameters with orthogonal matrices
        self.U = torch.nn.Parameter(torch.empty(b, self.m, self.k).contiguous())
        torch.nn.init.orthogonal_(self.U, generator=self.rng.torch())

        self.s_raw = torch.nn.Parameter(init((b, self.k), generator = self.rng.torch()))  # S = s_raw^2 ensures non-negativity

        self.V = torch.nn.Parameter(torch.empty(b, self.n, self.k).contiguous())
        torch.nn.init.orthogonal_(self.V, generator=self.rng.torch())

        # real SVD outputs for reference
        if make_images:
            self.add_reference_image('input', matrix)
            try:
                U, S, Vh = torch.linalg.svd(matrix, full_matrices=False) # pylint:disable = not-callable
                self.add_reference_image('U - true', U)
                self.add_reference_image('Vh - true', Vh)
            except Exception as e:
                print(f"true SVD decomposition failed: {e!r}")

            self.set_display_best('image SVD', True)


    def get_loss(self):
        S = self.non_negative_fn(self.s_raw)

        S_mat = torch.diag_embed(S)

        Vh = self.V.swapaxes(-1, -2)

        rec = self.U @ S_mat @ Vh
        recon_loss = self.loss(self.A, rec)

        # orthogonality loss
        U_ortho_loss = self.ortho_loss(self.U.swapaxes(-1,-2) @ self.U, self.I_k)
        V_ortho_loss = self.ortho_loss(Vh @ self.V, self.I_k)

        if self._make_images:
            S_sorted, indices = torch.sort(S**2, descending=True)
            U_sorted = torch.gather(self.U, 2, indices.unsqueeze(1).expand(-1, self.m, -1))
            Vh_sorted = torch.gather(Vh, 1, indices.unsqueeze(-1).expand(-1, -1, self.n))
            self.log('image reconstruction', rec, False, to_uint8=True)
            self.log('image U', U_sorted, False, to_uint8=True)
            self.log('image Vh',Vh_sorted, False, to_uint8=True)
            # self.log_difference('image update U', U_sorted, to_uint8=True)
            # self.log_difference('image update Vh', Vh_sorted, to_uint8=True)

        return recon_loss + self.ortho_weight * (U_ortho_loss + V_ortho_loss)

# Matches true QR with zeros, harder with randn
class QR(Benchmark):
    """QR as objective

    Args:
        A (Any): input matrix
        ortho_weight (_type_, optional): orthogonality loss weight. Defaults to 1..
        non_negative_fn (_type_, optional): function to ensure non-negativity, can also try sigmoid or softplus or something. Defaults to _square.
        make_images (bool, optional): saves images for plotting and video rednering. Defaults to True.
    """

    def __init__(
        self,
        A: Any,
        ortho_weight=1.0,
        loss: Callable = F.mse_loss,
        ortho_loss: Callable = F.mse_loss,
        init: Callable = _zeros,
        make_images=True,
    ):
        super().__init__(log_projections = True, seed=0)
        self._make_images = make_images
        self.loss = loss
        self.ortho_loss=ortho_loss
        self.ortho_weight = ortho_weight

        matrix: torch.Tensor = _make_float_chw_tensor(A)
        self.A = torch.nn.Buffer(matrix)

        b, m, n = matrix.shape
        self.k = min(m,n)
        self.I_k = torch.nn.Buffer(torch.eye(self.k).contiguous())

        # Initialize parameters with orthogonal matrices
        self.Q = torch.nn.Parameter(torch.empty(b, m, self.k).contiguous())
        torch.nn.init.orthogonal_(self.Q, generator=self.rng.torch())

        self.R = torch.nn.Parameter(init((b, self.k, n), generator = self.rng.torch()).triu().contiguous())

        identity = torch.eye(self.Q.size(-1), dtype=self.Q.dtype)
        identity = identity.unsqueeze(0).expand(self.Q.size(0), -1, -1)  # Match batch dimension
        self.identity = nn.Buffer(identity.contiguous())

        # real QR outputs for reference
        if make_images:
            self.add_reference_image('input', matrix)
            try:
                Q, R = torch.linalg.qr(matrix) # pylint:disable = not-callable
                self.add_reference_image('Q - true', Q)
                self.add_reference_image('R - true', R)
            except Exception as e:
                print(f"true QR decomposition failed: {e!r}")

            self.set_display_best('image QR', True)

        self.ones = torch.nn.Buffer(torch.ones_like(self.R, dtype=torch.bool).contiguous())

    def get_loss(self):
        R = self.R.triu()

        QR = self.Q @ R
        recon_loss = self.loss(self.A, QR)

        # orthogonality constraint: ||Q^T Q - I||_F^2
        QtQ = torch.bmm(self.Q.transpose(1, 2), self.Q)

        ortho_loss = self.ortho_loss(QtQ, self.identity)
        # orthogonality loss
        # ortho_loss = torch.linalg.norm(self.Q.swapaxes(-1,-2) @ self.Q - self.I_k, ord='fro', dim = (-2,-1)).mean()

        if self._make_images:
            self.log('image QR', QR, False, to_uint8=True)
            self.log('image Q', self.Q, False, to_uint8=True)
            self.log('image R', R, False, to_uint8=True)
            self.log_difference('image update Q', self.Q, to_uint8=True)
            self.log_difference('image update R', R, to_uint8=True)

        return recon_loss + ortho_loss*self.ortho_weight


# easier and less noisy with zeros
class LU(Benchmark):
    """LU as objective"""
    def __init__(self, A, loss = F.mse_loss, init = _zeros, make_images = True):
        super().__init__(log_projections = True, seed=0)
        matrix = _make_float_chw_tensor(A)

        self.A = torch.nn.Buffer(matrix.contiguous())
        b, m, n = matrix.shape
        k = min(m, n)

        self.L = nn.Parameter(init((b, m, k), generator=self.rng.torch()) * torch.tril(torch.ones(m, k), diagonal=-1).unsqueeze(0).contiguous())
        self.U = nn.Parameter(init((b, k, n), generator=self.rng.torch()) * torch.triu(torch.ones(k, n)).unsqueeze(0).contiguous())

        self.I = nn.Buffer(torch.eye(m, k, dtype = matrix.dtype).unsqueeze(0).contiguous())
        self.loss_fn = loss

        # real SVD outputs for reference
        self._make_images = make_images
        if make_images:
            self.add_reference_image('input', matrix)
            try:
                _, L, U = torch.linalg.lu(matrix.cuda(), pivot=False) # pylint:disable = not-callable
                self.add_reference_image('L - true', L)
                self.add_reference_image('U - true', U)
            except Exception as e:
                print(f"true LU decomposition failed: {e!r}")

            self.set_display_best('image LU', True)

    def get_loss(self):
        A = self.A
        L = torch.tril(self.L, diagonal=-1) + self.I  # shape (b, m, m)
        U = torch.triu(self.U, diagonal=0)
        LU = torch.bmm(L, U)  # shape (b, m, m)
        loss = self.loss_fn(LU, A)

        if self._make_images:
            self.log('image LU', LU, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)
            self.log_difference('image update U', U, to_uint8=True)


        return loss



# zeros and normal - same results, zeros looks better
class LUPivot(Benchmark):
    """LU with pivoting with P represented via sinkhorn iteration or by simple penalty if sinkhorn_iters is None.
    One of the few linalg objectives where L-BFGS is not the best."""
    def __init__(self, A, sinkhorn_iters: int | None=10, ortho_weight = 1, binary_weight = 1, loss=F.mse_loss, init = _zeros, make_images = True):

        super().__init__(log_projections = True, seed=0)
        self.A = torch.nn.Buffer(_make_float_chw_tensor(A))
        B, M, N = self.A.shape
        K = min(M, N)
        self.loss = loss

        self.L = nn.Parameter(init((B, M, K), generator=self.rng.torch()).contiguous())
        self.U = nn.Parameter(init((B, K, N), generator=self.rng.torch()).contiguous())
        self.P_logits = nn.Parameter(init((B, M, M), generator=self.rng.torch()).contiguous())
        self.L_mask = nn.Buffer(torch.tril(torch.ones(M, K), diagonal=-1).unsqueeze(0).contiguous())
        self.diag_mask = nn.Buffer(torch.eye(M, K).unsqueeze(0).contiguous())  # (1, M, K)
        self.U_mask = nn.Buffer(torch.triu(torch.ones(K, N)).unsqueeze(0).contiguous())  # (1, K, N)
        self.identity = nn.Buffer(torch.eye(M).unsqueeze(0).repeat_interleave(B, 0).contiguous())

        self.sinkhorn_iters = sinkhorn_iters
        self._make_images = make_images
        self.ortho_weight = ortho_weight
        self.binary_weight = binary_weight

        if make_images:
            self.add_reference_image('input', self.A)
            try:
                P, L, U = torch.linalg.lu(self.A) # pylint:disable = not-callable
                self.add_reference_image('P - true', P)
                self.add_reference_image('L - true', L)
                self.add_reference_image('U - true', U)
            except Exception as e:
                print(f"true LU decomposition failed: {e!r}")

            self.set_display_best('image LU', True)

    def get_loss(self):
        A = self.A

        if self.sinkhorn_iters is None: P = torch.softmax(self.P_logits, dim=-1)  # (b m m)
        else: P = sinkhorn(self.P_logits, self.sinkhorn_iters)

        PA = torch.bmm(P, A)  # (b m n)
        L = self.L * self.L_mask + self.diag_mask
        U = self.U * self.U_mask
        LU = torch.bmm(L, U)  # (b m n)
        reconstruction_loss = self.loss(PA, LU)

        # encourage P to be orthogonal (PP^T = I)
        PPt = torch.bmm(P, P.transpose(-2, -1))  # (b m m)
        ortho_loss = self.loss(PPt, self.identity)

        # encourage entries of P to be close to 0 or 1
        binary_loss = torch.sum(P * (1 - P))

        total_loss = reconstruction_loss + ortho_loss*self.ortho_weight + binary_loss*self.binary_weight

        if self._make_images:
            self.log('image LU', LU, False, to_uint8=True)
            self.log('image P logits', self.P_logits, False, to_uint8=True)
            self.log('image P', P, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)
            self.log_difference('image update U', U, to_uint8=True)
            self.log_difference('image update P logits', self.P_logits, to_uint8=True)

        return total_loss

# no progress with zeros
# hard with ones
# randn vs 0.01 - 0.01 converges much faster and looks better randn might be good because its a bit harder but stick to 001
class Cholesky(Benchmark):
    """Cholesky as objective"""
    def __init__(self, A, loss = F.mse_loss, non_negative_fn=_square, init = _full001, make_images = True):
        super().__init__(log_projections = True, seed=0)
        matrix = _make_float_chw_square_matrix(A)
        if matrix.shape[-1] != matrix.shape[-2]: raise ValueError(f'{matrix.shape = } - not a matrix!')

        self.A = torch.nn.Buffer(matrix.contiguous())
        b, n, _ = matrix.shape

        self.L = nn.Parameter(init((b, n, n), generator=self.rng.torch()).tril(-1).contiguous())
        self.diag_raw = nn.Parameter(init((b, n), generator=self.rng.torch()).contiguous())
        self.non_negative_fn = non_negative_fn
        self.loss_fn = loss

        # real cholesky outputs for reference
        self._make_images = make_images
        if make_images:
            self.add_reference_image('input', matrix)
            try:
                L, info = torch.linalg.cholesky_ex(matrix) # pylint:disable = not-callable
                self.add_reference_image('L - true', L)
            except Exception as e:
                print(f"True choleksy failed (could mean matrix is not positive definite): {e!r}")
            self.set_display_best('image LLT', True)


    def get_loss(self):
        A = self.A
        L = self.L.tril(-1) + torch.diag_embed(self.non_negative_fn(self.diag_raw))
        LLT = torch.bmm(L, L.transpose(-1, -2))

        loss = self.loss_fn(LLT, A)

        if self._make_images:
            self.log('image LLT', LLT, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)

        return loss


# zeros vs randn - adam barely converges with both, L-BFGS converges faster with randn
# ones seems to be the easiest, full001 close
class Eigen(Benchmark):
    """
    Args:
        input_matrix (torch.Tensor): Input matrix with leading channel dimension (C, N, N).
    """
    def __init__(self, A, loss = F.mse_loss, init: Callable | Literal['copy'] = 'copy', make_images=True): # alternatively _ones
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_make_float_chw_tensor(A))
        C, N, _ = self.A.shape

        # initialize eigenvectors (Q) with random orthogonal matrices
        if callable(init): rand_Q = init((C, N, N), generator=self.rng.torch())
        elif init == 'copy': rand_Q = self.A.clone()
        else: raise ValueError(init)

        Q, _ = torch.linalg.qr(rand_Q)
        self.Q = nn.Parameter(Q.contiguous())

        # initialize eigenvalues (diagonal of Lambda)
        self.eigenvalues = nn.Parameter(torch.ones(C, N).contiguous())

        self._make_images = make_images
        self.loss = loss

        if make_images:
            self.add_reference_image('input', self.A)
            try:
                L, V = torch.linalg.eig(self.A) # pylint:disable = not-callable
                self.add_reference_image('eugenvectors - true', V)
            except Exception as e:
                print(f"True eig failed: {e!r}")
            self.set_display_best('image V', True)

    def get_loss(self):
        """
        Compute the reconstruction loss using the current eigenvectors and eigenvalues.
        """
        mul = 1
        Lambda = torch.diag_embed(self.eigenvalues)
        try:
            Q_inv, info = torch.linalg.inv_ex(self.Q)  # (C, N, N)
        except torch.linalg.LinAlgError as e:
            Q_inv = torch.linalg.pinv(self.Q)
            mul = 2

        A_recon = torch.matmul(torch.matmul(self.Q, Lambda), Q_inv)

        loss = self.loss(self.A, A_recon) * mul

        if self._make_images:
            self.log('image reconstructed', A_recon, False, to_uint8=True)
            self.log('image eugenvectors', self.Q, False, to_uint8=True)
            self.log('image Q inv', Q_inv, False, to_uint8=True)
            self.log_difference('image update eugenvectors', self.Q, to_uint8=True)

        return loss.mean()

# slightly easier with randn but produces cool looking W and W_logits with zeros, but mostly black B/B_prime
# L-BFGS really struggles with zeros
# ones is like best of both worlds
class Bruhat(Benchmark):
    def __init__(self, A, entropy_weight=0.1, sinkhorn_iters=10, loss = F.mse_loss, init = _ones, make_images = True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_make_float_chw_tensor(A))
        self.b = self.A.size(0)
        self.n = self.A.size(-1)
        self.entropy_weight = entropy_weight
        self.sinkhorn_iters = sinkhorn_iters
        self.loss = loss
        self._make_images = make_images

        self.B = nn.Parameter(torch.zeros(self.b, self.n, self.n).contiguous())
        self.B_prime = nn.Parameter(torch.zeros(self.b, self.n, self.n).contiguous())
        self.w_logits = nn.Parameter(init((self.b, self.n, self.n), generator=self.rng.torch()).contiguous())

        # initialize off-diagonal elements with small random values
        with torch.no_grad():
            triu_rows, triu_cols = torch.triu_indices(self.n, self.n, offset=1)
            self.B[:, triu_rows, triu_cols] += init((triu_rows.size(0), ), generator=self.rng.torch()) * 0.1
            self.B_prime[:, triu_rows, triu_cols] += init((triu_rows.size(0), ), generator=self.rng.torch()) * 0.1

        if make_images:
            self.add_reference_image('input', self.A)
        self.set_display_best("image reconstruction")

    def get_loss(self):
        # B with exp(diag) and upper off-diagonal
        B_diag = torch.diag_embed(torch.exp(torch.diagonal(self.B, dim1 = -2, dim2 = -1)))
        B_upper_off = torch.triu(self.B, diagonal=1)
        B = B_diag + B_upper_off

        # B' same
        B_prime_diag = torch.diag_embed(torch.exp(torch.diagonal(self.B_prime, dim1 = -2, dim2 = -1)))
        B_prime_upper_off = torch.triu(self.B_prime, diagonal=1)
        B_prime = B_prime_diag + B_prime_upper_off

        w_ds = sinkhorn(self.w_logits, self.sinkhorn_iters)

        # reconstruction loss
        recon = B @ w_ds @ B_prime
        loss = self.loss(self.A, recon)

        # regularization to push w_ds towards permutation matrix
        entropy = -torch.sum(w_ds * torch.log(w_ds + 1e-10))
        entropy_loss = self.entropy_weight * entropy

        if self._make_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image B', B, False, to_uint8=True)
            self.log('image B prime', B_prime, False, to_uint8=True)
            self.log('image W logits', self.w_logits, False, to_uint8=True)
            self.log('image W', w_ds, False, to_uint8=True)
            self.log_difference('image update B', B, to_uint8=True)
            self.log_difference('image update B_prime', B_prime, to_uint8=True)
            self.log_difference('image update W logits', self.w_logits, to_uint8=True)

        total_loss = loss + entropy_loss
        return total_loss


# logits need to be randn otherwise it gets stuck, zeros and ones don't work, but S can be zeros
class Interpolative(Benchmark):
    """
    Args:
        A (torch.Tensor): Input matrix (m x n) to decompose
        k (int): Number of columns to select
        sinkhorn_iters (int | None): Number of Sinkhorn normalization iterations (None to just use softmax and penalty)
        temp (float): Temperature for logit sharpening
        lambda_reg (float): Regularization strength for column diversity
    """
    def __init__(self, A, k, lambda_reg=0.1, sinkhorn_iters: int | None=10, loss = F.mse_loss, init = _zeros,  make_images=True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_make_float_chw_tensor(A))
        self.k = k
        self.lambda_reg = lambda_reg
        self.b, self.m, self.n = self.A.shape
        self.sinkhorn_iters = sinkhorn_iters
        self._make_images = make_images
        self.loss = loss

        # logits for selecting k columns (k distributions over n columns)
        self.P_logits = torch.nn.Parameter(torch.randn((self.b, k, self.n), generator = self.rng.torch()).contiguous())
        # coefficient matrix
        self.S = torch.nn.Parameter(init((self.b, k, self.n), generator = self.rng.torch()).mul(0.01).contiguous())
        self.set_display_best("image reconstruction")
        if make_images:
            self.add_reference_image('input', self.A)

    def get_loss(self):
        """
        Computes the loss for the interpolative decomposition objective.

        Returns:
            torch.Tensor: The total loss (reconstruction + entropy regularization).
        """
        if self.sinkhorn_iters is not None: P = sinkhorn(self.P_logits, self.sinkhorn_iters)
        else: P = torch.softmax(self.P_logits, dim=2)  # Shape: (k, n)
        basis = self.A @ P.transpose(-2, -1)  # Shape: (m, k)
        recon = basis @ self.S  # Shape: (m, n)

        # reconstruction loss
        reconstruction_loss = self.loss(self.A, recon)

        # encourage distinct columns
        if self.sinkhorn_iters is not None:
            column_usage = P.sum(dim=1)  # (n,)
            diversity_loss = -torch.sum(column_usage * torch.log(column_usage + 1e-10))

        else:
            # encourage peaked distributions
            diversity_loss = -torch.sum(P * torch.log(P + 1e-10), dim=2).mean()

        total_loss = reconstruction_loss + self.lambda_reg * diversity_loss

        if self._make_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image P logits', self.P_logits, False, to_uint8=True)
            self.log('image P', P, False, to_uint8=True)
            self.log('image S', self.S, False, to_uint8=True)
            self.log('image basis', basis, False, to_uint8=True)
            self.log_difference('image update P logits', self.P_logits, to_uint8=True)
            self.log_difference('image update S', self.S, to_uint8=True)

        return total_loss

# no progress with zeros, stuck with ones and full, _normal01 converges perfectly
class Polar(Benchmark):
    def __init__(self, A, loss = F.mse_loss, ortho_loss = F.mse_loss, ortho_weight=1.0, init = _normal01, make_images = True):
        super().__init__(log_projections=True)
        self.loss = loss
        self.ortho_loss=ortho_loss
        self.ortho_weight = ortho_weight
        self._make_images = make_images

        self.A = torch.nn.Buffer(_make_float_chw_tensor(A))
        self.b, self.m, self.n = self.A.shape

        if self.m >= self.n:
            # U will be m x n with orthonormal columns
            self.U = nn.Parameter(init((self.b, self.m, self.n), generator = self.rng.torch()))
            self.Q = nn.Parameter(init((self.b, self.n, self.n), generator = self.rng.torch()))
        else:
            # U will be m x n with orthonormal rows
            self.U = nn.Parameter(init((self.b, self.m, self.n), generator = self.rng.torch()))
            self.Q = nn.Parameter(init((self.b, self.m, self.m), generator = self.rng.torch()))

        self.set_display_best("image reconstruction")
        if make_images:
            self.add_reference_image('input', self.A)

    def get_loss(self):
        A = self.A
        Q = self.Q
        P = Q.transpose(-2,-1) @ Q
        U = self.U

        if self.m >= self.n: recon = U @ P
        else: recon =  P @ U

        loss_recon = self.loss(A, recon)

        # orthonormality
        if self.m >= self.n:
            # U^T U should be identity (n x n)
            utu = U.transpose(-2,-1) @ U
            identity = torch.eye(self.n, device=A.device).view(1, self.n, self.n).repeat_interleave(self.b, 0)
            ortho_loss = self.ortho_loss(utu, identity)
        else:
            # U U^T should be identity (m x m)
            uut = U @ U.transpose(-2,-1)
            identity = torch.eye(self.m, device=A.device).view(1, self.m, self.m).repeat_interleave(self.b, 0)
            ortho_loss = self.ortho_loss(uut, identity)

        total_loss = loss_recon + self.ortho_weight * ortho_loss
        if self._make_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log('image Q', Q, False, to_uint8=True)
            self.log('image P', P, False, to_uint8=True)
            self.log_difference('image update U', U, to_uint8=True)
            self.log_difference('image update Q', Q, to_uint8=True)

        return total_loss


class CUR(Benchmark):
    """
    Args:
        input_matrix (torch.Tensor): Input matrix with leading channel dimensions (e.g., (..., m, n)).
        num_cols (int): Number of columns to select (k).
        num_rows (int): Number of rows to select (l).
        sinkhorn_iters (int): Number of Sinkhorn iterations for normalization.
    """
    def __init__(self, A, num_cols, num_rows, sinkhorn_iters=10, loss = F.mse_loss, make_images = True):
        super().__init__(log_projections=True)
        self.A = nn.Buffer(_make_float_chw_tensor(A))  # (..., m, n)
        self.k = num_cols
        self.l = num_rows
        self.sinkhorn_iters = sinkhorn_iters
        self.loss = loss
        self._make_images = make_images

        m, n = self.A.shape[-2], self.A.shape[-1]

        # logits for column (k x n) and row (l x m) selection matrices
        self.S_logits = nn.Parameter(torch.randn(self.k, n))
        self.T_logits = nn.Parameter(torch.randn(self.l, m))


        self.set_display_best("image reconstruction")
        if make_images:
            self.add_reference_image('input', self.A)

    def get_loss(self):
        """
        Computes the CUR decomposition loss.
        Returns:
            torch.Tensor: Frobenius norm loss between input and reconstructed matrix.
        """
        S = sinkhorn(self.S_logits, self.sinkhorn_iters)  # (k, n)
        T = sinkhorn(self.T_logits, self.sinkhorn_iters)  # (l, m)

        # selected columns C = A @ S^T (..., m, k)
        C = torch.matmul(self.A, S.T)

        # selected rows R = T @ A (..., l, n)
        R = torch.matmul(T, self.A)

        # intersection matrix W = T @ A @ S^T (..., l, k)
        W = torch.matmul(T, torch.matmul(self.A, S.T))

        U = torch.linalg.pinv(W)
        recon = torch.matmul(torch.matmul(C, U), R)
        loss = self.loss(self.A, recon)

        if self._make_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image S logits', self.S_logits, False, to_uint8=True)
            self.log('image cols', C, False, to_uint8=True)
            self.log('image T logits', self.T_logits, False, to_uint8=True)
            self.log('image T', self.T, False, to_uint8=True)
            self.log('image rows', R, False, to_uint8=True)
            self.log('image W', W, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log_difference('image update S logits', self.S_logits, to_uint8=True)
            self.log_difference('image update T logits', self.T_logits, to_uint8=True)

        return loss

#zeros is better
class NMF(Benchmark):
    """non negative elements foced by exp for a change"""
    def __init__(self, A, rank, loss = F.mse_loss, init = _zeros, make_images = True):
        super().__init__()
        A = _make_float_chw_tensor(A)
        A = A + A.min()
        self.A = nn.Buffer(A)
        self.input_shape = self.A.shape
        self.rank = rank
        self.loss=  loss
        self._make_images = make_images

        C, M, N = self.input_shape
        # log-normal distribution to ensure positivity after exp
        self.W_raw = nn.Parameter(init((C, M, self.rank), generator = self.rng.torch()))
        self.H_raw = nn.Parameter(init((C, self.rank, N), generator = self.rng.torch()))

        if make_images:
            self.add_reference_image("input", self.A)
            self.set_display_best("WH")

    def get_loss(self):
        W = torch.exp(self.W_raw)
        H = torch.exp(self.H_raw)
        WH = W @ H
        loss = self.loss(self.A, WH)

        if self._make_images:
            self.log("image WH", WH, False, to_uint8=True)
            self.log("image W", W, False, to_uint8=True)
            self.log("image H", H, False, to_uint8=True)

        return loss