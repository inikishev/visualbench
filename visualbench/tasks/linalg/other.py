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
from ._linalg_utils import _expand_channels, _svd_orthogonalize


class Whitening(Benchmark):
    """whitening a matrix inspired by https://github.com/ethansmith2000/WhitenBySGD/blob/main/whiten.ipynb

    Args:
        mat (Any): matrix any shape and can have one first batch dimension,  last two are orthogonalized
        id_loss (Callable, optional): identity loss of M @ M with identity matrix. Defaults to torch.nn.functional.mse_loss.
        dist_loss (Callable, optional): distance loss to original matrix (by default weight of this is 0). Defaults to torch.nn.functional.mse_loss.
        norm_loss (Callable, optional): loss for norms to match. Defaults to torch.nn.functional.mse_loss.
        weights (tuple, optional): weights for losses like this `(id_loss, dist_loss, norm_loss)`. Defaults to (1.0, 0.0, 0.5).
        side (str, optional): which side to pick smallest or largest. Defaults to "max".
        make_images (bool, optional): if true saves images for video rendering. Defaults to True.
        eps (float, optional): epsilon for dividing by norm. Defaults to 1e-8.
    """
    def __init__(
        self,
        A: Any,
        id_loss: Callable=torch.nn.functional.mse_loss,
        dist_loss: Callable=torch.nn.functional.mse_loss,
        norm_loss: Callable=torch.nn.functional.mse_loss,
        weights=(1.0, 0.1, 0.5),
        side: Literal["min", "max"] = "max",
        make_images = True,
        eps=1e-8,
    ):

        super().__init__(log_projections = True, seed=0)
        matrix = _make_float_chw_tensor(A)

        if matrix.shape[-2] <= matrix.shape[-1]:
            self.shorter_side = -2
            if side == 'min': self.norm_dim = -1
            else: self.norm_dim = -2
        else:
            self.shorter_side = -1
            if side == 'min': self.norm_dim = -2
            else: self.norm_dim = -1

        matrix = matrix / (matrix.norm(dim=self.norm_dim, keepdim=True) + eps)

        # SVD orthogonalized matrix for reference
        if make_images:
            self.add_reference_image('input', matrix)
            svd = _svd_orthogonalize(matrix)
            self.add_reference_image('SVD', svd)
            self.set_display_best('image whitened', True)

        self.original = torch.nn.Buffer(matrix.contiguous())
        self.whitened = torch.nn.Parameter(matrix.clone().contiguous(), requires_grad=True)

        size = min(matrix.shape[-2], matrix.shape[-1]) if side == 'min' else max(matrix.shape[-2], matrix.shape[-1])
        self.target = torch.nn.Buffer(_expand_channels(torch.eye(size).to(matrix), self.original.size(0)).contiguous())
        self.norm_target = torch.nn.Buffer(_expand_channels(torch.ones(size).to(matrix), self.original.size(0)).contiguous())
        self.weights = weights

        self.id_loss = id_loss
        self.dist_loss = dist_loss
        self.norm_loss = norm_loss
        self.side = side
        self.save_image = make_images

    def get_loss(self):

        if (self.side == 'min' and self.shorter_side == -2) or (self.side == 'max' and self.shorter_side == -1):
            pred = self.whitened @ self.whitened.swapaxes(-1,-2)
        else:
            pred = self.whitened.swapaxes(-1,-2) @ self.whitened

        id_loss = self.id_loss(pred, self.target)
        # self.log('id loss', id_loss, False)
        loss = id_loss * self.weights[0]

        if self.weights[1] != 0:
            dist_loss = self.dist_loss(self.whitened, self.original)
            # self.log('dist loss', dist_loss, False)
            loss = loss + dist_loss * self.weights[1]

        if self.weights[2] != 0:
            norm_loss = self.norm_loss(pred.norm(dim = self.norm_dim), self.norm_target)
            # self.log('norm loss', id_loss, False)
            loss = loss + norm_loss * self.weights[2]

        if self.save_image:
            self.log('image whitened', self.whitened, False, to_uint8=True)
            self.log('image matmul', pred, False, to_uint8=True)
            self.log_difference('image update whitened', self.whitened, to_uint8=True)


        return loss



class PCA(Benchmark):
    """
    Args:
        X (_type_): _description_
        output_dim (int): Number of principal components to compute.
        batched (bool, optional): _description_. Defaults to False.
        loss (_type_, optional): _description_. Defaults to F.mse_loss.
        init (_type_, optional): _description_. Defaults to torch.randn.
        make_images (bool, optional): _description_. Defaults to True.

    """
    def __init__(self, X, output_dim, batched = False, loss = F.mse_loss, init = torch.randn, make_images = True):
        """_summary_

        Args:
        """
        super().__init__(log_projections = True, seed=0)
        X = _make_float_chw_tensor(X)
        self.X = nn.Buffer(X - X.mean())

        self.loss = loss
        self.batched = batched
        if batched:
            self.W = nn.Parameter(init((self.X.shape[0], self.X.shape[2], output_dim), generator = self.rng.torch()))
        else:
            X_2d = self.X.view(-1, self.X.shape[-1])
            self.W = nn.Parameter(init((X_2d.shape[1], output_dim), generator = self.rng.torch()))

        self._make_images = make_images
        if make_images:
            self.add_reference_image('input', self.X)
            self.set_display_best("image reconstructed")

    def get_loss(self):
        # orthonormal projection matrix Q
        Q, _ = torch.linalg.qr(self.W)

        # project data onto the orthonormal basis (Q)
        if self.batched: X = self.X
        else: X = self.X.view(-1, self.X.shape[-1])
        X_proj = X @ Q  # (batch_size, output_dim)

        # reconstruct the data from the low-dimensional projection
        X_recon = X_proj @ Q.transpose(-2, -1)  # (batch_size, input_dim)
        loss = self.loss(X, X_recon)

        if self._make_images:
            self.log('image reconstructed', X_recon.view_as(self.X), False, to_uint8=True)
            self.log('image projected', X_proj, False, to_uint8=True)
            self.log('image W', self.W, False, to_uint8=True)
        return loss




class JordanForm(Benchmark):
    """objective is to find jordan form (note that if matrix)"""
    def __init__(self, M, loss = F.mse_loss, lambda_det=0.1, eps=1e-6,make_images = True):
        super().__init__()
        self.M = nn.Buffer(_make_float_chw_tensor(M))

        self.n = self.M.size(1)

        self.P = nn.Parameter(torch.eye(self.n))
        self.lambda_det = lambda_det
        self.loss = loss
        self.eps=eps

        self._make_images = make_images
        if make_images:
            self.add_reference_image("input", self.M, to_uint8=True)

        self.eye = nn.Buffer(torch.eye(self.n, dtype=torch.float32)*self.eps)



    def get_loss(self):
        mul = 1
        P = self.P + self.eye
        try: P_inv, _ = torch.linalg.inv_ex(P)
        except torch.linalg.LinAlgError:
            mul = 2
            P_inv = torch.linalg.pinv(P)

        J = P_inv @ self.M @ self.P

        # encourage diagonal
        loss = self.loss(J, torch.diag_embed(torch.diagonal(J, dim1=-2, dim2=-1))) * mul

        # encourage upper triangular
        lower_tri = torch.tril(J, diagonal=-1)
        lower_loss = lower_tri.pow(2).mean()

        # regularization to keep P invertible (maximize log|det(P)|)
        det = torch.det(self.P)
        det_loss = (torch.log(torch.abs(det) + self.eps)) ** 2  # Add small epsilon for numerical stability

        total_loss = loss + lower_loss + self.lambda_det * det_loss

        if self._make_images:
            self.log("image Jordan Norm", J, False, to_uint8=True)
            self.log('image P', self.P, False, to_uint8=True)
            self.log_difference('image P update', self.P, to_uint8=True)
            self.log('image P inv', P_inv, False, to_uint8=True)

        return total_loss



def _check_square(M):
    if M.ndim != 2: raise ValueError(M.shape)
    if M.size(0) != M.size(1): raise ValueError(M.shape)

class QEP(Benchmark):
    """Quadratic eigenvalue problem M C and K must be square matrices of same shape and there is no visualization.
    if you are looking for structured matrices to use, imags with a lot of white background seem much better for this."""
    def __init__(self, M, C, K):
        super().__init__(log_projections=True)
        self.M = nn.Buffer(_make_float_tensor(M).squeeze())
        self.C = nn.Buffer(_make_float_tensor(C).squeeze())
        self.K = nn.Buffer(_make_float_tensor(K).squeeze())
        _check_square(self.M)
        _check_square(self.C)
        _check_square(self.K)

        n = self.M.size(0)
        self.lambda_param = nn.Parameter(torch.randn(1))
        self.x = nn.Parameter(torch.randn(n))

    def get_loss(self):
        # norm penalty
        x_norm = torch.linalg.vector_norm(self.x) + 1e-8 # pylint:disable=not-callable
        x_normalized = self.x / x_norm
        norm_penalty = (1 - x_norm).pow(2)

        lambda_sq = self.lambda_param ** 2
        term1 = lambda_sq * (self.M @ x_normalized)
        term2 = self.lambda_param * (self.C @ x_normalized)
        term3 = self.K @ x_normalized

        residual = term1 + term2 + term3 + norm_penalty
        loss = torch.mean(residual ** 2)
        return loss



class LatticeBasisReduction(Benchmark):
    def __init__(self, A, penalty_loss = F.mse_loss, penalty_weight: float = 1, make_images=True):
        super().__init__(log_projections=True)
        self._make_images = make_images
        self.penalty_loss = penalty_loss
        self.penalty_weight = penalty_weight

        self.A = nn.Buffer(_make_float_chw_square_matrix(A))
        self.B = nn.Parameter(self.A.clone())

        if make_images:
            self.add_reference_image("input", self.A, to_uint8=True)
            self.set_display_best('image B')

    def get_loss(self):
        B = self.B
        b, d, n = B.shape
        Q, R = torch.linalg.qr(B)  # (d, n), (n, n)

        # size reduction coefficients (mu) from R matrix
        diag_R = torch.diagonal(R, dim1 = -2, dim2 = -1)
        safe_diag = diag_R + 1e-8 * torch.sign(diag_R) + 1e-9
        mu_matrix = R / safe_diag.unsqueeze(1)

        # extract lower triangular mu values (transpose upper triangle)
        mu = torch.triu(mu_matrix, diagonal=1).transpose(-2,-1)  # lower triangular without diagonal

        # length loss
        length_loss = torch.mean(B ** 2)

        # penalize off-diagonal elements of gram matrix (orthogonality loss)
        gram = B.transpose(-2,-1) @ B
        ortho_loss = torch.sum(gram ** 2) / gram.numel() - torch.sum(torch.diagonal(gram, dim1 = -2, dim2 = -1) ** 2) / gram.numel()

        # penalize |mu_ij| > 0.5 using only lower triangle
        mask = torch.tril(torch.ones((b, n, n), device=B.device), diagonal=-1).bool()
        size_red_mu = mu[mask]
        size_red_loss = torch.mean(torch.relu(torch.abs(size_red_mu) - 0.5) ** 2)

        # penalize divergence from A
        penalty = self.penalty_loss(B, self.A) * self.penalty_weight

        total_loss = length_loss + ortho_loss + size_red_loss + penalty

        if self._make_images:
            self.log("image B", self.B, False, to_uint8=True)
            self.log_difference("image update B", self.B, to_uint8=True)

        return total_loss



# randn conveges faster than ones and zeros makes nans
class UnbalancedProcrustes(Benchmark):
    """no visualization"""
    def __init__(self, A, B, loss = F.mse_loss):
        super().__init__(log_projections=True)
        self.loss = loss

        self.A = nn.Buffer(_make_float_chw_tensor(A))
        self.B = nn.Buffer(_make_float_chw_tensor(B))

        b, m, l = self.A.shape
        b, m_b, n = self.B.shape
        assert m == m_b, "A and B must have the same number of rows."
        assert m >= l >= n, "Dimensions must satisfy m >= l >= n."

        self.U = nn.Parameter(torch.randn(b, l, n))

    def get_loss(self):
        U_orth, _ = torch.linalg.qr(self.U, mode='reduced')
        AU = self.A @ U_orth
        loss = self.loss(AU, self.B)

        return loss


class WahbaProblem(Benchmark):
    """
    no visualization.
    Args:
        u (torch.Tensor): Input vectors in the source frame, shape (n, 3)
        v (torch.Tensor): Target vectors in the destination frame, shape (n, 3)
        w (torch.Tensor): Non-negative weights, shape (n,)
    """
    def __init__(self, u, v, w=None):
        super().__init__(log_projections=True)
        # initialize quaternion parameters (q0, q1, q2, q3) with identity rotation
        self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))

        self.u = nn.Buffer(_make_float_tensor(u))
        self.v = nn.Buffer(_make_float_tensor(v))
        if w is None: w = torch.ones(self.u.shape[0])
        self.w = nn.Buffer(_make_float_tensor(w))

    def quaternion_to_rotation_matrix(self, q):
        """Converts a normalized quaternion to a rotation matrix."""
        q = q / torch.norm(q)  # Ensure quaternion is normalized
        a, b, c, d = q[0], q[1], q[2], q[3]

        # rotation matrix elements
        R = torch.stack([
            torch.stack([a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d), 2*(b*d + a*c)]),
            torch.stack([2*(b*c + a*d), a**2 - b**2 + c**2 - d**2, 2*(c*d - a*b)]),
            torch.stack([2*(b*d - a*c), 2*(c*d + a*b), a**2 - b**2 - c**2 + d**2])
        ])
        return R

    def get_loss(self):
        u,v,w = self.u,self.v,self.w
        # normalize quaternion and compute rotation matrix
        R = self.quaternion_to_rotation_matrix(self.q)

        # rotate input vectors: u_rotated = u @ R.T (row vectors)
        u_rotated = torch.mm(u, R.T)

        # loss
        residuals = v - u_rotated
        squared_norms = torch.mean(residuals**2, dim=1)
        loss = torch.mean(w * squared_norms)

        return loss