# pylint:disable=not-callable, redefined-outer-name
"""linear algebra objectives"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from myai.transforms import normalize
from torch import nn
from torch.nn import functional as F

from .._utils import _normalize_to_uint8, _to_float_hwc_tensor
from ..benchmark import Benchmark


def _expand_channels(x:torch.Tensor, ch:int):
    if ch == 1: return x.unsqueeze(0)
    return x.unsqueeze(0).repeat_interleave(ch, 0)

class Inverse(Benchmark):
    """Finding inverse of a matrix.

    Args:
        mat (torch.Tensor):
            square matrix (last two dims must be the same), can have additional first channels dimension which is treated as batch dimension.
        loss (Callable, optional): final loss is `loss(A@B, B@A) + loss(A@B, I) + loss(B@A, I) + loss(diag(B@A), 1) + loss(diag(A@B), 1)`. Defaults to l1.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cuda'.
    """
    def __init__(self, A: Any, loss: Callable = torch.nn.functional.mse_loss, dtype: torch.dtype=torch.float32, save_images=True):
        super().__init__(log_projections = True, seed=0)
        matrix: torch.Tensor = _to_float_hwc_tensor(A).moveaxis(-1, 0)
        if matrix.shape[-1] != matrix.shape[-2]: raise ValueError(f'{matrix.shape = } - not a matrix!')
        matrix = matrix.to(dtype = dtype, memory_format = torch.contiguous_format)
        self.loss_fn = loss

        if save_images:
            self.add_reference_image('input', matrix)
            # invert the matrix to show as reference
            try:
                true_inv = torch.linalg.inv(matrix) # pylint:disable=not-callable
                self.add_reference_image('true inverse', true_inv)
            except torch.linalg.LinAlgError as e:
                pinv = torch.linalg.pinv(matrix) # pylint:disable=not-callable
                self.add_reference_image('pseudoinverse', pinv)

        self.A = torch.nn.Buffer(matrix.contiguous())
        self.B = torch.nn.Parameter(self.A.clone().contiguous().requires_grad_(True))
        self.save_images = save_images
        self.set_display_best('image inverse', True)


    def get_loss(self):
        ch = self.A.size(0)
        AB = self.A @ self.B
        BA = self.B @ self.A
        I = _expand_channels(torch.eye(self.A.shape[-1], device = AB.device, dtype=AB.dtype), ch)
        I_diag = _expand_channels(torch.ones(BA.shape[-1], device = AB.device, dtype=AB.dtype), ch)

        loss = self.loss_fn(AB, BA)  +\
            self.loss_fn(AB, I) +\
            self.loss_fn(BA, I) +\
            self.loss_fn(BA.diagonal(0,-2,-1), I_diag) +\
            self.loss_fn(AB.diagonal(0,-2,-1), I_diag)

        if self.save_images:
            self.log('image inverse', self.B, False, to_uint8=True)
            self.log('image AB', AB, False, to_uint8=True)
            self.log('image BA', BA, False, to_uint8=True)
            self.log_difference('image update B', self.B, to_uint8=True)

        return loss



@torch.no_grad()
def _svd_orthogonalize(mat: torch.Tensor) -> torch.Tensor:
    """adapted from https://github.com/MarkTuddenham/Orthogonal-Optimisers"""

    res = []
    for M in mat:
        try:
            u, s, vt = torch.linalg.svd(M, full_matrices=False) # pylint:disable=not-callable
            res.append(u @ vt)
        except RuntimeError:
            eps = 1e-8
            while True:
                try:
                    u, s, v = torch.svd_lowrank(
                        M,
                        q=1,    # assume rank is at least 1
                        M=eps * M.mean() * torch.randn_like(M))
                    res.append(u @ v.T)
                except RuntimeError:
                    eps *= 10

    return torch.stack(res)

class Whitening(Benchmark):
    """whitening a matrix inspired by https://github.com/ethansmith2000/WhitenBySGD/blob/main/whiten.ipynb

    Args:
        mat (Any): matrix any shape and can have one first batch dimension,  last two are orthogonalized
        id_loss (Callable, optional): identity loss of M @ M with identity matrix. Defaults to torch.nn.functional.mse_loss.
        dist_loss (Callable, optional): distance loss to original matrix (by default weight of this is 0). Defaults to torch.nn.functional.mse_loss.
        norm_loss (Callable, optional): loss for norms to match. Defaults to torch.nn.functional.mse_loss.
        weights (tuple, optional): weights for losses like this `(id_loss, dist_loss, norm_loss)`. Defaults to (1.0, 0.0, 0.5).
        side (str, optional): which side to pick smallest or largest. Defaults to "max".
        save_images (bool, optional): if true saves images for video rendering. Defaults to True.
        eps (float, optional): epsilon for dividing by norm. Defaults to 1e-8.
    """
    def __init__(
        self,
        A: Any,
        id_loss=torch.nn.functional.mse_loss,
        dist_loss=torch.nn.functional.mse_loss,
        norm_loss=torch.nn.functional.mse_loss,
        weights=(1.0, 0.1, 0.5),
        side: Literal["min", "max"] = "max",
        save_images = True,
        eps=1e-8,
    ):

        super().__init__(log_projections = True, seed=0)
        matrix = _to_float_hwc_tensor(A).movedim(-1, 0)

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
        if save_images:
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
        self.save_image = save_images

    def get_loss(self):

        if (self.side == 'min' and self.shorter_side == -2) or (self.side == 'max' and self.shorter_side == -1):
            pred = self.whitened @ self.whitened.swapaxes(-1,-2)
        else:
            pred = self.whitened.swapaxes(-1,-2) @ self.whitened

        id_loss = self.id_loss(pred, self.target)
        self.log('id loss', id_loss, False)
        loss = id_loss * self.weights[0]

        if self.weights[1] != 0:
            dist_loss = self.dist_loss(self.whitened, self.original)
            self.log('dist loss', dist_loss, False)
            loss = loss + dist_loss * self.weights[1]

        if self.weights[2] != 0:
            norm_loss = self.norm_loss(pred.norm(dim = self.norm_dim), self.norm_target)
            self.log('norm loss', id_loss, False)
            loss = loss + norm_loss * self.weights[2]

        if self.save_image:
            self.log('image whitened', self.whitened, False, to_uint8=True)
            self.log('image matmul', pred, False, to_uint8=True)
            self.log_difference('image update whitened', self.whitened, to_uint8=True)


        return loss

def _square(x):return x**2


def _fro_loss(x,y):
    return torch.linalg.norm(x - y, ord='fro', dim = (-2,-1)).mean()

def _zeros(size, generator):
    return torch.zeros(size, dtype=torch.float32)
def _ones(size, generator):
    return torch.ones(size, dtype=torch.float32)
def _full001(size, generator):
    return torch.full(size, 0.01, dtype = torch.float32)
def _normal01(size, generator):
    x = torch.empty(size, dtype = torch.float32)
    nn.init.normal_(x, mean=0.0, std=0.1, generator=generator)
    return x

# Matches true SVD with randn not with zeros
# randn and 0.01 produce similar results, ones slightly harder
# keep randn?
class SVD(Benchmark):
    """SVD as objective
    Args:
        A (Any): input matrix
        ortho_weight (_type_, optional): orthogonality loss weight. Defaults to 1..
        non_negative_fn (_type_, optional): function to ensure non-negativity, can also try sigmoid or softplus or something. Defaults to _square.
        save_images (bool, optional): saves images for plotting and video rednering. Defaults to True.
    """
    def __init__(self, A: Any, ortho_weight = 1., loss = _fro_loss, non_negative_fn=_square, init = torch.randn, save_images=True):
        super().__init__(log_projections = True, seed=0)
        self.save_images = save_images
        self.non_negative_fn = non_negative_fn
        self.ortho_weight = ortho_weight
        self.loss = loss

        matrix: torch.Tensor = _to_float_hwc_tensor(A).moveaxis(-1, 0).to(memory_format = torch.contiguous_format)
        self.A = torch.nn.Buffer(matrix.contiguous())

        b, self.m, self.n = matrix.shape
        self.k = min(self.m,self.n)
        self.I_k = torch.nn.Buffer(torch.eye(self.k).contiguous())

        # Initialize parameters with orthogonal matrices
        self.U = torch.nn.Parameter(torch.empty(b, self.m, self.k).contiguous())
        torch.nn.init.orthogonal_(self.U, generator=self.rng.torch())

        self.s_raw = torch.nn.Parameter(init((b, self.k), generator = self.rng.torch()))  # S = s_raw^2 ensures non-negativity

        self.V = torch.nn.Parameter(torch.empty(b, self.n, self.k).contiguous())
        torch.nn.init.orthogonal_(self.V, generator=self.rng.torch())

        # real SVD outputs for reference
        if save_images:
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
        # Ensure S is non-negative (e.g., using squared or abs in parameterization)

        S_mat = torch.diag_embed(S)

        Vh = self.V.swapaxes(-1, -2)

        rec = self.U @ S_mat @ Vh
        recon_loss = self.loss(self.A, rec)

        # orthogonality loss
        U_ortho_loss = torch.linalg.norm(self.U.swapaxes(-1,-2) @ self.U - self.I_k, ord='fro', dim = (-2,-1)).mean()
        V_ortho_loss = torch.linalg.norm(Vh @ self.V - self.I_k, ord='fro', dim = (-2,-1)).mean()

        if self.save_images:
            S_sorted, indices = torch.sort(S**2, descending=True)
            U_sorted = torch.gather(self.U, 2, indices.unsqueeze(1).expand(-1, self.m, -1))
            Vh_sorted = torch.gather(Vh, 1, indices.unsqueeze(-1).expand(-1, -1, self.n))
            self.log('image reconstruction', rec, False, to_uint8=True)
            self.log('image U', U_sorted, False, to_uint8=True)
            self.log('image Vh',Vh_sorted, False, to_uint8=True)
            self.log_difference('image update U', U_sorted, to_uint8=True)
            self.log_difference('image update Vh', Vh_sorted, to_uint8=True)

        return recon_loss + self.ortho_weight * (U_ortho_loss + V_ortho_loss)

# Matches true QR with zeros, harder with randn
class QR(Benchmark):
    """QR as objective

    Args:
        A (Any): input matrix
        ortho_weight (_type_, optional): orthogonality loss weight. Defaults to 1..
        non_negative_fn (_type_, optional): function to ensure non-negativity, can also try sigmoid or softplus or something. Defaults to _square.
        save_images (bool, optional): saves images for plotting and video rednering. Defaults to True.
    """
    def __init__(self, A: Any, ortho_weight = 1., loss = torch.nn.functional.mse_loss, init = _zeros, save_images=True):
        super().__init__(log_projections = True, seed=0)
        self.save_images = save_images
        self.loss = loss
        self.ortho_weight = ortho_weight

        matrix: torch.Tensor = _to_float_hwc_tensor(A).moveaxis(-1, 0).to(memory_format = torch.contiguous_format)
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
        if save_images:
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

        # Orthogonality constraint: ||Q^T Q - I||_F^2
        QtQ = torch.bmm(self.Q.transpose(1, 2), self.Q)

        ortho_loss = (QtQ - self.identity).pow(2).mean()
        # orthogonality loss
        # ortho_loss = torch.linalg.norm(self.Q.swapaxes(-1,-2) @ self.Q - self.I_k, ord='fro', dim = (-2,-1)).mean()

        if self.save_images:
            self.log('image QR', QR, False, to_uint8=True)
            self.log('image Q', self.Q, False, to_uint8=True)
            self.log('image R', R, False, to_uint8=True)
            self.log_difference('image update Q', self.Q, to_uint8=True)
            self.log_difference('image update R', R, to_uint8=True)

        return recon_loss + ortho_loss*self.ortho_weight


# easier and less noisy with zeros
class LU(Benchmark):
    """LU as objective"""
    def __init__(self, A, loss = F.mse_loss, init = _zeros, save_images = True):
        super().__init__(log_projections = True, seed=0)
        matrix = _to_float_hwc_tensor(A).moveaxis(-1, 0)
        # Input matrix is expected to have shape (channels, m, m)
        self.A = torch.nn.Buffer(matrix.contiguous())
        b, m, n = matrix.shape
        k = min(m, n)  # Determine decomposition rank

        # Initialize parameters for L and U with appropriate masks
        self.L = nn.Parameter(init((b, m, k), generator=self.rng.torch()) * torch.tril(torch.ones(m, k), diagonal=-1).unsqueeze(0).contiguous())
        self.U = nn.Parameter(init((b, k, n), generator=self.rng.torch()) * torch.triu(torch.ones(k, n)).unsqueeze(0).contiguous())

        self.I = nn.Buffer(torch.eye(m, k, dtype = matrix.dtype).unsqueeze(0).contiguous())
        self.loss_fn = loss

        # real SVD outputs for reference
        self.save_images = save_images
        if save_images:
            self.add_reference_image('input', matrix)
            try:
                _, L, U = torch.linalg.lu(matrix.cuda(), pivot=False) # pylint:disable = not-callable
                self.add_reference_image('L - true', L)
                self.add_reference_image('U - true', U)
            except Exception as e:
                print(f"true LU decomposition failed: {e!r}")

            self.set_display_best('image LU', True)

    def get_loss(self):
        # Retrieve the input matrix from buffer
        A = self.A
        L = torch.tril(self.L, diagonal=-1) + self.I  # shape (channels, m, m)

        # Construct U as upper triangular
        U = torch.triu(self.U, diagonal=0)

        # Compute the reconstructed matrix
        LU = torch.bmm(L, U)  # shape (channels, m, m)

        # Calculate mean squared error loss
        loss = self.loss_fn(LU, A)

        if self.save_images:
            self.log('image LU', LU, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)
            self.log_difference('image update U', U, to_uint8=True)


        return loss


def sinkhorn(logits, num_iters=10):
    """Applies Sinkhorn normalization to logits to generate a doubly stochastic matrix."""
    log_alpha = logits
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    return torch.exp(log_alpha)



# zeros and normal - same results, zeros looks better
class LUPivot(Benchmark):
    """LU with pivoting with P represented via sinkhorn iteration or by simple penalty if sinkhorn_iters is None.
    One of the few linalg objectives where L-BFGS is not the best."""
    def __init__(self, A, sinkhorn_iters: int | None=10, ortho_weight = 1, binary_weight = 1, sq_loss=False, init = _zeros, save_images = True):

        super().__init__(log_projections = True, seed=0)
        # Register the input matrix as a buffer
        self.A = torch.nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        B, M, N = self.A.shape
        K = min(M, N)

        # Initialize parameters for L, U, and P
        self.L = nn.Parameter(init((B, M, K), generator=self.rng.torch()).contiguous())
        self.U = nn.Parameter(init((B, K, N), generator=self.rng.torch()).contiguous())
        self.P = nn.Parameter(init((B, M, M), generator=self.rng.torch()).contiguous())
        self.L_mask = nn.Buffer(torch.tril(torch.ones(M, K), diagonal=-1).unsqueeze(0).contiguous())
        self.diag_mask = nn.Buffer(torch.eye(M, K).unsqueeze(0).contiguous())  # (1, M, K)
        self.U_mask = nn.Buffer(torch.triu(torch.ones(K, N)).unsqueeze(0).contiguous())  # (1, K, N)
        self.identity = nn.Buffer(torch.eye(M).unsqueeze(0).contiguous())

        self.sinkhorn_iters = sinkhorn_iters
        self.save_images = save_images
        self.ortho_weight = ortho_weight
        self.binary_weight = binary_weight
        self.sq_loss = sq_loss

        if save_images:
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

        # Compute row-stochastic permutation matrix using softmax
        if self.sinkhorn_iters is None: P = torch.softmax(self.P, dim=-1)  # (C, M, M)
        else: P = sinkhorn(self.P, self.sinkhorn_iters)

        # Compute PA = P @ A for each channel
        PA = torch.bmm(P, A)  # (C, M, N)

        L = self.L * self.L_mask + self.diag_mask

        # Construct upper triangular matrix U

        U = self.U * self.U_mask

        # Compute LU = L @ U
        LU = torch.bmm(L, U)  # (C, M, N)

        # Reconstruction loss: ||PA - LU||_F^2
        reconstruction_loss = torch.norm(PA - LU, p='fro')
        if self.sq_loss: reconstruction_loss = reconstruction_loss ** 2

        # Regularization to encourage P to be a permutation matrix
        # Encourage P to be orthogonal (PP^T = I)
          # (1, M, M)
        PPt = torch.bmm(P, P.transpose(1, 2))  # (C, M, M)
        ortho_loss = torch.norm(PPt - self.identity, p='fro')
        if self.sq_loss: reconstruction_loss = reconstruction_loss ** 2

        # Encourage entries of P to be close to 0 or 1
        binary_loss = torch.sum(P * (1 - P))

        # Total loss combines all components
        total_loss = reconstruction_loss + ortho_loss*self.ortho_weight + binary_loss*self.binary_weight

        if self.save_images:
            self.log('image LU', LU, False, to_uint8=True)
            self.log('image P logits', self.P, False, to_uint8=True)
            self.log('image P', P, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log('image U', U, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)
            self.log_difference('image update U', U, to_uint8=True)
            self.log_difference('image update P logits', self.P, to_uint8=True)

        return total_loss

# no progress with zeros
# hard with ones
# randn vs 0.01 - 0.01 converges much faster and looks better randn might be good because its a bit harder but stick to 001
class Cholesky(Benchmark):
    """Cholesky as objective"""
    def __init__(self, A, loss = F.mse_loss, non_negative_fn=_square, init = _full001, save_images = True):
        super().__init__(log_projections = True, seed=0)
        matrix = _to_float_hwc_tensor(A).moveaxis(-1, 0)
        if matrix.shape[-1] != matrix.shape[-2]: raise ValueError(f'{matrix.shape = } - not a matrix!')

        # Input matrix is expected to have shape (channels, m, m)
        self.A = torch.nn.Buffer(matrix.contiguous())
        b, n, _ = matrix.shape

        # Initialize parameters for L and U with appropriate masks
        self.L = nn.Parameter(init((b, n, n), generator=self.rng.torch()).tril(-1).contiguous())
        self.diag_raw = nn.Parameter(init((b, n), generator=self.rng.torch()).contiguous())
        self.non_negative_fn = non_negative_fn
        self.loss_fn = loss

        # real cholesky outputs for reference
        self.save_images = save_images
        if save_images:
            self.add_reference_image('input', matrix)
            try:
                L = torch.linalg.cholesky_ex(matrix) # pylint:disable = not-callable
                self.add_reference_image('L - true', L)
            except Exception as e:
                print(f"True choleksy failed (could mean matrix is not positive definite): {e!r}")
            self.set_display_best('image LLT', True)


    def get_loss(self):

        A = self.A
        L = self.L.tril(-1) + torch.diag_embed(self.non_negative_fn(self.diag_raw))

        LLT = torch.bmm(L, L.transpose(-1, -2))

        # Calculate mean squared error loss
        loss = self.loss_fn(LLT, A)

        if self.save_images:
            self.log('image LLT', LLT, False, to_uint8=True)
            self.log('image L', L, False, to_uint8=True)
            self.log_difference('image update L', L, to_uint8=True)

        return loss

# zeros is much better
class MoorePenrose(Benchmark):
    def __init__(self, A, loss = F.mse_loss, init = _zeros, save_images = True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        C, M, N = self.A.shape

        self.X = nn.Parameter(init((C, N, M), generator=self.rng.torch()).contiguous())
        self.loss_fn = loss
        # real pinv outputs for reference
        self.save_images = save_images
        if save_images:
            self.add_reference_image('input', self.A)
            try:
                pinv = torch.linalg.pinv(self.A) # pylint:disable = not-callable
                self.add_reference_image('pseudoinverse - true', pinv)
            except Exception as e:
                print(f"true pseudoinverse somehow managed to fail: {e!r}")
            self.set_display_best('image pseudoinverse', True)


    def get_loss(self):
        A = self.A
        X = self.X

        # Compute AX and XA for efficiency
        AX = torch.matmul(A, X)
        XA = torch.matmul(X, A)

        # Term 1: || A X A - A ||_F^2
        AXA = torch.matmul(AX, A)
        term1 = self.loss_fn(AXA, A)

        # Term 2: || X A X - X ||_F^2
        XAX = torch.matmul(XA, X)
        term2 = self.loss_fn(XAX, X)

        # Term 3: || (A X)^T - A X ||_F^2 (symmetry of A X)
        term3 = self.loss_fn(AX.transpose(-2, -1), AX)

        # Term 4: || (X A)^T - X A ||_F^2 (symmetry of X A)
        term4 = self.loss_fn(XA.transpose(-2, -1), XA)

        if self.save_images:
            self.log('image pseudoinverse', X, False, to_uint8=True)
            self.log('image XA', XA, False, to_uint8=True)
            self.log('image AX', AX, False, to_uint8=True)
            self.log('image AXA', AXA, False, to_uint8=True)
            self.log('image XAX', XAX, False, to_uint8=True)
            self.log_difference('image update pseudoinverse', X, to_uint8=True)


        loss = term1 + term2 + term3 + term4
        return loss


# zeros vs randn - adam barely converges with both, L-BFGS converges faster with randn
# ones seems to be the easiest, full001 close
class EigenDecomposition(Benchmark):
    """
    Args:
        input_matrix (torch.Tensor): Input matrix with leading channel dimension (C, N, N).
    """
    def __init__(self, A, sq=False, init = _ones, save_images=True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        C, N, _ = self.A.shape

        # Initialize eigenvectors (Q) with random orthogonal matrices
        rand_Q = init((C, N, N), generator=self.rng.torch())
        Q, _ = torch.linalg.qr(rand_Q)  # QR decomposition for orthogonal initialization
        self.Q = nn.Parameter(Q.contiguous())

        # Initialize eigenvalues as a parameter (diagonal of Lambda)
        self.eigenvalues = nn.Parameter(torch.ones(C, N).contiguous())

        self.sq = sq
        self.save_images = save_images

        if save_images:
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
        # Form diagonal eigenvalue matrices (C, N, N)
        Lambda = torch.diag_embed(self.eigenvalues)

        # Compute inverse of eigenvectors matrix Q
        Q_inv = torch.linalg.inv(self.Q)  # (C, N, N)

        # Reconstruct the matrix: A_recon = Q * Lambda * Q^{-1}
        A_recon = torch.matmul(torch.matmul(self.Q, Lambda), Q_inv)

        # Frobenius norm squared loss between input and reconstructed matrix
        loss = torch.linalg.norm(self.A - A_recon, ord='fro', dim = (-1, -2))
        if self.sq: loss = loss**2

        if self.save_images:
            self.log('image reconstructed', A_recon, False, to_uint8=True)
            self.log('image eugenvectors', self.Q, False, to_uint8=True)
            self.log('image Q inv', Q_inv, False, to_uint8=True)
            self.log_difference('image update eugenvectors', self.Q, to_uint8=True)

        return loss.mean()

# slightly easier with randn but produces cool looking W and W_logits with zeros, but mostly black B/B_prime
# L-BFGS really struggles with zeros
# ones is like best of both worlds
class Bruhat(Benchmark):
    def __init__(self, A, entropy_weight=0.1, sinkhorn_iters=10, sq_loss = False, init = _ones, save_images = True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        self.b = self.A.size(0)
        self.n = self.A.size(-1)
        self.entropy_weight = entropy_weight
        self.sinkhorn_iters = sinkhorn_iters
        self.sq_loss = sq_loss
        self.save_images = save_images

        # Initialize parameters for B and B' (upper triangular including diagonal)
        self.B = nn.Parameter(torch.zeros(self.b, self.n, self.n).contiguous())
        self.B_prime = nn.Parameter(torch.zeros(self.b, self.n, self.n).contiguous())
        # Logits for the permutation matrix
        self.w_logits = nn.Parameter(init((self.b, self.n, self.n), generator=self.rng.torch()).contiguous())

        # Initialize off-diagonal elements with small random values
        with torch.no_grad():
            triu_rows, triu_cols = torch.triu_indices(self.n, self.n, offset=1)
            self.B[:, triu_rows, triu_cols] += init((triu_rows.size(0), ), generator=self.rng.torch()) * 0.1
            self.B_prime[:, triu_rows, triu_cols] += init((triu_rows.size(0), ), generator=self.rng.torch()) * 0.1

        if save_images:
            self.add_reference_image('input', self.A)
        self.set_display_best("image reconstruction")

    def get_loss(self):
        # Construct B with exp(diag) and upper off-diagonal
        B_diag = torch.diag_embed(torch.exp(torch.diagonal(self.B, dim1 = -2, dim2 = -1)))
        B_upper_off = torch.triu(self.B, diagonal=1)
        B = B_diag + B_upper_off

        # Construct B' similarly
        B_prime_diag = torch.diag_embed(torch.exp(torch.diagonal(self.B_prime, dim1 = -2, dim2 = -1)))
        B_prime_upper_off = torch.triu(self.B_prime, diagonal=1)
        B_prime = B_prime_diag + B_prime_upper_off

        # Apply Sinkhorn to get doubly stochastic matrix
        w_ds = sinkhorn(self.w_logits, self.sinkhorn_iters)

        # Reconstruction loss
        recon = B @ w_ds @ B_prime
        frob_loss = torch.norm(self.A - recon, p='fro', dim = (-2, -1))
        if self.sq_loss: frob_loss = frob_loss.pow(2).mean()
        else: frob_loss = frob_loss.mean()

        # Entropy regularization to push w_ds towards permutation matrix
        entropy = -torch.sum(w_ds * torch.log(w_ds + 1e-10))
        entropy_loss = self.entropy_weight * entropy

        if self.save_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image B', B, False, to_uint8=True)
            self.log('image B prime', B_prime, False, to_uint8=True)
            self.log('image W logits', self.w_logits, False, to_uint8=True)
            self.log('image W', w_ds, False, to_uint8=True)
            self.log_difference('image update B', B, to_uint8=True)
            self.log_difference('image update B_prime', B_prime, to_uint8=True)
            self.log_difference('image update W logits', self.w_logits, to_uint8=True)

        total_loss = frob_loss + entropy_loss
        return total_loss


# logits need to be randn otherwise it gets stuck, zeros and ones don't work, but S can be zeros
class InterpolativeDecomposition(Benchmark):
    """
    Args:
        A (torch.Tensor): Input matrix (m x n) to decompose
        k (int): Number of columns to select
        sinkhorn_iters (int | None): Number of Sinkhorn normalization iterations (None to just use softmax and penalty)
        temp (float): Temperature for logit sharpening
        lambda_reg (float): Regularization strength for column diversity
    """
    def __init__(self, A, k, lambda_reg=0.1, sinkhorn_iters: int | None=10, init = _zeros,  save_images=True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        self.k = k
        self.lambda_reg = lambda_reg
        self.b, self.m, self.n = self.A.shape
        self.sinkhorn_iters = sinkhorn_iters
        self.save_images = save_images

        # Logits for selecting k columns (k distributions over n columns)
        self.P_logits = torch.nn.Parameter(torch.randn((self.b, k, self.n), generator = self.rng.torch()).contiguous())
        # Coefficient matrix
        self.S = torch.nn.Parameter(init((self.b, k, self.n), generator = self.rng.torch()).mul(0.01).contiguous())
        self.set_display_best("image reconstruction")
        if save_images:
            self.add_reference_image('input', self.A)

    def get_loss(self):
        """
        Computes the loss for the interpolative decomposition objective.

        Returns:
            torch.Tensor: The total loss (reconstruction + entropy regularization).
        """
        if self.sinkhorn_iters is not None: P = sinkhorn(self.P_logits, self.sinkhorn_iters)
        else: P = torch.softmax(self.P_logits, dim=2)  # Shape: (k, n)

        # Compute the basis matrix as weighted sum of A's columns
        basis = self.A @ P.transpose(-2, -1)  # Shape: (m, k)

        # Reconstruct the matrix using the basis and coefficients
        recon = basis @ self.S  # Shape: (m, n)

        # Frobenius norm reconstruction loss
        reconstruction_loss = torch.norm(self.A - recon, p='fro', dim = (-2, -1)).mean()

        # Diversity regularization (encourage distinct column selection)
        if self.sinkhorn_iters is not None:
            column_usage = P.sum(dim=1)  # (n,)
            diversity_loss = -torch.sum(column_usage * torch.log(column_usage + 1e-10))

        else:
            # Entropy regularization to encourage peaked distributions
            diversity_loss = -torch.sum(P * torch.log(P + 1e-10), dim=2).mean()

        # Total loss combines reconstruction and regularization
        total_loss = reconstruction_loss + self.lambda_reg * diversity_loss

        if self.save_images:
            self.log('image reconstruction', recon, False, to_uint8=True)
            self.log('image P logits', self.P_logits, False, to_uint8=True)
            self.log('image P', P, False, to_uint8=True)
            self.log('image S', self.S, False, to_uint8=True)
            self.log('image basis', basis, False, to_uint8=True)
            self.log_difference('image update P logits', self.P_logits, to_uint8=True)
            self.log_difference('image update S', self.S, to_uint8=True)

        return total_loss

# no progress with zeros, large values with ones but good with 0.01
class MatrixSqrt(Benchmark):

    def __init__(self, A, loss = F.mse_loss, init = _full001, save_images=True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_to_float_hwc_tensor(A).moveaxis(-1, 0).contiguous())
        if self.A.shape[-1] != self.A.shape[-2]: raise ValueError(f'{self.A.shape = } - not a matrix!')

        self.B = nn.Parameter(init(self.A.shape, generator = self.rng.torch()).contiguous())

        self.loss = loss
        self.save_images = save_images
        self.set_display_best("image BB")
        if save_images:
            self.add_reference_image('input', self.A)

    def get_loss(self):
        BB = self.B @ self.B
        loss = self.loss(BB, self.A)
        if self.save_images:
            self.log('image BB', BB, False, to_uint8=True)
            self.log('image B', self.B, False, to_uint8=True)
            self.log_difference('image update B', self.B, to_uint8=True)
        return loss

