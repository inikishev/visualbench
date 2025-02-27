# pylint: disable = not-callable, redefined-outer-name
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from myai.transforms import normalize, totensor
from torch import nn
from torch.nn import functional as F

from ..._utils import (
    _make_float_tensor,
    _normalize_to_uint8,
    sinkhorn,
)
from ...benchmark import Benchmark
from ._linalg_utils import _expand_channels, _square, _zeros, _full01, _full001, _ones


# better results with randn than zeros ones and full001 but maybe ones is better because it is harder
class CanonicalPolyadic(Benchmark):
    """Canonical polyadic decomposition (this is for 3D tensors which RGB image is btw)"""
    def __init__(self, T, rank, loss = F.mse_loss, init = torch.randn, init_std=0.1, make_images = True):
        super().__init__(log_projections = True, seed=0)
        self.rank = rank
        T = _make_float_tensor(T)
        if T.shape[2] <= 4: T = T.moveaxis(-1, 0)
        self.T = nn.Buffer(T)
        self.dims = self.T.shape
        self.loss = loss

        if len(self.dims) != 3:
            raise ValueError("TensorRankDecomposition currently supports 3D tensors only.")

        self.A = nn.Parameter(init((self.dims[0], rank), generator = self.rng.torch()) * init_std)
        self.B = nn.Parameter(init((self.dims[1], rank), generator = self.rng.torch()) * init_std)
        self.C = nn.Parameter(init((self.dims[2], rank), generator = self.rng.torch()) * init_std)

        self._make_images = make_images
        if make_images:
            self.add_reference_image('input', self.T)
            self.set_display_best("image reconstructed")

    def get_loss(self):
        reconstructed = torch.einsum('ir,jr,kr->ijk', self.A, self.B, self.C)
        loss = self.loss(reconstructed, self.T)

        if self._make_images:
            self.log('image reconstructed', reconstructed, False, to_uint8=True)

        return loss



# no convergence with zeros and stuck on ones or full
class TensorTrain(Benchmark):
    """_summary_

    Args:
        T  target tensor (note if path to rgb image it will be channel first)
        ranks (_type_): list of ints length 1 less than T.ndim

    Raises:
        ValueError: _description_
    """
    def __init__(self, T: torch.Tensor | np.ndarray | Any, ranks: Sequence[int], loss: Callable = F.mse_loss, init = torch.randn, make_images=True):
        super().__init__(log_projections = True, seed=0)
        self.T = nn.Buffer(_make_float_tensor(T))

        self.shape = list(self.T.shape)
        self.ndim = len(self.shape)
        self.ranks = ranks
        self.loss = loss
        self._make_images = make_images

        if len(ranks) != self.ndim - 1:
            raise ValueError(
                f"Length of ranks must be {self.ndim - 1} for a {self.ndim}-dimensional tensor."
            )

        # TT cores
        self.cores = nn.ParameterList()
        for i in range(self.ndim):
            if i == 0:
                in_rank = 1
                out_rank = ranks[i]
            elif i == self.ndim - 1:
                in_rank = ranks[i - 1]
                out_rank = 1
            else:
                in_rank = ranks[i - 1]
                out_rank = ranks[i]
            core = nn.Parameter(init((in_rank, self.shape[i], out_rank), generator = self.rng.torch()))
            self.cores.append(core)

        self.is_image = False
        if make_images:
            if self.T.ndim == 2 or (self.T.ndim == 3 and (self.T.shape[0]<=3 or self.T.shape[2]<=3)):
                self.is_image = True
                self.add_reference_image("target", self.T, to_uint8=True)
            self.set_display_best("image reconstructed")

    def get_loss(self):
        current = self.cores[0].squeeze(0)

        for i in range(1, self.ndim):
            core = self.cores[i]
            r_prev, d_i, r_next = core.shape
            # reshape for matrix multiplication
            core_reshaped = core.reshape(r_prev, -1)
            current = torch.matmul(current, core_reshaped)
            # merge current dimension and prepare for next core
            current = current.reshape(-1, r_next)

        reconstructed = current.reshape(self.T.shape)
        loss = self.loss(reconstructed, self.T)

        if self._make_images and self.is_image:
            self.log("image reconstructed", reconstructed, False, to_uint8=True)

        return loss


# no convergence with zeros and stuck on ones or full
class MPS(Benchmark):
    """matrix product state bend dims 1 less length than T.ndim"""
    def __init__(self, T, bond_dims: Sequence[int], loss = F.mse_loss, make_images = True):
        super().__init__(log_projections = True, seed=0)
        self.T = nn.Buffer(_make_float_tensor(T))
        self.bond_dims = bond_dims
        d = self.T.shape
        N = len(d)
        self.loss_fn = loss

        assert len(bond_dims) == N - 1, (
            f"bond_dims length ({len(bond_dims)}) must be one less than "
            f"target tensor's number of dimensions ({N})"
        )

        self.cores = nn.ParameterList()
        # 1st core: (1, d[0], bond_dims[0])
        self.cores.append(nn.Parameter(torch.randn(1, d[0], bond_dims[0])))
        # middle cores: (bond_dims[i-1], d[i], bond_dims[i])
        for i in range(1, N-1):
            self.cores.append(nn.Parameter(
                torch.randn(bond_dims[i-1], d[i], bond_dims[i])
            ))
        # last core: (bond_dims[-1], d[-1], 1)
        self.cores.append(nn.Parameter(
            torch.randn(bond_dims[-1], d[-1], 1)
        ))

        self.is_image = False
        self._make_images = make_images
        if make_images:
            if self.T.ndim == 2 or (self.T.ndim == 3 and (self.T.shape[0]<=3 or self.T.shape[2]<=3)):
                self.is_image = True
                self.add_reference_image("target", self.T, to_uint8=True)
            self.set_display_best("image reconstructed")

    def get_loss(self):
        current = self.cores[0].squeeze(0)  # Shape: (d0, bond_dim_0)

        for i in range(1, len(self.cores)):
            core = self.cores[i]
            #contracts the bond dimension between current and core
            current = torch.einsum('ab,bcd->acd', current, core)
            # merge the physical dimensions
            current = current.reshape(-1, core.shape[2])

        reconstructed = current.view(self.T.shape)
        loss = self.loss_fn(reconstructed, self.T)

        if self._make_images and self.is_image:
            self.log("image reconstructed", reconstructed, False, to_uint8=True)

        return loss


class CompactHOSVD(Benchmark):
    """Compact Higher-order singular value decomposition

    ranks same length as T.ndim"""
    def __init__(self, T, ranks, loss = F.mse_loss, ortho_weight=1.0, make_images = True):
        super().__init__(log_projections = True, seed=0)
        self.T = nn.Buffer(_make_float_tensor(T))
        assert self.T.ndim == len(ranks), (self.T.ndim, len(ranks))
        self.tensor_shape = self.T.shape
        self.ranks = ranks
        self.ortho_weight = ortho_weight
        self.loss = loss

        self.factors = nn.ParameterList()
        for dim, rank in zip(self.T.shape, ranks):
            U = torch.randn(dim, rank)
            U, _ = torch.linalg.qr(U)  # Orthogonal initialization
            self.factors.append(nn.Parameter(U.contiguous()))

        self.core = nn.Parameter(torch.randn(*ranks).contiguous())
        nn.init.normal_(self.core, mean=0.0, std=0.02)

        self.is_image = False
        self._make_images = make_images
        if make_images:
            if self.T.ndim == 2 or (self.T.ndim == 3 and (self.T.shape[0]<=3 or self.T.shape[2]<=3)):
                self.is_image = True
                self.add_reference_image("target", self.T, to_uint8=True)
            self.set_display_best("image reconstructed")

    def get_loss(self):
        current_core = self.core
        for i, U in enumerate(self.factors):
            current_core = torch.tensordot(U, current_core, dims=([1], [i])) # type:ignore

        current_core = current_core.permute(2, 1, 0)
        reconstruction_loss = self.loss(current_core, self.T)

        # orthogonality regularization
        ortho_loss = 0.0
        for U in self.factors:
            ortho = torch.mm(U.T, U)
            identity = torch.eye(U.size(1), device=U.device)
            ortho_loss += torch.sum((ortho - identity) ** 2)

        # Total loss combining both terms
        total_loss = reconstruction_loss + self.ortho_weight * ortho_loss

        if self._make_images and self.is_image:
            self.log("image reconstructed", current_core, False, to_uint8=True)

        return total_loss