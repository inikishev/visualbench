from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from myai.transforms import normalize, totensor
from torch import nn
from torch.nn import functional as F
import torchalgebras as ta

from ..._utils import (
    _make_float_chw_square_matrix,
    _make_float_chw_tensor,
    _make_float_tensor,
    _normalize_to_uint8,
    sinkhorn,
)
from ...benchmark import Benchmark
from ._linalg_utils import _expand_channels, _zeros


class Inverse(Benchmark):
    """Finding inverse of a matrix.

    Args:
        mat (torch.Tensor):
            square matrix (last two dims must be the same), can have additional first channels dimension which is treated as batch dimension.
        loss (Callable, optional): final loss is `loss(A@B, B@A) + loss(A@B, I) + loss(B@A, I) + loss(diag(B@A), 1) + loss(diag(A@B), 1)`. Defaults to l1.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cuda'.
    """
    def __init__(self, A: Any, loss: Callable = torch.nn.functional.mse_loss, algebra: ta.AlgebraType | None = None, dtype: torch.dtype=torch.float32, make_images=True):
        super().__init__(log_projections = True, seed=0)
        matrix: torch.Tensor = _make_float_chw_square_matrix(A)
        if matrix.shape[-1] != matrix.shape[-2]: raise ValueError(f'{matrix.shape = } - not a matrix!')
        matrix = matrix.to(dtype = dtype, memory_format = torch.contiguous_format)
        self.loss_fn = loss

        if make_images:
            self.add_reference_image('input', matrix)
            # invert the matrix to show as reference
            try:
                true_inv, info = torch.linalg.inv_ex(matrix) # pylint:disable=not-callable
                self.add_reference_image('true inverse', true_inv)
            except torch.linalg.LinAlgError:
                pinv = torch.linalg.pinv(matrix) # pylint:disable=not-callable
                self.add_reference_image('pseudoinverse', pinv)

        self.A = torch.nn.Buffer(matrix.contiguous())
        self.B = torch.nn.Parameter(self.A.clone().contiguous().requires_grad_(True))
        self._make_images = make_images
        self.set_display_best('image inverse', True)

        self.algebra = algebra


    def get_loss(self):
        ch = self.A.size(0)
        A = self.A; B = self.B
        if self.algebra is not None:
            A = ta.AlgebraicTensor(A, self.algebra)
            B = ta.AlgebraicTensor(B, self.algebra)
        AB = ta.totorch(A @ B)
        BA = ta.totorch(B @ A)
        I = _expand_channels(torch.eye(A.shape[-1], device = AB.device, dtype=AB.dtype), ch)
        I_diag = _expand_channels(torch.ones(BA.shape[-1], device = AB.device, dtype=AB.dtype), ch)

        loss = self.loss_fn(AB, BA)  +\
            self.loss_fn(AB, I) +\
            self.loss_fn(BA, I) +\
            self.loss_fn(BA.diagonal(0,-2,-1), I_diag) +\
            self.loss_fn(AB.diagonal(0,-2,-1), I_diag)

        if self._make_images:
            self.log('image inverse', self.B, False, to_uint8=True)
            self.log('image reconstructed', torch.linalg.inv_ex(self.B)[0], False, to_uint8=True) # pylint:disable=not-callable
            self.log('image AB', AB, False, to_uint8=True)
            self.log('image BA', BA, False, to_uint8=True)
            self.log_difference('image update B', self.B, to_uint8=True)

        return loss



# zeros is much better
class MoorePenrose(Benchmark):
    def __init__(self, A, loss = F.mse_loss, init: Callable | Literal['copy'] = _zeros, algebra: ta.AlgebraType | None = None,  make_images = True):
        super().__init__(log_projections = True, seed=0)
        self.A = nn.Buffer(_make_float_chw_tensor(A))
        C, M, N = self.A.shape

        if callable(init): self.X = nn.Parameter(init((C, N, M), generator=self.rng.torch()).contiguous())
        elif init == 'copy': self.X = nn.Parameter(self.A.clone().contiguous(), requires_grad=True)
        else: raise ValueError(init)

        self.loss_fn = loss
        self.algebra = algebra
        # real pinv outputs for reference
        self._make_images = make_images
        if make_images:
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

        if self.algebra is not None:
            A = ta.AlgebraicTensor(A, self.algebra)
            X = ta.AlgebraicTensor(X, self.algebra)

        AX = A @ X
        XA = X @ A

        # Term 1: || A X A - A ||_F^2
        AXA = ta.totorch(AX @ A)
        term1 = self.loss_fn(AXA, ta.totorch(A))

        # Term 2: || X A X - X ||_F^2
        XAX = ta.totorch(XA @ X)
        term2 = self.loss_fn(XAX, ta.totorch(X))

        # Term 3: || (A X)^T - A X ||_F^2 (symmetry of A X)
        AX = ta.totorch(AX); XA = ta.totorch(XA)
        term3 = self.loss_fn(AX.mT, AX)

        # Term 4: || (X A)^T - X A ||_F^2 (symmetry of X A)
        term4 = self.loss_fn(XA.mT, XA)

        if self._make_images:
            self.log('image pseudoinverse', self.X, False, to_uint8=True)
            self.log('image reconstructed', torch.linalg.pinv(self.X), False, to_uint8=True) # pylint:disable=not-callable
            self.log('image XA', XA, False, to_uint8=True)
            self.log('image AX', AX, False, to_uint8=True)
            self.log('image AXA', AXA, False, to_uint8=True)
            self.log('image XAX', XAX, False, to_uint8=True)
            self.log_difference('image update pseudoinverse', X, to_uint8=True)


        loss = term1 + term2 + term3 + term4
        return loss
