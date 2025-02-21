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
from ._linalg_utils import _expand_channels, _square, _zeros, _full01, _full001, _ones


# with copy initial loss very high
class MatrixLogarithm(Benchmark):
    """finds matrix such that exp(matrix) = M"""
    def __init__(self, M, loss = F.mse_loss, init: Callable | Literal['copy'] = _zeros, make_images=True): # alternatively _zeros
        super().__init__()
        self.M = nn.Buffer(_make_float_chw_square_matrix(M))
        if self.M.shape[-1] != self.M.shape[-2]: raise ValueError(f'{self.M.shape = } - not a matrix!')

        if callable(init): self.log_M = nn.Parameter(init(self.M.shape, generator = self.rng.torch()).contiguous())
        elif init == 'copy': self.log_M = nn.Parameter(self.M.clone().contiguous())
        else: raise ValueError(init)

        self.loss = loss

        self._make_images = make_images
        if make_images:
            self.add_reference_image("target", self.M, to_uint8=True)
            self.set_display_best("image reconstructed")


    def get_loss(self):
        """
        Compute the loss between exp(A) and the target matrix B.

        Args:
            B (torch.Tensor): Target matrix. Expected shape (n, n).

        Returns:
            torch.Tensor: Frobenius norm loss between exp(A) and B.
        """
        exp_log_M = torch.linalg.matrix_exp(self.log_M)
        loss = self.loss(exp_log_M, self.M)

        if self._make_images:
            self.log("image reconstructed", exp_log_M, False, to_uint8=True)
            self.log("image log(M)", self.log_M, False, to_uint8=True)
            self.log_difference("image update log(M)", self.log_M, to_uint8=True)

        return loss



# huge initial losses with copy
class MatrixRoot(Benchmark):
    """finds A such that A^n = M"""
    def __init__(self, M, n: int, loss = F.mse_loss, init: Callable | Literal['copy'] = _full01, make_images=True): # _full01
        super().__init__()
        self.M = nn.Buffer(_make_float_chw_square_matrix(M))
        if self.M.shape[-1] != self.M.shape[-2]: raise ValueError(f'{self.M.shape = } - not a matrix!')

        if callable(init): self.A = nn.Parameter(init(self.M.shape, generator = self.rng.torch()))
        elif init == 'copy': self.A = nn.Parameter(self.M.clone().contiguous())

        self.n = n
        self.loss = loss

        self._make_images = make_images
        if make_images:
            self.add_reference_image("target", self.M, to_uint8=True)
            self.set_display_best("image A^n")


    def get_loss(self):
        """
        Compute the loss between exp(A) and the target matrix B.

        Args:
            B (torch.Tensor): Target matrix. Expected shape (n, n).

        Returns:
            torch.Tensor: Frobenius norm loss between exp(A) and B.
        """
        An = torch.linalg.matrix_power(self.A, n = self.n)
        loss = self.loss(An, self.M)

        if self._make_images:
            self.log("image A^n", An, False, to_uint8=True)
            self.log("image A", self.A, False, to_uint8=True)
            self.log_difference("image update A", self.A, to_uint8=True)

        return loss


class MatrixSign(Benchmark):
    """objective is to converge to fixed point of Newton-Schulz iteration while not diverging too much from M"""
    def __init__(self, M, fixed_loss = F.mse_loss, penalty_loss = F.mse_loss, penalty_weight: float = 1, make_images = True):
        """_summary_

        Args:
            M (_type_): _description_
            fixed_loss (_type_, optional): loss for fixed point convergence. Defaults to F.mse_loss.
            M_loss (_type_, optional): penalty for divergence from M. Defaults to F.mse_loss.
            M_weight (float, optional): weight for penalty. Defaults to 1.
            make_images (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.M = nn.Buffer(_make_float_chw_square_matrix(M))
        if self.M.shape[-1] != self.M.shape[-2]: raise ValueError(f'{self.M.shape = } - not a matrix!')

        self.fixed_loss = fixed_loss
        self.penalty_loss = penalty_loss
        self.penalty_weight = penalty_weight
        b = self.M.size(0)
        n = self.M.size(1)
        self.n = n
        self.I = torch.nn.Buffer(torch.eye(n, dtype=torch.float32).unsqueeze(0).repeat_interleave(b, 0).contiguous())

        # Compute spectral norm of A and scale to ensure convergence
        with torch.no_grad():
            spectral_norm = torch.linalg.norm(self.M, ord=2, dim = (-2, -1), keepdim = True)
            scaling_factor = spectral_norm * 1.1  # Ensure ||S_init||_2 < 1
            S_init = self.M / scaling_factor

        # Initialize learnable parameter S
        self.S = nn.Parameter(S_init.clone().contiguous())

        self._make_images = make_images
        if make_images:
            self.add_reference_image("input", self.M, to_uint8=True)


    def get_loss(self):
        S = self.S
        # Newton-Schulz step
        S_next = 0.5 * S @ (3.0 * self.I - S @ S)
        # loss to drive residual to zero
        loss = self.fixed_loss(S, S_next) + self.penalty_loss(S, self.M) * self.penalty_weight

        if self._make_images:
            self.log("image S next", S_next, False, to_uint8=True)
            self.log("image S", S, False, to_uint8=True)
            self.log_difference("image update S", S, to_uint8=True)

        return loss
