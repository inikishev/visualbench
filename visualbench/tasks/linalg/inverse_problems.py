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

class InverseInverse(Benchmark):
    """the goal is to find a matrix whose inverse is A, which is just the inverse of M, but involves backprop through inverse"""
    def __init__(self, A, loss = F.mse_loss, pinv = False, make_images = True):
        super().__init__(log_projections=True)
        self.loss = loss
        self._make_images = make_images
        self.pinv = pinv


        if pinv: self.target = nn.Buffer(_make_float_chw_tensor(A))
        else: self.target = nn.Buffer(_make_float_chw_square_matrix(A))
        self.pred = nn.Parameter(self.target.clone().contiguous())

        if make_images:
            self.add_reference_image('matrix', self.target)
            # invert the matrix to show as reference
            try:
                if pinv: raise torch.linalg.LinAlgError
                true_inv, info = torch.linalg.inv_ex(self.target) # pylint:disable=not-callable
                self.add_reference_image('true inverse', true_inv)
            except torch.linalg.LinAlgError as e:
                pinv = torch.linalg.pinv(self.target) # pylint:disable=not-callable
                self.add_reference_image('pseudoinverse', pinv)

            self.add_reference_image('target', self.target)
            self.set_display_best('image pred inverse')

    def get_loss(self):
        mul = 1

        if self.pinv:
            inv = inv = torch.linalg.pinv(self.pred)

        else:
            try:
                inv, info = torch.linalg.inv_ex(self.pred)  # (C, N, N)
            except torch.linalg.LinAlgError as e:
                inv = torch.linalg.pinv(self.pred)
                mul = 2

        loss = self.loss(inv, self.target) * mul

        if self._make_images:
            self.log('image pred', self.pred, False, to_uint8=True)
            self.log('image pred inverse', inv, False, to_uint8=True)
            self.log_difference('image update pred', self.pred, to_uint8=True)

        return loss



class SinkhornInverse(Benchmark):
    """the goal is to find a matrix, which, after applying sinkhorn iteration, produces A"""
    def __init__(self, A, sinkhorn_iters:int = 4, loss = F.mse_loss, make_images = True):
        super().__init__(log_projections=True)
        self.loss = loss
        self._make_images = make_images
        self.sinkhorn_iters = sinkhorn_iters


        self.target = nn.Buffer(_make_float_chw_tensor(A))
        self.pred = nn.Parameter(self.target.clone().contiguous())

        if make_images:
            self.add_reference_image('target', self.target)
            self.set_display_best('image pred sinkhorn')

    def get_loss(self):
        s = sinkhorn(self.pred, self.sinkhorn_iters)
        loss = self.loss(s, self.target)

        if self._make_images:
            self.log('image pred', self.pred, False, to_uint8=True)
            self.log('image pred sinkhorn', s, False, to_uint8=True)
            self.log_difference('image update pred', self.pred, to_uint8=True)

        return loss

# code from https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    # if G.size(-2) > G.size(-1):
    #     X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    # if G.size(-2) > G.size(-1):
    #     X = X.mT
    return X

class NewtonSchulzInverse(Benchmark):
    """the goal is to find a matrix which after applying Newton-Schulz produces A,
    and Muon's one is perfect because it requires very few iterations"""
    def __init__(self, A, newtonschulz_iters:int = 5, loss = F.mse_loss, make_images = True):
        super().__init__(log_projections=True)
        self.loss = loss
        self._make_images = make_images
        self.newtonschulz_iters = newtonschulz_iters

        self.target = nn.Buffer(_make_float_chw_tensor(A))
        self.pred = nn.Parameter(self.target.clone().contiguous())

        if make_images:
            self.add_reference_image('target', self.target)
            self.set_display_best('image pred Newton-Schulz')

    def get_loss(self):
        s = zeropower_via_newtonschulz5(self.pred, self.newtonschulz_iters)
        loss = self.loss(s, self.target)

        if self._make_images:
            self.log('image pred', self.pred, False, to_uint8=True)
            self.log('image pred Newton-Schulz', s, False, to_uint8=True)
            self.log_difference('image update pred', self.pred, to_uint8=True)

        return loss

