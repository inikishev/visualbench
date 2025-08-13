import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils

letters = "abcdefghijklmnopqrstuvwxyz"

class TensorRankDecomposition(Benchmark):
    def __init__(self, T:Any = (10,20,30,100), rank: int=10, true_rank: int | None = None, criterion=F.mse_loss):
        super().__init__()

        if true_rank is None: true_rank = rank

        T = format.totensor(T)
        if T.ndim == 1:
            sizes = T.int().clip(min=1)
            mats = [torch.randn(true_rank, s, generator=self.rng.torch()) for s in sizes]
            T = linalg_utils.mats_to_tensor(mats)

        self.T = nn.Buffer(T)
        self.mats = nn.ParameterList(torch.randn(rank, s, generator=self.rng.torch()) for s in T.size())

        self.criterion = criterion

    def get_loss(self):
        rec = linalg_utils.mats_to_tensor(self.mats)
        loss = self.criterion(rec, self.T)
        return loss

def _make_tensor(T, generator):
    if isinstance(T, str): T = format.to_CHW(T)
    else: T = format.totensor(T)
    if T.ndim == 1: return torch.randn(size=T.int().tolist(), generator=generator)
    return T


# good one is low rank 10,20,30,100 with rank=10
# I think spectral norm is around -950 with seed=0
class TensorSpectralNorm(Benchmark):
    def __init__(self, T:Any, w_unit:float=0.):
        super().__init__()
        self.T = nn.Buffer(_make_tensor(T, self.rng.torch()))
        self.vecs = nn.ParameterList(torch.randn(s, generator=self.rng.torch()) for s in self.T.size())
        self.w_unit = w_unit

    def get_loss(self):
        loss = 0
        unit_vecs = []
        for v in self.vecs:
            vv = v.dot(v)

            if self.w_unit != 0:
                loss += (vv - 1).abs() * self.w_unit

            unit_vecs.append(v / vv.sqrt().clip(min=1e-8))

        let = letters[:len(unit_vecs)]

        # "a,b,c->abc"
        Tr1 = torch.einsum(f"{','.join(let)}->{let}", *unit_vecs)
        loss -= (self.T*Tr1).sum()

        return loss


class BilinearLeastSquares(Benchmark):
    """Solve y^T A_i x = g_i, where x and y are the decision variables.

    ``A`` must be a 3D tensor and ``g`` is a vector with same number of elements as first dimension of ``A``.

    if ``g`` is None, it is generated randomly.
    """
    def __init__(self, A:Any, g:Any | None = None, criterion=F.mse_loss, algebra=None):
        super().__init__()

        self.A = nn.Buffer(_make_tensor(A, generator=self.rng.torch()))
        if self.A.ndim != 3: raise ValueError(f"A.ndim should be 3, got {A.shape = }")

        if g is None: g = self.A.size(0)
        if isinstance(g, int): g = torch.randn(g, generator=self.rng.torch())
        self.g = nn.Buffer(format.totensor(g).flatten())

        if self.A.size(0) != self.g.numel():
            raise ValueError(f"first dim in T must have same number of elements as size of g, got {A.shape = }, {g.shape = }")

        self.x = nn.Parameter(torch.randn(self.A.size(2), generator=self.rng.torch()))
        self.y = nn.Parameter(torch.randn(self.A.size(1), generator=self.rng.torch()))
        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)

    def get_loss(self):
        yA = algebras.matmul(self.y, self.A, algebra=self.algebra)
        yAx = algebras.matmul(yA, self.x, algebra=self.algebra)
        loss = self.criterion(yAx, self.g)
        return loss
