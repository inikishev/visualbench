"""This is stuff that I came up with purely for benchmarking"""
import warnings
from collections.abc import Callable
from typing import Literal

import functools
import torch
from torch import nn

from ...benchmark import Benchmark
from ...utils import algebras, format
from . import linalg_utils
from .decompositions import _brute_find_closest_ab


class SumOfKrons(Benchmark):
    """Decompose square A into a sum of ``k`` kronecker factorizations.

    Args:
        A (Any): something to load and use as a matrix.
        k (int, optional): number of kronecker factorizations.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        algebra (Any, optional): use custom algebra for matrix multiplications. Defaults to None.
        seed (int, optional): seed. Defaults to 0.
    """
    def __init__(
        self,
        A,
        k: int = 2,
        criterion:Callable=torch.nn.functional.mse_loss,
        algebra=None,
        seed=0,
    ):
        super().__init__(seed=seed)
        self.A = torch.nn.Buffer(format.to_square(format.to_CHW(A, generator=self.rng.torch())))

        self.criterion = criterion
        self.algebra = algebras.get_algebra(algebra)
        self.k = k

        *b, s1, s2 = self.A.shape

        m, p = _brute_find_closest_ab(s1)
        n, q = _brute_find_closest_ab(s2)

        self.B = nn.ParameterList(nn.Parameter(linalg_utils.orthogonal((*b, m, n), generator=self.rng.torch())) for _ in range(k))
        self.C = nn.ParameterList(nn.Parameter(linalg_utils.orthogonal((*b, p, q), generator=self.rng.torch())) for _ in range(k))

        self.add_reference_image('A', self.A, to_uint8=True)

    def get_loss(self):
        krons = [torch.stack([algebras.kron(b, c, self.algebra) for b,c in zip(B, C)]) for B, C in zip(self.B, self.C)]
        rec = algebras.sum(torch.stack(krons, 0), self.algebra, dim=0)
        loss = self.criterion(rec, self.A)

        if self._make_images:
            for i in range(self.k):
                self.log_image(f"B_{i}", self.B[i], to_uint8=True, log_difference=True)
                self.log_image(f"C_{i}", self.C[i], to_uint8=True, log_difference=True)
                self.log_image(f"B_{i} ⊗ C_{i}", krons[i], to_uint8=True)
            self.log_image("recreated", rec, to_uint8=True, show_best=True)
        return loss

