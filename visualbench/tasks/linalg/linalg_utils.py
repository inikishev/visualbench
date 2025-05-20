from typing import Literal, Any
import warnings
import torch
from torch.nn import functional as F
from ...utils import algebras

OrthoMode = float | Literal['svd']
def orthonormality_constraint(M: torch.Tensor, ortho: OrthoMode, algebra, criterion=F.mse_loss):
    """either orthonormality penalty or projects onto the Stiefel manifold via svd"""
    if not isinstance(M, torch.Tensor): raise TypeError(M)

    if ortho == 'svd':
        U, S, V = torch.linalg.svd(M) #pylint:disable=not-callable
        return U @ V.T, 0

    *b, m, n = M.shape
    if n > m:
        M = M.mT
        m, n = n, m

    I = torch.eye(n, dtype=M.dtype, device=M.device).expand(M.shape).clone()
    penalty = criterion(algebras.matmul(M.T, M, algebra=algebra), I)

    return M, penalty*ortho