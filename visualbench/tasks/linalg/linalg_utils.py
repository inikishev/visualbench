from typing import Literal, Any
import warnings
import torch
from torch.nn import functional as F
from ...utils import algebras



def svd(M, driver=None):
    device=M.device
    if driver is not None: M = M.cuda()
    U, S, V = torch.linalg.svd(M, driver=driver) #pylint:disable=not-callable
    if driver is not None: return U.to(device), S.to(device), V.to(device)
    return U, S, V

def orthonormalize_svd(M, driver=None):
    U,S,V = svd(M, driver)
    return (U @ V.mT)

OrthoMode = float | Literal['svd', 'qr']
def orthonormality_constraint(M: torch.Tensor, ortho: OrthoMode, algebra, criterion=F.mse_loss):
    """either orthonormality penalty or projects onto the Stiefel manifold via svd"""
    if not isinstance(M, torch.Tensor): raise TypeError(M)

    if ortho == 'svd':
        try:
            return orthonormalize_svd(M), 0
        except torch.linalg.LinAlgError:
            ortho = 1

    if ortho == 'qr':
        try:
            return torch.linalg.qr(M)[0], 0 #pylint:disable=not-callable
        except torch.linalg.LinAlgError:
            ortho = 1

    *b, m, n = M.shape
    if n > m:
        M = M.mH # works for unitary too
        m, n = n, m

    I = torch.eye(n, dtype=M.dtype, device=M.device).expand(M.shape).clone()
    penalty = criterion(algebras.matmul(M.mH, M, algebra=algebra), I)

    return M, penalty*ortho
