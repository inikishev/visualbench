from typing import Literal, Any
import warnings
import torch
from torch.nn import functional as F
from ...utils import algebras



def svd(M: torch.Tensor, driver=None):
    device=M.device
    if driver is not None: M = M.cuda()
    U, S, V = torch.linalg.svd(M, driver=driver) #pylint:disable=not-callable
    if driver is not None: return U.to(device), S.to(device), V.to(device)
    return U, S, V

def orthonormalize_svd(M: torch.Tensor, driver=None):
    U,S,V = svd(M, driver)
    return (U @ V.mT)

OrthoMode = float | Literal['svd', 'qr'] | None
def orthonormality_constraint(M: torch.Tensor, ortho: OrthoMode, algebra, criterion=F.mse_loss) -> tuple[torch.Tensor, float | torch.Tensor]:
    """either orthonormality penalty or projects onto the Stiefel manifold via svd"""
    if ortho is None: return M, 0
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


def sinkhorn(logits: torch.Tensor, iters: int | None=10) -> torch.Tensor:
    """Applies Sinkhorn normalization to logits to generate a doubly stochastic matrix."""
    if iters is None or iters <= 0: return logits
    log_alpha = logits
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def make_permutation(logits: torch.Tensor, iters: int | None, binary_weight: float, ortho: OrthoMode, algebra) -> tuple[torch.Tensor, float | torch.Tensor]:
    P = sinkhorn(logits, iters)
    penalty = 0
    if binary_weight != 0: penalty = torch.mean(P * (1 - P))
    if ortho is not None:
        P, penalty2 = orthonormality_constraint(P, ortho, algebra=algebra)
        penalty = penalty + penalty2
    return P, penalty
