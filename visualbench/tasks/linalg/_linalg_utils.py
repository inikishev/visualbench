import torch

# not-negative definite function
def _square(x):return x**2

# inits
def _zeros(size, generator):
    return torch.zeros(size, dtype=torch.float32)
def _ones(size, generator):
    return torch.ones(size, dtype=torch.float32)
def _full01(size, generator):
    return torch.full(size, 0.1, dtype = torch.float32)
def _full001(size, generator):
    return torch.full(size, 0.01, dtype = torch.float32)
def _normal01(size, generator):
    x = torch.empty(size, dtype = torch.float32)
    torch.nn.init.normal_(x, mean=0.0, std=0.1, generator=generator)
    return x

# losses
def _fro_loss(x,y):
    return torch.linalg.norm(x-y, ord='fro', dim=(-2,-1)).mean() / (x.shape[-1]*x.shape[-2]) # pylint:disable = not-callable

def _expand_channels(x:torch.Tensor, ch:int):
    if ch == 1: return x.unsqueeze(0)
    return x.unsqueeze(0).repeat_interleave(ch, 0)



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
