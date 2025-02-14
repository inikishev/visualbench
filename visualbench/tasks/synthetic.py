import torch
from torch import nn
from .. import Benchmark
from torch.nn import functional as F

class Convex(Benchmark):
    """Convex function with mostly non-zero hessian, no visualization.

    Args:
        dim (int, optional): number of dimensions. Defaults to 165125384.
        seed (int, optional): rng seed. Defaults to 0.
    """
    def __init__(self, dim=512, mul = 1e-2, seed=0):
        super().__init__(seed=seed)
        generator = torch.Generator().manual_seed(seed)
        self.dim = dim

        #positive definite matrix W^T W + I efficiently
        self.W = torch.nn.Buffer(torch.randn(dim, dim, generator=generator, requires_grad=False))
        self.I = torch.nn.Buffer(torch.eye(dim))
        self.C = torch.nn.Buffer(torch.randn(dim, generator=generator, requires_grad=False))

        self.x = torch.nn.Parameter(torch.randn((1, dim), generator=generator))
        self.mul = mul


    def get_loss(self):
        x = self.x + self.C

        #  x^T (W^T W + I) x
        Wx = torch.mm(x, self.W.T)
        quadratic_term = torch.sum(Wx**2)
        identity_term = torch.sum(x**2)

        return (quadratic_term + identity_term) * self.mul


class Rosenbrock(Benchmark):
    """multidimensional rosenbrock"""

    def __init__(self, n_dims=4096,seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        p = torch.full((n_dims,), -1.5, dtype=torch.float32)
        self.params = torch.nn.Parameter(p + torch.randn(p.shape, dtype=torch.float32, generator=g)*0.05)

    def get_loss(self):
        params = self.params
        x_i = params[:-1]
        x_i_plus_1 = params[1:]

        term1 = (1 - x_i)**2
        term2 = 100 * (x_i_plus_1 - x_i**2)**2
        loss = torch.sum(term1 + term2)

        return loss



class NonlinearMatrixFactorization(Benchmark):
    def __init__(self, latent_dim=64, n=1000, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.latent_dim = latent_dim

        # Core trainable parameters
        self.U = nn.Parameter(torch.randn(n, latent_dim, generator=g))
        self.V = nn.Parameter(torch.randn(latent_dim, latent_dim, generator=g))

        # Fixed random matrices
        self.A = nn.Buffer(torch.randn(latent_dim, latent_dim, generator=g))
        self.B = nn.Buffer(torch.randn(latent_dim, latent_dim, generator=g))
        self.C = nn.Buffer(torch.randn(latent_dim, latent_dim, generator=g))

        # Fixed random projections for target generation
        self.proj1 = torch.nn.Buffer(torch.randn(n, latent_dim, generator=g))
        self.proj2 = torch.nn.Buffer(torch.randn(n, latent_dim, generator=g))

        # Precompute targets with matching dimensions
        self.target1 = torch.nn.Buffer(torch.tanh(self.proj1 @ self.A).detach().requires_grad_(False))
        self.target2 = torch.nn.Buffer(torch.sigmoid(self.proj2 @ self.B @ self.C.T).detach().requires_grad_(False))

    def get_loss(self):
        # Core transformation
        UV = self.U @ self.V

        # First pathway (n x latent_dim)
        pathway1 = torch.tanh(UV @ self.A)
        loss1 = F.mse_loss(pathway1, self.target1)

        # Second pathway (n x latent_dim)
        pathway2 = torch.sigmoid(UV @ self.B)
        intermediate = pathway2 @ self.C
        loss2 = F.mse_loss(intermediate, self.target2)

        # Orthogonality constraint
        ortho_loss = torch.norm(self.U.T @ self.U - torch.eye(self.latent_dim, device=self.U.device))

        return loss1 + loss2 + 0.1 * ortho_loss
