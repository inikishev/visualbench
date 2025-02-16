"""synthetic funcs"""
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .._utils import _normalize_to_uint8, _make_float_tensor
from ..benchmark import Benchmark


class Sphere(Benchmark):
    """Sphere benchmark (or other function depending on criterion).

    Directly minimizes `criterion` between `target` and `init`.

    Supports rendering a video as long as `target` is an image (2d or 3d tensor or path to image)
    Args:
        target (Any): if int, used as number of dims, otherwise is the target itself.
        init (Any, optional): initial values same shape as target, if None initializes to zeros. Defaults to None.
        criterion (Callable, optional): loss function. Defaults to torch.nn.functional.mse_loss.
        log_image (bool, optional): if true logs predicted images for video rendering. Defaults to True.
    """
    def __init__(self, target: Any, init=None, criterion = torch.nn.functional.mse_loss, make_images = True,):
        super().__init__(log_params=False, log_projections=True)
        self.make_images = make_images

        # target tensor
        if isinstance(target, int): target = torch.randn(target, dtype=torch.float32, generator=self.rng.torch())
        self.target = torch.nn.Buffer(_make_float_tensor(target))

        # prediction tensor
        if init is None: init = torch.zeros_like(self.target)
        self.x = torch.nn.Parameter(_make_float_tensor(init).contiguous())
        self.criterion = criterion

        # reference image for plotting
        if self.target.squeeze().ndim in (2, 3):
            self.add_reference_image('target', self.target.squeeze())
            # enable showing best solution so far
            self.set_display_best('image preds', True)

    def get_loss(self):
        # log current recreated image if target is an image
        if self.make_images and len(self.reference_images) != 0:
            self.log('image preds', self.x, False, to_uint8=True)
            self.log_difference('image update', self.x, to_uint8=True)

        # return loss
        return self.criterion(self.x, self.target)

class Convex(Benchmark):
    """Convex function with mostly non-zero hessian, no visualization.

    Args:
        dim (int, optional): number of dimensions. Defaults to 165125384.
        seed (int, optional): rng seed. Defaults to 0.
    """
    def __init__(self, dim=512, mul = 1e-2, seed=0):
        super().__init__(seed=seed, log_projections=True)
        generator = self.rng.torch()
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
    """multidimensional rosenbrock, no visualization"""

    def __init__(self, n_dims=4096, seed=0):
        super().__init__(seed=seed, log_projections=True)
        p = torch.full((n_dims,), -1.5, dtype=torch.float32)
        self.params = torch.nn.Parameter(p + torch.randn(p.shape, dtype=torch.float32, generator=self.rng.torch())*0.05)

    def get_loss(self):
        params = self.params
        x_i = params[:-1]
        x_i_plus_1 = params[1:]

        term1 = (1 - x_i)**2
        term2 = 100 * (x_i_plus_1 - x_i**2)**2
        loss = torch.sum(term1 + term2)

        return loss



class NonlinearMatrixFactorization(Benchmark):
    """matrix factorization with some extra things to make it harder, no visualization"""
    def __init__(self, latent_dim=64, n=1000, seed=0):
        super().__init__(seed=seed, log_projections=True)
        g = self.rng.torch()
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



class AlphaBeta1(Benchmark):
    """functions from https://arxiv.org/abs/2410.12455v1 (not sure how to make this good)"""
    def __init__(self, ndim, num1 = 10, num2 = 10, seed=0):
        super().__init__(seed=seed, log_projections=True)
        g = self.rng.torch()
        self.x = torch.nn.Parameter(torch.randn(ndim, generator = g))
        if num1 > 0: self.a = torch.nn.Buffer(torch.randn((num1), generator=g))
        else: self.a = None
        if num2 > 0: self.b = torch.nn.Buffer(torch.randn((num1, ndim), generator=g))
        else: self.b = None

    def get_loss(self):
        a = self.a; x = self.x; b = self.b; d = self.x.size(0)
        loss = 0
        if a is not None:
            sums_squared = (x.sum() + a * d) ** 2
            ratios = sums_squared / (sums_squared + 1)
            loss = loss + ratios.sum()

        if b is not None:
            squared_diff = (x.reshape(1, d) - b) ** 2
            sum_squared_diff = torch.sum(squared_diff, dim=1)
            loss = loss + torch.sum(sum_squared_diff / (1 + sum_squared_diff))

        return loss


