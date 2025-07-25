import math
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import gpytorch
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from linear_operator.utils.errors import NanError, NotPSDError
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark
from .function_descent.test_functions import TEST_FUNCTIONS, TestFunction


def _tomodule(x, *args, **kwargs) -> Any:
    if isinstance(x, nn.Module): return x
    return x(*args, **kwargs)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean: Callable | nn.Module = gpytorch.means.ConstantMean,
        covar: Callable | nn.Module = lambda: gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        distribution: Callable = gpytorch.mlls.ExactMarginalLogLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = _tomodule(mean)
        self.covar_module = _tomodule(covar)

        if hasattr(self.covar_module, "initialize_from_data"):
            self.covar_module.initialize_from_data(train_x, train_y)

        self.distribution = distribution

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return self.distribution(mean_x, covar_x)


class GaussianProcesses(Benchmark):
    """Optimize a GP model to approximate a given function.

    Renders:
        current approximation and the points.

    Args:
        func (Callable[..., torch.Tensor] | str | TestFunction): test function, can be the name of one of the test functions.
        n_points (int): _description_
        domain (tuple[float,float,float,float] | Sequence[float] | None, optional): _description_. Defaults to None.
        mean (Callable | nn.Module, optional): _description_. Defaults to gpytorch.means.ConstantMean.
        covar (_type_, optional): _description_. Defaults to lambda:gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).
        distribution (Callable, optional): _description_. Defaults to gpytorch.distributions.MultivariateNormal.
        likelihood (Callable | nn.Module, optional): _description_. Defaults to gpytorch.likelihoods.GaussianLikelihood.
        mll (Callable, optional): _description_. Defaults to gpytorch.mlls.ExactMarginalLogLikelihood.
        fallback_mll (_type_, optional): _description_. Defaults to gpytorch.mlls.LeaveOneOutPseudoLikelihood.
        maximize (bool, optional): _description_. Defaults to True.
        normalize (bool, optional): _description_. Defaults to True.
        noise (float, optional): _description_. Defaults to 0.01.
        resolution (int, optional): _description_. Defaults to 128.
        grid (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
    """
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        n_points: int,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        mean: Callable | nn.Module = gpytorch.means.ConstantMean,
        covar: Callable | nn.Module = lambda: gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        distribution: Callable = gpytorch.distributions.MultivariateNormal,
        likelihood: Callable | nn.Module = gpytorch.likelihoods.GaussianLikelihood,
        mll: Callable = gpytorch.mlls.ExactMarginalLogLikelihood,
        fallback_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood,
        maximize: bool = True,
        normalize: bool=True,
        noise: float = 0.01,
        resolution: int = 128,
        grid: bool = False
    ):
        super().__init__()
        if isinstance(func, str): f = TEST_FUNCTIONS[func].to(device = 'cpu')
        else: f = func

        if isinstance(f, TestFunction):
            if domain is None: domain = f.domain()

        self.func: Callable[..., torch.Tensor] | TestFunction = f # type:ignore
        self.on_device = False
        if domain is None: raise ValueError("domain not specified")
        self.domain = domain
        self.resolution = resolution
        self.maximize = maximize

        x_min, x_max, y_min, y_max = self.domain

        # sample random points in the domain
        if grid:
            X_coords, y_coords = torch.meshgrid(
                torch.linspace(x_min, x_max, round(math.sqrt(n_points))),
                torch.linspace(y_min, y_max, round(math.sqrt(n_points))),
                indexing='xy',
            )
        else:
            X_coords = torch.rand(n_points, generator=self.rng.torch()) * (x_max - x_min) + x_min
            y_coords = torch.rand(n_points, generator=self.rng.torch()) * (y_max - y_min) + y_min

        X = torch.stack([X_coords.ravel(), y_coords.ravel()], dim=1)

        # evaluate random points
        y = self.func(X[:, 0], X[:, 1])
        y += torch.randn(y.size(), generator=self.rng.torch(), device=y.device, dtype=y.dtype) * y.abs().mean() * noise
        if normalize:
            y = y - y.mean()
            y = y / y.std().clip(min=1e-8)

        self.X = nn.Buffer(X)
        self.y = nn.Buffer(y)
        self.I = nn.Buffer(torch.eye(self.X.shape[0], device=self.X.device, dtype=self.X.dtype))
        self.constant_term = 0.5 * self.X.shape[0] * math.log(2 * math.pi)

        # moedl
        self.likelihood = _tomodule(likelihood)
        self.model = ExactGPModel(X, y, likelihood=self.likelihood, mean=mean, covar=covar, distribution=distribution)
        self.mll = _tomodule(mll, self.likelihood, self.model)
        self.fallback_mll = _tomodule(fallback_mll, self.likelihood, self.model)

        # grid for visualization
        x_vis = torch.linspace(x_min, x_max, resolution)
        y_vis = torch.linspace(y_min, y_max, resolution)
        grid_x, grid_y = torch.meshgrid(x_vis, y_vis, indexing='xy')
        X_vis = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        self.X_vis = nn.Buffer(X_vis)

        self.add_reference_image('true', self.func(grid_x, grid_y), to_uint8=True)


    def get_loss(self) -> torch.Tensor:
        output = self.model(self.X)
        try: loss = self.mll(output, self.y)
        except (NotPSDError,NanError):
            try: loss = self.fallback_mll(output, self.y)
            except (NotPSDError,NanError):
                loss = F.mse_loss(self.likelihood(output).mean, self.y) # what do i do???
                if self.maximize: loss = -loss
        if self.maximize: loss = -loss


        if self._make_images:
            with torch.no_grad():
                self.eval()
                try: pred = self.likelihood(self.model(self.X_vis)).mean
                except (NotPSDError,NanError): pred = torch.zeros_like(self.X_vis[:,0])
                img = pred.reshape(self.resolution, self.resolution)

                minv, maxv = img.min(), img.max()
                img = (img - minv) / (maxv - minv + 1e-8) * 255.0
                img = img.cpu().numpy().astype(np.uint8)

                # Apply a colormap for better visualization
                frame = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS) # pylint:disable=no-member

                # Draw the training points on the frame
                x_min, x_max, y_min, y_max = self.domain
                for i in range(self.X.shape[0]):
                    px = int((self.X[i, 0] - x_min) / (x_max - x_min) * (self.resolution - 1))
                    py = int((self.X[i, 1] - y_min) / (y_max - y_min) * (self.resolution - 1))
                    # OpenCV uses (y, x) for indexing
                    cv2.drawMarker(frame, (px, py), color=(0, 0, 255), markerSize=1) # pylint:disable=no-member

                self.log_image('reconstructed', frame[:,:,::-1], to_uint8=False)
                self.train()

        return loss

