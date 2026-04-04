from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark import Benchmark
from ..utils import totensor
from ..utils.scatter import TrainingVisualizer


def _init(shape, init:Literal['zeros','ones','randn','jitter']):
    if init == 'zeros': return torch.zeros(shape, dtype=torch.float32)
    if init == 'ones': return torch.ones(shape, dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(shape, dtype=torch.float32, generator=generator)
    if init == 'randn': return z
    return z * torch.finfo(z.dtype).eps

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor):
        return self.w*x + self.b

class PolynomialRegression(nn.Module):
    def __init__(self, order: int = 3, reduce_fn=torch.mean, init: Literal['zeros','ones','randn','jitter']='zeros'):
        super().__init__()
        self.w = nn.Parameter(_init(order+1, init))
        self.powers = nn.Buffer(torch.arange(order+1))
        self.reduce_fn = reduce_fn

    def forward(self, x: torch.Tensor):
        x = (x.unsqueeze(1) ** self.powers)
        return self.reduce_fn(x * self.w, dim=1)

class SinusoidalRegression(nn.Module):
    def __init__(self, n: int = 1, reduce_fn=torch.mean):
        super().__init__()
        self.amplitude = nn.Parameter(torch.ones(n))
        self.frequency = nn.Parameter(torch.randn(n)*10)
        self.phase = nn.Parameter(torch.zeros(n))
        self.midline = nn.Parameter(torch.zeros(n))
        self.reduce_fn = reduce_fn

    def forward(self, x: torch.Tensor):
        y = self.amplitude * torch.sin(self.frequency * x.unsqueeze(1) + self.phase) + self.midline
        return self.reduce_fn(y, dim=1)


class SegmentedRegression(nn.Module):
    def __init__(self, n_knots: int = 10):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-5, 5, n_knots))
        self.h = nn.Parameter(torch.zeros(n_knots))

    def forward(self, x: torch.Tensor):
        w, indices = torch.sort(self.w)
        h = self.h[indices]

        # find which interval each x falls into
        # side='left' gives us the index i such that sorted_x[i-1] <= x < sorted_x[i]
        idx = torch.searchsorted(w, x, right=False)
        idx = torch.clamp(idx, 1, len(w) - 1)

        x0, x1 = w[idx-1], w[idx]
        y0, y1 = h[idx-1], h[idx]

        # linear interp
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


class StepRegression(nn.Module):
    def __init__(self, n_steps: int = 10):
        super().__init__()
        self.thresholds = nn.Parameter(torch.linspace(-5, 5, n_steps))
        self.values = nn.Parameter(torch.zeros(n_steps + 1))

    def forward(self, x: torch.Tensor):
        thresholds, _ = torch.sort(self.thresholds)
        idx = torch.searchsorted(thresholds, x)
        return self.values[idx]


class RBFRegression(nn.Module):
    def __init__(self, n_kernels: int = 10):
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(-5, 5, n_kernels))
        self.log_widths = nn.Parameter(torch.zeros(n_kernels))
        self.weights = nn.Parameter(torch.randn(n_kernels))
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor):
        diff = x.unsqueeze(1) - self.centers
        rbf = torch.exp(-(diff**2) / torch.exp(self.log_widths)**2)
        return (rbf * self.weights).sum(dim=1) + self.bias

def synthetic_dataset(n=128, c1=3, c2=4, noise=0.5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=generator)
    y = torch.max(torch.sin(x*c1) + torch.randn(n, generator=generator) * noise, torch.cos(x*c2))
    return x, y

class FitData(Benchmark):
    """Fit a regression model to 1d data.

    Args:
        x: X coordinates of the dataset
        y: Y coordinates of the dataset
        model: model, should accept and return a vector.
        criterion: loss function. Defaults to F.mse_loss.
        n_points: number of points for visualization. Defaults to 300.
        expand: how much to expand visualization field of view relative to ``max(x) - min(x)``. Defaults to 0.5.
        width: width of the visualization frame. Defaults to 800.
        height: height of the visualization frame. Defaults to 600.
        margin: margin of the visualization frame. Defaults to 50.
    """
    LINEAR = LinearRegression
    POLY = PolynomialRegression
    SINE = SinusoidalRegression
    SEGMENTED = SegmentedRegression
    STEP = StepRegression
    RBF = RBFRegression
    DATA = synthetic_dataset

    def __init__(
        self,
        x,
        y,
        model: torch.nn.Module,
        criterion=F.mse_loss,
        n_points=300,
        expand: float = 0.5,
        width=500,
        height=300,
        margin=50,
    ):
        super().__init__()
        self.x = nn.Buffer(totensor(x))
        self.y = nn.Buffer(totensor(y))

        if self.x.ndim != 1: raise RuntimeError(f"x should be a vector, got {x.shape}")
        if self.y.ndim != 1: raise RuntimeError(f"y should be a vector, got {y.shape}")
        if self.y.shape != self.y.shape:
            raise RuntimeError(f"x and y should have the same length, got {x.shape = } and {y.shape = }")

        self.model = model
        self.criterion = criterion


        xmin = self.x.amin()
        xmax = self.x.amax()
        range_ = xmax - xmin
        self.x_vis = nn.Buffer(torch.linspace(xmin-range_*expand, xmax+range_*expand, n_points))
        self.x_vis_np = self.x_vis.numpy(force=True)

        self.vis = TrainingVisualizer(x=x.numpy(force=True), y=y.numpy(force=True),
                                      x_vis=self.x_vis_np, width=width, height=height, margin=margin)

    def get_loss(self):
        y_hat = self.model(self.x)
        loss = self.criterion(y_hat, self.y)

        if self._make_images:
            y_vis = self.model(self.x_vis)
            frame = self.vis.render_frame(self.x_vis_np, y_vis.numpy(force=True))
            self.log_image("model", frame, to_uint8=False, show_best=True)

        return loss
