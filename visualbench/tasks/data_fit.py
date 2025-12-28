from torch import nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from ..benchmark import Benchmark
from ..utils import totensor

class TrainingVisualizer:
    def __init__(self, x, y, width=800, height=600, margin=50):
        self.width = width
        self.height = height
        self.margin = margin

        # Determine data boundaries for scaling
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()

        # Pre-compute scaling factors
        # We subtract margin*2 from the drawable area
        self.scale_x = (width - 2 * margin) / (self.x_max - self.x_min)
        self.scale_y = (height - 2 * margin) / (self.y_max - self.y_min)

        # Pre-render the background (static dataset)
        self.background = np.full((height, width, 3), 255, dtype=np.uint8)

        # Convert static points to pixel coordinates
        px = self._to_pixels_x(x)
        py = self._to_pixels_y(y)

        # Draw dataset points (Blue)
        for i in range(len(px)):
            cv2.circle(self.background, (px[i], py[i]), 2, (255, 100, 0), -1, cv2.LINE_AA)

    def _to_pixels_x(self, x_vals):
        return (self.margin + (x_vals - self.x_min) * self.scale_x).astype(np.int32)

    def _to_pixels_y(self, y_vals):
        # Y is inverted in image coordinates (0 is top)
        return (self.height - self.margin - (y_vals - self.y_min) * self.scale_y).astype(np.int32)

    def render_frame(self, x_pred, y_pred):
        """
        Generates a frame with the model prediction line.
        x_pred/y_pred: numpy arrays
        """
        # Copy the pre-rendered background
        frame = self.background.copy()

        # Convert prediction to pixel coordinates
        px = self._to_pixels_x(x_pred)
        py = self._to_pixels_y(y_pred)

        # Stack into (N, 1, 2) array for cv2.polylines
        pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))

        # Draw the prediction line (Red)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255),
                      thickness=2, lineType=cv2.LINE_AA)

        return frame


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(0.))

    def forward(self, x: torch.Tensor):
        return self.w*x + self.b

class PolynomialRegression(nn.Module):
    def __init__(self, order: int = 3, reduce_fn=torch.mean):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(order+1))
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
        self.w = nn.Parameter(torch.linspace(0, 10, n_knots))
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
        self.thresholds = nn.Parameter(torch.linspace(0, 10, n_steps))
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
    SIN = SinusoidalRegression
    SEGMENTED = SegmentedRegression
    STEP = StepRegression
    RBF = RBFRegression

    def __init__(
        self,
        x,
        y,
        model: torch.nn.Module,
        criterion=F.mse_loss,
        n_points=300,
        expand: float = 0.5,
        width=800,
        height=600,
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

        self.vis = TrainingVisualizer(x=x.numpy(force=True), y=y.numpy(force=True), width=width, height=height, margin=margin)

        xmin = self.x.amin()
        xmax = self.x.amax()
        range_ = xmax - xmin
        self.x_vis = nn.Buffer(torch.linspace(xmin-range_*expand, xmax+range_*expand, n_points))
        self.x_vis_np = self.x_vis.numpy(force=True)

    def get_loss(self):
        y_hat = self.model(self.x)
        loss = self.criterion(y_hat, self.y)

        if self._make_images:
            y_vis = self.model(self.x_vis)
            frame = self.vis.render_frame(self.x_vis_np, y_vis.numpy(force=True))
            self.log_image("model", frame, to_uint8=False, show_best=True)

        return loss
