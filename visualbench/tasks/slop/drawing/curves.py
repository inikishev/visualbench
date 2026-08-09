from typing import Any
from collections.abc import Callable
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ....benchmark import Benchmark
from ....utils import to_HW3, normalize


def _cubic_bezier_curve(t: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                         p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Compute points on a cubic Bezier curve at parameter t.

    Args:
        t: Parameter values in [0, 1], shape (num_samples,)
        p0, p1, p2, p3: Control points, each shape (N, 2) for N curves

    Returns:
        Points on the curve, shape (N, num_samples, 2)
    """
    # Bernstein polynomials
    one_minus_t = 1 - t
    b0 = one_minus_t ** 3
    b1 = 3 * one_minus_t ** 2 * t
    b2 = 3 * one_minus_t * t ** 2
    b3 = t ** 3

    # Stack control points: (N, 4, 2)
    control_points = torch.stack([p0, p1, p2, p3], dim=1)

    # Bernstein weights: (num_samples, 4)
    bernstein = torch.stack([b0, b1, b2, b3], dim=1)

    # Result: (N, num_samples, 2)
    # bernstein: (S, 4), control_points: (N, 4, 2)
    # Output: (N, S, 2)
    result = torch.einsum('sc,ncd->nsd', bernstein, control_points)
    return result


class CurvesDrawer(Benchmark):
    """Reconstructs the passed colored image with soft semi-transparent Bezier curves.

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_curves (int):
            number of Bezier curves for image reconstruction.
            Each cubic Bezier curve has 13 parameters - 8 coordinates (4 control points x, y),
            3 color values, 1 thickness parameter, and 1 alpha (transparency).
            There are also always 3 parameters for background color and 1 for sharpness.
        num_samples (int, optional):
            number of samples per curve for rasterization. Defaults to 50.
        initial_sharpness (float, optional):
            Initial sharpness (it is a learnable parameter and will get optimized). Defaults to 100.
        exp_sharpness (bool, optional):
            Applies exp to sharpness. Defaults to False.
        min_sharpness (float, optional):
            Applies squared penalty for when sharpness is below this. Defaults to 50.
        penalty (float, optional):
            Multiplier to penalty for when sharpness is too low. Defaults to 1.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """

    def __init__(
        self,
        target_image,
        num_curves: int = 50,
        num_samples: int = 50,
        initial_sharpness: float = 100,
        exp_sharpness: bool = False,
        min_sharpness: float = 50,
        penalty: float = 1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', (target_image * 255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_curves = num_curves
        self.num_samples = num_samples
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Curve parameters: 4 control points (x, y each) + RGB + thickness + alpha = 8 + 3 + 1 + 1 = 13
        self.curve_params = nn.Parameter(torch.rand(num_curves, 13, generator=self.rng.torch()).logit())

        # Learnable background color (RGB)
        self.bg_color = nn.Parameter(torch.zeros(3))  # Initialized to black

        # Coordinate grid
        H, W = target_image.shape[-2:]

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing='ij'
        )
        self.x_grid = nn.Buffer(x_grid)
        self.y_grid = nn.Buffer(y_grid)

        # Precompute t values for curve sampling
        t_values = torch.linspace(0, 1, num_samples)
        self.t_values = nn.Buffer(t_values)

        self._show_titles_on_video = False

    def get_loss(self):
        # Normalize parameters to (0,1)
        p = torch.sigmoid(self.curve_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        # Extract curve components
        # Control points
        p0x, p0y = p[:, 0], p[:, 1]  # control point 0
        p1x, p1y = p[:, 2], p[:, 3]  # control point 1
        p2x, p2y = p[:, 4], p[:, 5]  # control point 2
        p3x, p3y = p[:, 6], p[:, 7]  # control point 3

        colors = p[:, 8:11]  # (N, 3) RGB
        thickness = p[:, 11]  # (N,) thickness (0 to 1, will be scaled)
        alpha_param = p[:, 12]  # (N,) alpha/transparency

        # Scale thickness to reasonable range (1 to 10 pixels diagonal-normalized)
        H, W = self.target_image.shape[-2:]
        max_thickness = 0.05 * math.sqrt(H**2 + W**2) / max(H, W)
        thickness = thickness * max_thickness + 0.005  # minimum 0.5% of image size

        if self.exp_sharpness:
            sharpness = torch.exp(self.sharpness)
        else:
            sharpness = self.sharpness

        if sharpness < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty
        else:
            penalty = 0

        # Reshape control points for Bezier computation
        p0 = torch.stack([p0x, p0y], dim=-1)  # (N, 2)
        p1 = torch.stack([p1x, p1y], dim=-1)  # (N, 2)
        p2 = torch.stack([p2x, p2y], dim=-1)  # (N, 2)
        p3 = torch.stack([p3x, p3y], dim=-1)  # (N, 2)

        # Sample curves: (N, num_samples, 2)
        curve_points = _cubic_bezier_curve(self.t_values, p0, p1, p2, p3)

        # Create soft curve masks by computing distance from each pixel to curve
        # Pixel coordinates (H, W, 2)
        pixel_coords = torch.stack([self.x_grid, self.y_grid], dim=-1)  # (H, W, 2)

        # For each curve, compute distance from each pixel to nearest point on curve
        # curve_points: (N, num_samples, 2)
        # pixel_coords: (H, W, 2)
        # We need (N, H, W) distance map

        # Reshape for broadcasting
        curve_points_expanded = curve_points.view(self.num_curves, 1, 1, self.num_samples, 2)  # (N, 1, 1, S, 2)
        pixel_coords_expanded = pixel_coords.view(1, self.target_image.shape[-2], self.target_image.shape[-1], 1, 2)  # (1, H, W, 1, 2)

        # Compute distances: (N, H, W, S)
        distances = torch.sqrt(((curve_points_expanded - pixel_coords_expanded) ** 2).sum(dim=-1) + 1e-8)

        # Minimum distance to curve for each pixel: (N, H, W)
        min_distances = distances.min(dim=-1)[0]

        # Convert to soft mask using Gaussian-like kernel
        # thickness reshaped for broadcasting
        thickness_expanded = thickness.view(-1, 1, 1)  # (N, 1, 1)

        # Soft mask: high when close to curve, low when far
        # Using exp(-distance^2 / (2 * sigma^2)) where sigma relates to thickness
        sigma = thickness_expanded / 2
        masks = torch.exp(-(min_distances ** 2) / (2 * sigma ** 2 + 1e-8))  # (N, H, W)

        # Apply sharpness to create cleaner edges
        masks = torch.sigmoid((masks - 0.5) * sharpness * 2)

        # Reshape for broadcasting
        # Apply per-curve alpha to the mask
        alpha = masks * alpha_param.view(-1, 1, 1)  # (N, H, W)
        colors_expanded = colors.view(-1, 3, 1, 1)  # (N, 3, 1, 1)
        alpha_expanded = alpha.unsqueeze(1)  # (N, 1, H, W)

        # Curve contributions
        curve_contributions = colors_expanded * alpha_expanded  # (N, 3, H, W)
        total_curve_contrib = torch.sum(curve_contributions, dim=0)  # (3, H, W)

        # Background contribution (visible where curves don't cover)
        total_alpha = torch.sum(alpha, dim=0).unsqueeze(0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - torch.clamp(total_alpha, 0, 1))  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_curve_contrib
        reconstructed = torch.clamp(reconstructed, 0, 1)
        loss = self.loss_fn(reconstructed, self.target_image)

        if self.make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty
