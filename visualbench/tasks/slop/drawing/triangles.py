from typing import Any
from collections.abc import Callable
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ....benchmark import Benchmark
from ....utils import to_HW3, normalize


def _point_in_triangle_barycentric(p, a, b, c):
    """
    Compute barycentric coordinates for points p with respect to triangle abc.
    p: (..., 2) - points
    a, b, c: (..., 2) - triangle vertices
    Returns: (u, v, w) barycentric coordinates, each (...)
    """
    # Compute vectors
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # Compute dot products
    dot00 = torch.sum(v0 * v0, dim=-1)
    dot01 = torch.sum(v0 * v1, dim=-1)
    dot02 = torch.sum(v0 * v2, dim=-1)
    dot11 = torch.sum(v1 * v1, dim=-1)
    dot12 = torch.sum(v1 * v2, dim=-1)

    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    return u, v, w


class TrianglesDrawer(Benchmark):
    """Reconstructs the passed colored image with soft semi-transparent triangles.

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_triangles (int):
            number of triangles for image reconstruction.
            Each triangle has 10 parameters - 6 coordinates (3 vertices x, y), 3 color values, and alpha.
            There are also always 3 parameters for background color and 1 for sharpness.
        initial_sharpness (float, optional):
            Initial sharpness (it is a learnable parameter and will get optimized). Defaults to 150.
        exp_sharpness (bool, optional):
            Applies exp to sharpness. Defaults to False.
        min_sharpness (float, optional):
            Applies squared penalty for when sharpness is below this. Defaults to 100.
        penalty (float, optional):
            Multiplier to penalty for when sharpness is too low. Defaults to 1.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """
    x_grid: torch.nn.Buffer
    y_grid: torch.nn.Buffer
    target_image: torch.nn.Buffer

    def __init__(
        self,
        target_image,
        num_triangles: int = 100,
        initial_sharpness: float = 150,
        exp_sharpness: bool = False,
        min_sharpness: float = 100,
        penalty: float = 1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', (target_image * 255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_triangles = num_triangles
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Triangle parameters (v1x, v1y, v2x, v2y, v3x, v3y, r, g, b, a).
        # Positions are stored directly in [0, 1] (used without sigmoid) so the
        # initial triangles can be spread over the frame. When init_spread is True
        # the vertices are placed around spread-out seeds with jitter, giving a much
        # better starting point than fully random placement that clusters near center.
        self.triangle_params = nn.Parameter(torch.rand(num_triangles, 10, generator=self.rng.torch()).logit())

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

        self._show_titles_on_video = False

    def get_loss(self):
        p = torch.sigmoid(self.triangle_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        v1x, v1y = p[:, 0], p[:, 1]  # vertex 1
        v2x, v2y = p[:, 2], p[:, 3]  # vertex 2
        v3x, v3y = p[:, 4], p[:, 5]  # vertex 3
        colors = p[:, 6:9]
        alpha = p[:, 9]  # (N,) transparency

        if self.exp_sharpness:
            sharpness = torch.exp(self.sharpness)
        else:
            sharpness = self.sharpness

        if sharpness < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty
        else:
            penalty = 0

        # Reshape for broadcasting
        v1x = v1x.view(-1, 1, 1)
        v1y = v1y.view(-1, 1, 1)
        v2x = v2x.view(-1, 1, 1)
        v2y = v2y.view(-1, 1, 1)
        v3x = v3x.view(-1, 1, 1)
        v3y = v3y.view(-1, 1, 1)

        # Triangle vertices (N, 1, 1, 2)
        v1 = torch.stack([v1x, v1y], dim=-1)
        v2 = torch.stack([v2x, v2y], dim=-1)
        v3 = torch.stack([v3x, v3y], dim=-1)

        # Pixel coordinates (1, H, W, 2)
        pixel_coords = torch.stack([self.x_grid, self.y_grid], dim=-1).unsqueeze(0)

        # Compute barycentric coordinates (N, H, W)
        u, v, w = _point_in_triangle_barycentric(pixel_coords, v1, v2, v3)

        # Soft triangle mask using barycentric coordinates
        # A point is inside if all barycentric coords are >= 0
        # Use sigmoid to create soft edges
        edge1 = torch.sigmoid(u * sharpness)
        edge2 = torch.sigmoid(v * sharpness)
        edge3 = torch.sigmoid(w * sharpness)

        # Combine edges to get smooth triangle mask
        mask = edge1 * edge2 * edge3  # (N, H, W)

        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        colors = colors.view(-1, 3, 1, 1)
        mask = mask.view(-1, 1, *mask.shape[1:])

        # Triangle contributions
        triangle_contributions = colors * alpha * mask  # (N, 3, H, W)
        total_triangle_contrib = torch.sum(triangle_contributions, dim=0)  # (3, H, W)

        # Background contribution (visible where triangles don't cover)
        total_alpha = torch.sum(alpha * mask, dim=0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - total_alpha)  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_triangle_contrib
        loss = self.loss_fn(reconstructed, self.target_image)

        if self.make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty
