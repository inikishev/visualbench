import math
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import to_HW3, normalize


class CirclesDrawer(Benchmark):
    """Reconstructs the passed colored image with soft semi-transparent circles.

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_circles (int):
            number of circles for image reconstruction.
            Each circle has 5 parameters - 2 coordinates (center), 1 radius, 3 color values.
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

    def __init__(
        self,
        target_image,
        num_circles: int = 100,
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

        self.num_circles = num_circles
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Circle parameters (cx, cy, radius, r, g, b, a)
        self.circle_params = nn.Parameter(torch.rand(num_circles, 7, generator=self.rng.torch()))

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
        # Normalize parameters to (0,1)
        p = torch.sigmoid(self.circle_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        # Extract circle components
        cx, cy = p[:, 0], p[:, 1]  # center coordinates
        radius = p[:, 2]  # radius
        colors = p[:, 3:6]  # (N, 3) RGB
        alpha = p[:, 6]  # (N,) transparency

        if self.exp_sharpness:
            sharpness = torch.exp(self.sharpness)
        else:
            sharpness = self.sharpness

        if sharpness < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty
        else:
            penalty = 0

        # Reshape for broadcasting
        cx = cx.view(-1, 1, 1)
        cy = cy.view(-1, 1, 1)
        radius = radius.view(-1, 1, 1)

        # Calculate distance from each pixel to circle center
        # Distance map for each circle: (N, H, W)
        dist_sq = (self.x_grid - cx) ** 2 + (self.y_grid - cy) ** 2

        # Edge detection using sigmoid on (radius - distance) * sharpness
        # This creates a soft circle mask
        mask = torch.sigmoid((radius - torch.sqrt(dist_sq + 1e-8)) * sharpness)

        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        colors = colors.view(-1, 3, 1, 1)
        mask = mask.view(-1, 1, *mask.shape[1:])

        # Circle contributions
        circle_contributions = colors * alpha * mask  # (N, 3, H, W)
        total_circle_contrib = torch.sum(circle_contributions, dim=0)  # (3, H, W)

        # Background contribution (visible where circles don't cover)
        total_alpha = torch.sum(alpha * mask, dim=0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - total_alpha)  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_circle_contrib
        loss = self.loss_fn(reconstructed, self.target_image)

        if self._make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty
