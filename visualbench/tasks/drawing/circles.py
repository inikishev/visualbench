import math
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import to_HW3, normalize


class CirclesDrawer(Benchmark):
    """Reconstructs the passed colored image using circles.

    Each circle has parameters:
        - position (x, y): center of the dot
        - radius: size of the dot
        - color (RGB): color of the dot
        - alpha: transparency/opacity of the dot

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_stipples (int):
            number of stipples (dots) for image reconstruction.
            There are also always 3 parameters for background color.
        initial_sharpness (float, optional):
            Initial sharpness parameter controlling edge softness of dots. Defaults to 100.
        exp_sharpness (bool, optional):
            Applies exp to sharpness. Defaults to False.
        min_sharpness (float, optional):
            Applies squared penalty for when sharpness is below this. Defaults to 50.
        penalty (float, optional):
            Multiplier to penalty for when sharpness is too low. Defaults to 0.1.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """

    def __init__(
        self,
        target_image: Any,
        num_stipples: int = 200,
        initial_sharpness: float = 100,
        exp_sharpness: bool = False,
        min_sharpness: float = 50,
        penalty: float = 0.1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', (target_image * 255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_stipples = num_stipples
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness (controls edge softness of stipples)
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Stipple parameters (7 per stipple):
        # - x, y: center position (2)
        # - log_radius: log of radius for better optimization (1)
        # - RGB color (3)
        # - alpha: transparency (1)
        self.stipple_params = nn.Parameter(torch.rand(num_stipples, 7, generator=self.rng.torch()))

        # Learnable background color (RGB)
        self.bg_color = nn.Parameter(torch.zeros(3))

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
        # Normalize parameters
        p = torch.sigmoid(self.stipple_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        # Extract stipple components
        # Position (normalized to image coordinates 0 to 1)
        cx = p[:, 0]  # x position in [0, 1]
        cy = p[:, 1]  # y position in [0, 1]

        # Radius (log scale for better optimization, range roughly 0.005 to 0.1 of image size)
        log_radius = torch.lerp(
            torch.tensor(math.log(0.005), device=self.stipple_params.device),
            torch.tensor(math.log(0.1), device=self.stipple_params.device),
            p[:, 2]
        )
        radius = torch.exp(log_radius)  # (N,)

        # Color and alpha
        colors = p[:, 3:6]  # (N, 3) RGB
        alpha = p[:, 6]  # (N,) transparency

        if self.exp_sharpness:
            sharpness = torch.exp(self.sharpness)
        else:
            sharpness = self.sharpness

        if sharpness.item() < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty
        else:
            penalty = 0

        # Compute distance from each pixel to each stipple center
        # cx, cy: (N,), x_grid, y_grid: (H, W)
        # Result: (N, H, W)
        cx_expanded = cx.view(-1, 1, 1)  # (N, 1, 1)
        cy_expanded = cy.view(-1, 1, 1)  # (N, 1, 1)
        x_grid_expanded = self.x_grid.unsqueeze(0)  # (1, H, W)
        y_grid_expanded = self.y_grid.unsqueeze(0)  # (1, H, W)

        # Distance from each pixel to each stipple center
        distances = torch.sqrt((x_grid_expanded - cx_expanded) ** 2 + (y_grid_expanded - cy_expanded) ** 2 + 1e-8)

        # Create soft circular masks for each stipple
        # Using sigmoid for soft edges: high when distance < radius, low when distance > radius
        # sharpness controls the softness of the edge
        radius_expanded = radius.view(-1, 1, 1)  # (N, 1, 1)
        masks = torch.sigmoid((radius_expanded - distances) * sharpness)  # (N, H, W)

        # Apply per-stipple alpha to the mask
        alpha_expanded = alpha.view(-1, 1, 1)  # (N, 1, 1)
        weighted_masks = masks * alpha_expanded  # (N, H, W)

        # Weighted color contributions
        colors_expanded = colors.view(-1, 3, 1, 1)  # (N, 3, 1, 1)
        weighted_masks_expanded = weighted_masks.unsqueeze(1)  # (N, 1, H, W)

        # Stipple contributions: (N, 3, H, W)
        stipple_contributions = colors_expanded * weighted_masks_expanded
        total_stipple_contrib = torch.sum(stipple_contributions, dim=0)  # (3, H, W)

        # Background contribution (visible where stipples don't cover)
        total_alpha = torch.sum(weighted_masks, dim=0).unsqueeze(0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - torch.clamp(total_alpha, 0, 1))  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_stipple_contrib
        reconstructed = torch.clamp(reconstructed, 0, 1)
        loss = self.loss_fn(reconstructed, self.target_image)

        if self._make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty
