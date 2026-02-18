import math
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import to_HW3, normalize


class VoronoiDrawer(Benchmark):
    """Reconstructs the passed colored image using Voronoi cells with hard boundaries.

    Unlike PartitionDrawer which uses soft softmax-weighted blending, this creates
    hard Voronoi cells where each pixel belongs to exactly one cell, producing a
    stained-glass / mosaic effect.

    Each Voronoi cell has:
        - A site position (x, y) that defines the cell center
        - A solid color for the cell interior
        - Optional edge highlighting (border color and thickness)

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_cells (int):
            number of Voronoi cells for image reconstruction.
        edge_thickness (float, optional):
            Thickness of cell edges/borders. Defaults to 0.02 (2% of image size).
        edge_sharpness (float, optional):
            Sharpness of cell edge boundaries. Higher = sharper edges. Defaults to 100.
        edge_color (tuple, optional):
            Color of cell edges as RGB tuple. Defaults to (0, 0, 0) black.
        edge_alpha (float, optional):
            Alpha/transparency of cell edges. Defaults to 0.5.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """

    def __init__(
        self,
        target_image: Any,
        num_cells: int = 100,
        edge_thickness: float = 0.02,
        edge_sharpness: float = 100.0,
        edge_color: tuple = (0, 0, 0),
        edge_alpha: float = 0.5,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', (target_image * 255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_cells = num_cells
        self.edge_thickness = edge_thickness
        self.edge_sharpness = edge_sharpness
        self.edge_color = torch.tensor(edge_color, dtype=torch.float32) / 255.0
        self.edge_alpha = edge_alpha
        self.loss_fn = loss_fn

        # Cell parameters (5 per cell):
        # - x, y: site position (2)
        # - RGB color (3)
        self.cell_params = nn.Parameter(torch.rand(num_cells, 5, generator=self.rng.torch()))

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
        p = torch.sigmoid(self.cell_params)

        # Extract cell components
        cx = p[:, 0]  # x position in [0, 1]
        cy = p[:, 1]  # y position in [0, 1]
        colors = p[:, 2:5]  # (N, 3) RGB

        # Compute distance from each pixel to each cell site
        # cx, cy: (N,), x_grid, y_grid: (H, W)
        # Result: (N, H, W)
        cx_expanded = cx.view(-1, 1, 1)  # (N, 1, 1)
        cy_expanded = cy.view(-1, 1, 1)  # (N, 1, 1)
        x_grid_expanded = self.x_grid.unsqueeze(0)  # (1, H, W)
        y_grid_expanded = self.y_grid.unsqueeze(0)  # (1, H, W)

        # Distance from each pixel to each cell site
        distances = torch.sqrt((x_grid_expanded - cx_expanded) ** 2 + (y_grid_expanded - cy_expanded) ** 2 + 1e-8)

        # Hard Voronoi assignment: each pixel belongs to the nearest cell
        # cell_assignment: (H, W) with values in [0, num_cells)
        min_distances, cell_assignment = torch.min(distances, dim=0)  # (H, W), (H, W)

        # Create the cell color image by indexing into colors
        # colors: (N, 3), cell_assignment: (H, W)
        # result: (H, W, 3)
        cell_colors = colors[cell_assignment]  # (H, W, 3)

        # Compute edge mask: pixels near Voronoi boundaries
        # A pixel is on the boundary if its distance to the nearest site
        # is close to the distance to the second-nearest site
        # Sort distances to get first and second minimum
        sorted_distances, _ = torch.sort(distances, dim=0)  # (N, H, W)
        first_dist = sorted_distances[0]  # (H, W)
        second_dist = sorted_distances[1]  # (H, W)

        # Edge detection: where first and second distances are similar
        # Use sigmoid to create soft edge mask
        edge_diff = second_dist - first_dist  # (H, W)
        edge_mask = torch.sigmoid((self.edge_thickness - edge_diff) * self.edge_sharpness)  # (H, W)

        # Blend cell colors with edge color
        edge_color = self.edge_color.to(self.target_image.device)  # (3,)
        edge_color_expanded = edge_color.view(1, 1, 3)  # (1, 1, 3)

        # Final image: blend cell colors with edge based on edge_mask
        # cell_colors: (H, W, 3), edge_mask: (H, W)
        edge_mask_expanded = edge_mask.unsqueeze(-1)  # (H, W, 1)
        reconstructed = cell_colors * (1 - edge_mask_expanded * self.edge_alpha) + edge_color_expanded * (edge_mask_expanded * self.edge_alpha)  # (H, W, 3)

        # Apply background where no cells cover (shouldn't happen with Voronoi, but for completeness)
        reconstructed = reconstructed.permute(2, 0, 1)  # (3, H, W)
        reconstructed = torch.clamp(reconstructed, 0, 1)

        loss = self.loss_fn(reconstructed, self.target_image)

        if self._make_images:
            with torch.no_grad():
                img = reconstructed.detach().permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

                # Also log the Voronoi cell structure (edges only)
                edges_only = edge_mask.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
                self.log_image('voronoi_edges', edges_only.detach().cpu().numpy(), to_uint8=True)

        return loss
