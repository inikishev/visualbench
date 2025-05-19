# pylint:disable=not-callable
import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ...benchmark import Benchmark
from ...utils import tonumpy, totensor

RADII1 = tuple([1,2,3,4,1,2,3,1,2,1]*4 + [10])
RADII2 = list(range(100))

class SpherePacking(Benchmark):
    """The goal is to pack 2D spheres as densely as possible.

    The objective is for each sphere to be as close to the origin as possible while penalizing overlaps.

    Args:
        radii (Sequence[float] | np.ndarray | torch.Tensor): list of radii of each sphere

    """
    def __init__(self, radii: Sequence[float] | np.ndarray | torch.Tensor = RADII1):
        super().__init__()
        self.N = len(radii)
        self.radii = nn.Buffer(totensor(radii, dtype=torch.float32))

        # we arrange spheres in a circle which is quite involved but its good initialization...
        # Calculate sum of consecutive radii (including last to first)
        sum_r = self.radii + torch.roll(self.radii, shifts=-1, dims=0)
        sum_r_max = torch.max(sum_r)

        # Binary search to find optimal circle radius R
        R_low = sum_r_max / 2.0
        R_high = torch.sum(self.radii)  # Initial upper bound

        # Perform binary search
        for _ in range(100):
            R = (R_low + R_high) / 2.0
            theta_total = 2.0 * torch.sum(torch.asin(sum_r / (2.0 * R)))
            if theta_total < 2 * math.pi:
                R_high = R  # Need larger theta_total, decrease R
            else:
                R_low = R

        R = (R_low + R_high) / 2.0

        # Calculate angles between consecutive spheres
        theta_i = 2 * torch.asin(sum_r / (2.0 * R))

        # Compute cumulative angles for sphere positions
        theta_without_last = theta_i[:-1]
        cum_sum = torch.cumsum(theta_without_last, dim=0)
        alpha = torch.zeros(self.N, device=self.radii.device)
        alpha[1:] = cum_sum

        # Initialize positions
        x = R * torch.cos(alpha)
        y = R * torch.sin(alpha)
        self.positions = torch.nn.Parameter(torch.stack([x, y], dim=1))

        self.spread_coeff = 0.01
        self.edge_epsilon = 0.015

        # Visualization setup
        H, W = 256, 256
        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.grid = nn.Buffer(torch.stack([grid_x, grid_y], -1))

    def get_loss(self):
        pos, radii = self.positions, self.radii

        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        distances = torch.sqrt((diff**2).sum(-1) + 1e-8)
        i, j = torch.triu_indices(self.N, self.N, 1)
        overlaps = torch.clamp_min(radii[i] + radii[j] - distances[i, j], 0)
        total_loss = (overlaps**2).sum() + self.spread_coeff * (pos**2).sum()

        # Visualization
        if self._make_images:
            with torch.no_grad():
                # Scale to fit spheres in view
                combined = torch.cat([pos - radii.unsqueeze(-1),
                                    pos + radii.unsqueeze(-1)], -1)
                min_val, _ = combined.min(0)[0].min(0)
                max_val, _ = combined.max(0)[0].max(0)
                scale = 2 / (max_val - min_val).max().clamp_min(1e-8)
                offset = (min_val + max_val) / 2

                # Transform positions and radii
                scaled_pos = (pos - offset) * scale
                scaled_radii = radii * scale

                # Calculate distance from grid to all spheres
                scaled_pos_reshaped = scaled_pos.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, N, 2)

                grid_dist = torch.norm(self.grid.unsqueeze(2) - scaled_pos_reshaped, dim=-1)

                # Create masks
                in_sphere = grid_dist <= scaled_radii  # (H, W, N)
                in_edge = (grid_dist >= (scaled_radii - self.edge_epsilon)) & in_sphere

                # Combine across spheres
                any_sphere = in_sphere.any(-1)  # (H, W)
                any_edge = in_edge.any(-1)

                # Create image: white=inside, black=edges, gray=background
                image = torch.full((256, 256), 192, dtype=torch.uint8, device=pos.device)
                image[any_sphere] = 255
                image[any_edge] = 0

                self.log_image('spheres', image, to_uint8=False)
        return total_loss


