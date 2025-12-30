import math

import cv2
import numpy as np
import torch
from torch import nn
from ...benchmark import Benchmark

class Smale7(Benchmark):
    """Smale's 7th problem - minimizes the potential energy V = sum_{i<j} -log(||x_i - x_j||^2), where x_i
    are points on the unit sphere S^2.

    Points are parameterized by spherical coordinates (theta, phi).

    Renders:
        points.

    Args:
        num_points (int): The number of points (N) on the sphere.
        initial_dist_epsilon (float): Small value to perturb initial positions
                                        to avoid stacking points and poles.
    """
    def __init__(self, num_points: int, initial_dist_epsilon: float = 1e-3, resolution=256, draw_lines=None):
        super().__init__()
        if num_points < 2:
            raise ValueError("Number of points must be at least 2.")
        self.num_points = num_points

        initial_thetas = torch.rand(num_points) * (math.pi - 2 * initial_dist_epsilon) + initial_dist_epsilon
        initial_phis = torch.rand(num_points) * (2 * math.pi)

        self.thetas = nn.Parameter(initial_thetas)
        self.phis = nn.Parameter(initial_phis)

        self.eps = 1e-12

        if draw_lines is None: draw_lines = num_points < 12
        self.draw_lines = draw_lines
        self.resolution = resolution

    def spherical_to_cartesian(self, thetas: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """Converts spherical coordinates (unit radius) to Cartesian coordinates."""
        x = torch.sin(thetas) * torch.cos(phis)
        y = torch.sin(thetas) * torch.sin(phis)
        z = torch.cos(thetas)
        # (num_points, 3)
        coords = torch.stack([x, y, z], dim=1)
        return coords

    @torch.no_grad
    def _make_frame(
        self,
        coords: torch.Tensor,
        img_size: int = 512,
        point_radius: int = 5,
        draw_lines: bool = False,
        line_thickness: int = 1,
        line_color: tuple[int, int, int] = (70, 70, 70) # Faint grey BGR
        ) -> np.ndarray:
        frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cv2.circle(frame, (img_size // 2, img_size // 2), img_size // 2 - 1, (50, 50, 50), 1, cv2.LINE_AA) # pylint:disable=no-member

        coords_np = coords.detach().cpu().numpy()

        # Project onto xy plane and scale to image coordinates
        # x, y are in [-1, 1], map to [0, img_size]
        img_coords = []
        for i in range(self.num_points):
            x, y, z = coords_np[i]
            # Scale x, y from [-1, 1] to [0, img_size]
            img_x = int((x + 1.0) / 2.0 * img_size)
            img_y = int((y + 1.0) / 2.0 * img_size)
            img_coords.append(((img_x, img_y), z))

        # Draw lines firstso points are drawn on top
        if draw_lines:
            for i in range(self.num_points):
                pt1, _ = img_coords[i]
                for j in range(i + 1, self.num_points):
                    pt2, _ = img_coords[j]
                    cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA) # pylint:disable=no-member

        # Points (circles)
        for i in range(self.num_points):
            (img_x, img_y), z = img_coords[i]

            # Color/brightness to indicate depth (z coordinate)
            intensity = int((z + 1.0) / 2.0 * 200) + 55 # Map z=[-1,1] to brightness [55, 255]
            color = (intensity // 2, intensity // 2, intensity) # BGR, bias towards blue/white

            cv2.circle(frame, (img_x, img_y), point_radius, color, -1, cv2.LINE_AA) # filled circle # pylint:disable=no-member

        return frame

    def get_loss(self) -> torch.Tensor:
        cartesian = self.spherical_to_cartesian(self.thetas, self.phis)
        pdists = torch.cdist(cartesian, cartesian, p=2)
        pdists_sq = pdists.pow(2)
        log_potential = -torch.log(pdists_sq + self.eps)
        loss = torch.triu(log_potential, diagonal=1).sum()

        if self._make_images:
            frame = self._make_frame(cartesian, img_size=self.resolution, draw_lines=self.draw_lines)
            self.log_image('solution', frame, to_uint8=False, show_best=True)

        return loss
