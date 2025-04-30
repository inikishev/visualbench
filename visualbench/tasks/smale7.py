import math

import cv2
import numpy as np
import torch
from torch import nn
from ..benchmark import Benchmark

# note: predominantly Gemini's work
class Smale7(Benchmark):
    """
    Smale's 7th problem - minimizes the potential energy V = sum_{i<j} -log(||x_i - x_j||^2), where x_i
    are points on the unit sphere S^2.
    Points are parameterized by spherical coordinates (theta, phi).

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

        # Initialize parameters (theta, phi)
        # Theta in [eps, pi - eps], Phi in [0, 2*pi)
        initial_thetas = torch.rand(num_points) * (math.pi - 2 * initial_dist_epsilon) + initial_dist_epsilon
        initial_phis = torch.rand(num_points) * (2 * math.pi)

        self.thetas = nn.Parameter(initial_thetas)
        self.phis = nn.Parameter(initial_phis)

        # Small value to avoid log(0) in loss calculation
        self.eps = 1e-12 # Needs to be small but non-zero

        if draw_lines is None: draw_lines = num_points < 12
        self.draw_lines = draw_lines
        self.resolution = resolution
        self.set_display_best('image')

    def spherical_to_cartesian(self, thetas: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """Converts spherical coordinates (unit radius) to Cartesian coordinates."""
        x = torch.sin(thetas) * torch.cos(phis)
        y = torch.sin(thetas) * torch.sin(phis)
        z = torch.cos(thetas)
        # Shape: (num_points, 3)
        coords = torch.stack([x, y, z], dim=1)
        return coords

    def generate_frame(
        self,
        coords: torch.Tensor,
        img_size: int = 512,
        point_radius: int = 5,
        draw_lines: bool = False,
        line_thickness: int = 1,
        line_color: tuple[int, int, int] = (70, 70, 70) # Faint grey BGR
        ) -> np.ndarray:
        """
        Generates a cv2 image visualizing the points (orthographic projection).

        Args:
            coords (torch.Tensor): Cartesian coordinates of the points (N, 3).
            img_size (int): Size (width and height) of the output image.
            point_radius (int): Radius of the circles representing points.
            draw_lines (bool): If True, draws lines between all pairs of points.
            line_thickness (int): Thickness of the lines between points.
            line_color (Tuple[int, int, int]): BGR color for the lines.

        Returns:
            np.ndarray: Visualization frame as a uint8 NumPy array (BGR).
        """
        frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # Draw a faint circle for the sphere's equator projection
        cv2.circle(frame, (img_size // 2, img_size // 2), img_size // 2 - 1, (50, 50, 50), 1, cv2.LINE_AA) # pylint:disable=no-member

        # Detach coords from computation graph and move to CPU for numpy/cv2
        coords_np = coords.detach().cpu().numpy()

        # Project onto xy plane and scale to image coordinates
        # x, y are in [-1, 1], map to [0, img_size]
        img_coords = []
        for i in range(self.num_points):
            x, y, z = coords_np[i]
            # Scale x, y from [-1, 1] to [0, img_size]
            img_x = int((x + 1.0) / 2.0 * img_size)
            img_y = int((y + 1.0) / 2.0 * img_size) # OpenCV y is down, but projection is fine
            img_coords.append(((img_x, img_y), z)) # Store scaled coords and original z

        # Draw lines first (if requested) so points are drawn on top
        if draw_lines:
            for i in range(self.num_points):
                pt1, _ = img_coords[i]
                for j in range(i + 1, self.num_points):
                    pt2, _ = img_coords[j]
                    cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA) # pylint:disable=no-member

        # Draw points (circles)
        for i in range(self.num_points):
            (img_x, img_y), z = img_coords[i]

            # Use color/brightness to indicate depth (z coordinate)
            intensity = int((z + 1.0) / 2.0 * 200) + 55 # Map z=[-1,1] to brightness [55, 255]
            color = (intensity // 2, intensity // 2, intensity) # BGR, bias towards blue/white

            cv2.circle(frame, (img_x, img_y), point_radius, color, -1, cv2.LINE_AA) # filled circle # pylint:disable=no-member

        return frame

    def get_loss(self) -> torch.Tensor:
        # 1. Convert current parameters (thetas, phis) to Cartesian coordinates
        # Clamp thetas to avoid issues near poles if optimizer goes wild,
        # although usually not strictly necessary with spherical_to_cartesian's trig.
        # clamped_thetas = torch.clamp(self.thetas, self.eps, math.pi - self.eps)
        # Using original thetas, relying on trig functions
        cartesian_coords = self.spherical_to_cartesian(self.thetas, self.phis)

        # 2. Calculate pairwise Euclidean distances
        # p=2 means Euclidean distance. Output shape: (num_points, num_points)
        pairwise_distances = torch.cdist(cartesian_coords, cartesian_coords, p=2)

        # 3. Calculate potential energy: V = sum_{i<j} -log(||x_i - x_j||^2)
        # Calculate squared distances
        pairwise_distances_sq = pairwise_distances.pow(2)

        # Calculate the logarithmic potential energy (ensure input to log is > 0)
        # Adding epsilon inside the log prevents log(0) -> -inf
        log_potential = -torch.log(pairwise_distances_sq + self.eps)

        # Sum over unique pairs (i < j) using upper triangle (excluding diagonal)
        loss = torch.triu(log_potential, diagonal=1).sum()

        if self._make_images:
            frame = self.generate_frame(
                cartesian_coords,
                img_size=self.resolution,
                draw_lines=self.draw_lines
            )
            self.log('image', frame, log_test=False, to_uint8=False)

        return loss
