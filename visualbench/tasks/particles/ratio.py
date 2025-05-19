# pylint:disable=no-member
import math

import cv2
import numpy as np
import torch
from torch import nn

from ...benchmark import Benchmark

class ParticleDistanceRatio(Benchmark):
    """minimize ratio between furthest and closest points"""
    def __init__(
        self,
        n: int,
        bounds: tuple = (0, 0, 100, 100),  # (x_min, y_min, x_max, y_max)
        penalty: float = 10.0,
        eps: float = 1e-6,
        spread: float = 0.8,
    ):
        super().__init__()

        if n < 2:
            raise ValueError("n_points must be at least 2.")

        self.n = n
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.penalty = penalty
        self.eps = eps

        w = self.xmax - self.xmin
        h = self.ymax - self.ymin

        center_x = self.xmin + w / 2
        center_y = self.ymin + h / 2

        # Random points in [-0.5, 0.5] range, then scale and shift
        initial_points = torch.rand(n, 2) - 0.5 # in [-0.5, 0.5]
        initial_points[:, 0] = center_x + initial_points[:, 0] * w * spread
        initial_points[:, 1] = center_y + initial_points[:, 1] * h * spread

        self.points = nn.Parameter(initial_points)

    def _calculate_distances(self):
        # Pairwise distances: shape (n_points, n_points)
        dist_matrix = torch.cdist(self.points, self.points, p=2)
        return dist_matrix

    def _get_min_max_distances_and_indices(self, dist_matrix):
        max_dist = dist_matrix.max()
        I = torch.eye(self.n, device=dist_matrix.device, dtype=torch.bool)
        dist_no_diag = dist_matrix.clone()
        dist_no_diag[I] = float('inf')
        min_dist = dist_no_diag.min()

        argmax = dist_matrix.argmax()
        max1, max2 = np.unravel_index(argmax.item(), dist_matrix.shape)

        argmin = dist_no_diag.argmin()
        min1, min2 = np.unravel_index(argmin.item(), dist_no_diag.shape)

        return min_dist, max_dist, (min1, min2), (max1, max2)

    def _oob_penalty(self):
        penalty = 0.0
        # x < x_min
        penalty += torch.relu(self.xmin - self.points[:, 0]).sum()
        # x > x_max
        penalty += torch.relu(self.points[:, 0] - self.xmax).sum()
        # y < y_min
        penalty += torch.relu(self.ymin - self.points[:, 1]).sum()
        # y > y_max
        penalty += torch.relu(self.points[:, 1] - self.ymax).sum()
        return penalty

    @torch.no_grad
    def _make_frame(self, current_points_np, min_dist_indices, max_dist_indices,
                      frame_size=(640, 480), fov_padding_factor=0.1):

        world_x_min, world_y_min, world_x_max, world_y_max = self.xmin, self.ymin, self.xmax, self.ymax
        world_width = world_x_max - world_x_min
        world_height = world_y_max - world_y_min

        # FOV bounds
        fov_x_min = world_x_min - world_width * fov_padding_factor
        fov_y_min = world_y_min - world_height * fov_padding_factor
        fov_x_max = world_x_max + world_width * fov_padding_factor
        fov_y_max = world_y_max + world_height * fov_padding_factor

        fov_width = fov_x_max - fov_x_min
        fov_height = fov_y_max - fov_y_min

        frame_w_px, frame_h_px = frame_size

        frame = np.ones((frame_h_px, frame_w_px, 3), dtype=np.uint8) * 240 # Light gray background

        if fov_width == 0 or fov_height == 0: # Avoid division by zero if rect is a line/point
            scale = 1
        else:
            scale_x = frame_w_px / fov_width
            scale_y = frame_h_px / fov_height
            scale = min(scale_x, scale_y) # Maintain aspect ratio

        def world_to_pixel(wx, wy):
            offset_x = (frame_w_px - fov_width * scale) / 2
            offset_y = (frame_h_px - fov_height * scale) / 2

            px = int(offset_x + (wx - fov_x_min) * scale)
            py = int(offset_y + (wy - fov_y_min) * scale)
            return px, py

        rect_start_px = world_to_pixel(self.xmin, self.ymin)
        rect_end_px = world_to_pixel(self.xmax, self.ymax)
        cv2.rectangle(frame, rect_start_px, rect_end_px, (150, 150, 150), 2)


        point_radius = max(3, int(0.01 * min(frame_w_px, frame_h_px))) # Dynamic radius
        for i in range(self.n):
            px, py = world_to_pixel(current_points_np[i, 0], current_points_np[i, 1])
            cv2.circle(frame, (px, py), point_radius, (255, 0, 0), -1) # Blue points


        if self.n >= 2 and min_dist_indices:
            pt1_idx, pt2_idx = min_dist_indices
            p1 = world_to_pixel(current_points_np[pt1_idx, 0], current_points_np[pt1_idx, 1])
            p2 = world_to_pixel(current_points_np[pt2_idx, 0], current_points_np[pt2_idx, 1])
            cv2.line(frame, p1, p2, (0, 255, 0), 2) # Green line


        if self.n >=2 and max_dist_indices:
            pt1_idx, pt2_idx = max_dist_indices
            p1 = world_to_pixel(current_points_np[pt1_idx, 0], current_points_np[pt1_idx, 1])
            p2 = world_to_pixel(current_points_np[pt2_idx, 0], current_points_np[pt2_idx, 1])
            cv2.line(frame, p1, p2, (0, 0, 255), 2) # Red line

        return frame


    def get_loss(self):
        dist_matrix = self._calculate_distances()

        min_dist, max_dist, min_indices, max_indices = \
            self._get_min_max_distances_and_indices(dist_matrix)

        loss = max_dist / (min_dist + self.eps)
        loss = loss + self._oob_penalty()

        if self._make_images:
            frame = self._make_frame(self.points.detach().cpu().numpy(), min_indices, max_indices) # pylint:disable=not-callable
            self.log_image('points', frame, to_uint8=False, show_best=True)

        return loss
