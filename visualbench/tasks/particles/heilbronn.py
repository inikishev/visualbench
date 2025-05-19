# pylint:disable=no-member
import cv2
import numpy as np
import torch
from torch import nn
from ...benchmark import Benchmark

class HeilbronnTrianglesProblem(Benchmark):
    def __init__(self, n, bounds=(0.0, 1.0), penalty=10.0, resolution=512):
        super().__init__(bounds=bounds)
        if n < 3:
            raise ValueError("Number of points must be at least 3 to form a triangle.")

        self.n = n
        self.bounds = bounds
        self.penalty = penalty
        self.resolution = resolution

        initial_points = torch.rand(n, 2) * (bounds[1] - bounds[0]) + bounds[0]
        self.points = nn.Parameter(initial_points)

        self.triangle_indices = nn.Buffer(torch.combinations(torch.arange(n), 3))


    def _calculate_triangle_areas(self, points):
        p1s = points[self.triangle_indices[:, 0]]
        p2s = points[self.triangle_indices[:, 1]]
        p3s = points[self.triangle_indices[:, 2]]

        # Area = 0.5 * |x1(y2−y3) + x2(y3−y1) + x3(y1−y2)|
        # Shoelace formula component: x1y2 - y1x2 + x2y3 - y2x3 + x3y1 - y3x1
        areas = 0.5 * torch.abs(
            p1s[:, 0] * (p2s[:, 1] - p3s[:, 1]) +
            p2s[:, 0] * (p3s[:, 1] - p1s[:, 1]) +
            p3s[:, 0] * (p1s[:, 1] - p2s[:, 1])
        )
        return areas

    def _boundary_penalty(self):
        assert self.bounds is not None
        penalty = 0
        penalty += torch.relu(self.bounds[0] - self.points).pow(2).sum()
        penalty += torch.relu(self.points - self.bounds[1]).pow(2).sum()
        return penalty * self.penalty

    @torch.no_grad
    def _make_frame(self, points, smallest_triangle_indices_np=None, min_area_val=0.0):
        frame = np.ones((self.resolution, self.resolution, 3), dtype=np.uint8) * 255

        def scale_point(p):
            assert self.bounds is not None
            x = int(((p[0] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])) * self.resolution)
            y = int(((p[1] - self.bounds[0]) / (self.bounds[1] - self.bounds[0])) * self.resolution)
            # x = np.clip(x, 0, self.resolution - 1)
            # y = np.clip(y, 0, self.resolution - 1)
            return (x, y)

        # Draw all points
        for i in range(points.shape[0]):
            pt_scaled = scale_point(points[i])
            try: cv2.circle(frame, pt_scaled, 5, (0, 0, 255), -1) # Red points
            except Exception: pass

        # Draw smallest triangle
        if smallest_triangle_indices_np is not None and len(smallest_triangle_indices_np) == 3:
            tri_pts = np.array([
                scale_point(points[smallest_triangle_indices_np[0]]),
                scale_point(points[smallest_triangle_indices_np[1]]),
                scale_point(points[smallest_triangle_indices_np[2]])
            ], dtype=np.int32)
            try: cv2.polylines(frame, [tri_pts], isClosed=True, color=(0, 255, 0), thickness=2) # Green triangle
            except Exception: pass

        # Add text for min area
        cv2.putText(frame, f"Min Area: {min_area_val:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        return frame

    def get_loss(self):
        points = self.points

        triangle_areas = self._calculate_triangle_areas(points)
        min_area = torch.min(triangle_areas)
        min_area_idx = torch.argmin(triangle_areas)
        smallest_triangle_indices = self.triangle_indices[min_area_idx].detach().cpu().numpy()

        loss = -min_area + self._boundary_penalty()

        if self._make_images:
            frame = self._make_frame(points.detach().cpu().numpy(), smallest_triangle_indices, min_area.item()) # pylint:disable=not-callable
            self.log_image("solution", frame, to_uint8=False, show_best=True)

        return loss
