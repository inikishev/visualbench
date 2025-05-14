# pylint:disable = no-member
import cv2  # Make sure you have opencv-python installed: pip install opencv-python
import numpy as np
import torch
from torch import nn
from ...benchmark import Benchmark


class ClosestFurthestParticles(Benchmark):
    """Objective is to maximize distance between 2 closest points and minimize distance between 2 furthest points. This is a sub-differentiable analogy to an ill conditioned objective, since you only get feedback from 2 pairs at a time."""
    def __init__(self, n_points: int=20, w_pull:float=1, w_push:float=1.5, initial_spread_factor: float = 0.5, image_size: int = 512, ):
        super().__init__(log_projections=True)
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        self.n_points = n_points
        self.w_pull = w_pull
        self.w_push = w_push
        self.image_size = image_size

        start = 0.5 - initial_spread_factor / 2.0
        initial_points = torch.rand(n_points, 2) * initial_spread_factor + start
        self.points = nn.Parameter(initial_points)

        self.color_normal = (200, 200, 200)   # Light grey for normal points
        self.color_closest = (0, 255, 0)     # Green for closest points
        self.color_furthest = (0, 0, 255)    # Red for furthest points
        self.color_line_closest = (0, 180, 0)  # Darker Green for line
        self.color_line_furthest = (0, 0, 180) # Darker Red for line

    def get_loss(self):
        pdist05_matrix = torch.cdist(self.points, self.points, p=0.5) # L0.5 norm
        pdist2_matrix = torch.cdist(self.points, self.points, p=2) # L2 norm
        pdist05_matrix = pdist05_matrix + torch.eye(self.n_points, device=self.points.device) * (pdist05_matrix.detach().amax() + 1)

        min_dist_val, min_flat_idx = torch.min(pdist05_matrix.view(-1), dim=0)
        min_idx1 = min_flat_idx // self.n_points
        min_idx2 = min_flat_idx % self.n_points

        max_dist_val, max_flat_idx = torch.max(pdist2_matrix.view(-1), dim=0)
        max_idx1 = max_flat_idx // self.n_points
        max_idx2 = max_flat_idx % self.n_points

        penalty = self.points.clip(max=0).pow(2).sum() + (self.points-1).clip(min=0).pow(2).sum()
        loss = penalty + max_dist_val*self.w_pull - min_dist_val*self.w_push

        if self._make_images:
            points_relative_np = self.points.detach().cpu().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).numpy()
            points_pixel_np = points_relative_np * self.image_size # Scale to pixel coordinates

            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Get pixel coordinates of special points
            pt_closest1_px = tuple(points_pixel_np[min_idx1].astype(int))
            pt_closest2_px = tuple(points_pixel_np[min_idx2].astype(int))
            pt_furthest1_px = tuple(points_pixel_np[max_idx1].astype(int))
            pt_furthest2_px = tuple(points_pixel_np[max_idx2].astype(int))

            try:
                # Draw lines first, so circles are on top
                cv2.line(frame, pt_closest1_px, pt_closest2_px, self.color_line_closest, 1)
                cv2.line(frame, pt_furthest1_px, pt_furthest2_px, self.color_line_furthest, 1)

                # Draw all points
                for i in range(self.n_points):
                    pt_px = tuple(points_pixel_np[i].astype(int))
                    color = self.color_normal
                    radius = 3
                    # Check if this point is part of a special pair
                    if i == min_idx1 or i == min_idx2:
                        color = self.color_closest
                        radius = 5
                    # A point can be involved in both closest and furthest pairs (e.g., in a 3-point scenario)
                    # If so, furthest color might override closest if this specific ordering of if-blocks is kept.
                    if i == max_idx1 or i == max_idx2:
                        color = self.color_furthest
                        radius = 5

                    # Ensure points are within image bounds for drawing before cv2.circle
                    # cv2.circle will clip/error for out-of-bound points, this provides a manual clip.
                    draw_pt_px = (max(0, min(pt_px[0], self.image_size - 1)),
                                  max(0, min(pt_px[1], self.image_size - 1)))
                    cv2.circle(frame, draw_pt_px, radius, color, -1) # -1 for filled circle

                # Add text for distances (these are relative distances from 0-1 space)
                cv2.putText(frame, f"Min D (rel): {min_dist_val.item():.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_closest, 1)
                cv2.putText(frame, f"Max D (rel): {max_dist_val.item():.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_furthest, 1)
                cv2.putText(frame, f"Loss: {loss.item():.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_normal, 1)
            except Exception as e:
                # print(f"Error during visualization: {e}") # Optional: log error
                pass

            self.log('image', frame, log_test=False, to_uint8=False) # frame is already uint8

        return loss
