# pylint:disable = no-member
import cv2  # Make sure you have opencv-python installed: pip install opencv-python
import numpy as np
import torch
import torch.nn as nn

from ...benchmark import Benchmark


class ClosestFurthestParticles(Benchmark):
    """Objective is to maximize distance between 2 closest points and minimize distance between 2 furthest points. This is a sub-differentiable analogy to an ill conditioned objective, since you only get feedback from 2 pairs at a time."""
    def __init__(self, n_points: int, w_pull=1, w_push=3, initial_spread_factor: float = 0.8, image_size: int = 512, ):
        super().__init__(log_projections=True)
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        self.n_points = n_points
        self.w_pull = w_pull
        self.w_push = w_push
        self.image_size = image_size
        self.frames = [] # To store visualization frames

        # Initialize points randomly within the image, but not too close to edges initially
        # And ensure they are learnable parameters
        center = image_size / 2
        spread = image_size * initial_spread_factor / 2

        initial_points = torch.rand(n_points, 2) * spread * 2 + (center - spread)
        self.points = nn.Parameter(initial_points)

        # Define colors (BGR for OpenCV)
        self.color_normal = (200, 200, 200)   # Light grey for normal points
        self.color_closest = (0, 255, 0)     # Green for closest points
        self.color_furthest = (0, 0, 255)    # Red for furthest points
        self.color_line_closest = (0, 180, 0)  # Darker Green for line
        self.color_line_furthest = (0, 0, 180) # Darker Red for line

    def get_loss(self):
        pdist_matrix = torch.cdist(self.points, self.points, p=2) # L2 norm
        pdist_no_diag = pdist_matrix + torch.eye(self.n_points, device=self.points.device) * pdist_matrix.detach().amax()

        min_dist_val, min_flat_idx = torch.min(pdist_no_diag.view(-1), dim=0)
        min_idx1 = min_flat_idx // self.n_points
        min_idx2 = min_flat_idx % self.n_points

        max_dist_val, max_flat_idx = torch.max(pdist_matrix.view(-1), dim=0)
        max_idx1 = max_flat_idx // self.n_points
        max_idx2 = max_flat_idx % self.n_points
        loss = max_dist_val*self.w_pull - min_dist_val*self.w_push

        if self._make_images:
            points_np = self.points.detach().cpu().nan_to_num(0,0,0).numpy()
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Get coordinates of special points
            pt_closest1 = tuple(points_np[min_idx1].astype(int))
            pt_closest2 = tuple(points_np[min_idx2].astype(int))
            pt_furthest1 = tuple(points_np[max_idx1].astype(int))
            pt_furthest2 = tuple(points_np[max_idx2].astype(int))

            # Draw lines first, so circles are on top
            try:
                cv2.line(frame, pt_closest1, pt_closest2, self.color_line_closest, 1)
                cv2.line(frame, pt_furthest1, pt_furthest2, self.color_line_furthest, 1)

                # Draw all points
                for i in range(self.n_points):
                    pt = tuple(points_np[i].astype(int))
                    color = self.color_normal
                    radius = 3
                    # Check if this point is part of a special pair
                    if i == min_idx1 or i == min_idx2:
                        color = self.color_closest
                        radius = 5
                    if i == max_idx1 or i == max_idx2: # A point can be both closest and furthest to different points
                        color = self.color_furthest # Furthest color might override closest if one point is involved in both
                        radius = 5

                    # Ensure points are within image bounds for drawing
                    # cv2.circle will clip, but good practice to be mindful
                    draw_pt = (max(0, min(pt[0], self.image_size-1)), max(0, min(pt[1], self.image_size-1)))
                    cv2.circle(frame, draw_pt, radius, color, -1) # -1 for filled circle

                # Add text for distances
                cv2.putText(frame, f"Min D: {min_dist_val.item():.2f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_closest, 1)
                cv2.putText(frame, f"Max D: {max_dist_val.item():.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_furthest, 1)
                cv2.putText(frame, f"Loss: {loss.item():.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_normal, 1)
            except Exception as e:
                pass
            self.log('image', frame, log_test=False, to_uint8=False)

        return loss
