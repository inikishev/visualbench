# pylint:disable = no-member
import cv2
import numpy as np
import torch
from torch import nn

from ...benchmark import Benchmark


class ClosestFurthestParticles(Benchmark):
    """Objective is to maximize distance between 2 closest points and minimize distance between 2 furthest points. This is a sub-differentiable analogy to an ill conditioned objective, since you only get feedback from 2 pairs at a time."""
    def __init__(self, n: int=20, w_pull:float=1, w_push:float=1.5, spread: float = 0.5, resolution: int = 512):
        super().__init__()
        if n < 2:
            raise ValueError("n_points must be at least 2.")

        self.n = n
        self.w_pull = w_pull
        self.w_push = w_push
        self.resolution = resolution

        start = 0.5 - spread / 2.0
        initial_points = torch.rand(n, 2, generator=self.rng.torch()) * spread + start
        self.points = nn.Parameter(initial_points)

        self.color_normal = (200, 200, 200) # light grey
        self.color_closest = (0, 255, 0) # green
        self.color_furthest = (0, 0, 255) # red
        self.color_line_closest = (0, 180, 0)  # dark green
        self.color_line_furthest = (0, 0, 180) # dart red
        self._show_titles_on_video = False

    def get_loss(self):
        pdist05 = torch.cdist(self.points, self.points, p=0.5) # L0.5 norm
        pdist2 = torch.cdist(self.points, self.points, p=2) # L2 norm
        pdist05 = pdist05 + torch.eye(self.n, device=self.points.device) * (pdist05.detach().amax() + 1)

        min_dist, min_idx = torch.min(pdist05.view(-1), dim=0)
        min_idx1 = min_idx // self.n
        min_idx2 = min_idx % self.n

        max_dist, max_idx = torch.max(pdist2.view(-1), dim=0)
        max_idx1 = max_idx // self.n
        max_idx2 = max_idx % self.n

        penalty = self.points.clip(max=0).pow(2).sum() + (self.points-1).clip(min=0).pow(2).sum()
        loss = penalty + max_dist*self.w_pull - min_dist*self.w_push

        if self._make_images:
            with torch.no_grad():
                points_rel = self.points.detach().cpu().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clip(-1e10,1e10).numpy() # pylint:disable=not-callable
                points_px = points_rel * self.resolution # Scale to pixel coordinates

                frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

                # coords of closest and furthest points
                closest1 = tuple(points_px[min_idx1].astype(int))
                closest2 = tuple(points_px[min_idx2].astype(int))
                furthest1 = tuple(points_px[max_idx1].astype(int))
                furthest2 = tuple(points_px[max_idx2].astype(int))

                try:
                    cv2.line(frame, closest1, closest2, self.color_line_closest, 1)
                    cv2.line(frame, furthest1, furthest2, self.color_line_furthest, 1)

                    # points
                    for i in range(self.n):
                        point = tuple(points_px[i].astype(int))
                        color = self.color_normal
                        radius = 3

                        # point in closest pair
                        if i == min_idx1 or i == min_idx2: # pylint:disable=consider-using-in
                            color = self.color_closest
                            radius = 5

                        # point in furthest pair
                        if i == max_idx1 or i == max_idx2:# pylint:disable=consider-using-in
                            color = self.color_furthest
                            radius = 5

                        draw_pt_px = (max(0, min(point[0], self.resolution - 1)),
                                    max(0, min(point[1], self.resolution - 1)))
                        cv2.circle(frame, draw_pt_px, radius, color, -1)

                    cv2.putText(frame, f"Min D (rel): {min_dist.item():.2f}", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_closest, 1)
                    cv2.putText(frame, f"Max D (rel): {max_dist.item():.2f}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_furthest, 1)
                    cv2.putText(frame, f"Loss: {loss.item():.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_normal, 1)
                except Exception:
                    pass

                self.log_image('points', frame, to_uint8=False, show_best=True)

        return loss
