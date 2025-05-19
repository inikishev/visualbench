import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ...benchmark import Benchmark


def _get_colors(n):
    cmap = plt.get_cmap('rainbow', n)
    rbga = [cmap(i) for i in range(n)]
    bgr = [(int(r*255), int(g*255), int(b*255)) for b,g,r,_ in rbga]
    return bgr

class ColoredParticles(Benchmark):
    def __init__(
        self,
        n_groups,
        points_per_group,
        repulsion=1.0,
        attraction=0.1,
        repulsion_dist=0.5,
        world_size=1.0,
        resolution=512,
    ):
        """Particles of same color are repulsed from each other, particles of different colors attract each other."""
        super().__init__()

        if isinstance(points_per_group, int):
            points_per_group = [points_per_group] * n_groups
        if len(points_per_group) != n_groups:
            raise ValueError("Length of num_points_per_group must match num_groups")

        self.num_groups = n_groups
        self.total_points = sum(points_per_group)
        self.world_size = world_size
        self.resolution = resolution

        # params
        initial_points = torch.rand(self.total_points, 2) * self.world_size
        self.points = nn.Parameter(initial_points)

        # buffers
        self.k_repulsion = nn.Buffer(torch.tensor(repulsion))
        self.k_attraction = nn.Buffer(torch.tensor(attraction))
        self.eps = nn.Buffer(torch.tensor(1e-6))
        self.repulsion_dist = nn.Buffer(torch.tensor(repulsion_dist * self.world_size))
        # smaller = more repulsion at short distances

        # vis
        group_ids = []
        for i, num_in_group in enumerate(points_per_group):
            group_ids.extend([i] * num_in_group)
        self.group_ids = nn.Buffer(torch.tensor(group_ids, dtype=torch.long))

        self.cv2_colors = _get_colors(self.num_groups)
        self.point_radius_px = max(3, resolution // 100)

    @torch.no_grad
    def _make_frame(self):
        points = self.points.data.clone().detach().cpu().numpy()
        group_ids = self.group_ids.cpu().numpy()

        frame = np.ones((self.resolution, self.resolution, 3), dtype=np.uint8) * 255

        for i in range(self.total_points):
            x, y = points[i]
            group_id = group_ids[i]
            color = self.cv2_colors[group_id]

            center_x = int(x / self.world_size * self.resolution)
            center_y = int(y / self.world_size * self.resolution)

            center_x = np.clip(center_x, self.point_radius_px, self.resolution - self.point_radius_px -1)
            center_y = np.clip(center_y, self.point_radius_px, self.resolution - self.point_radius_px -1)

            cv2.circle(frame, (center_x, center_y), self.point_radius_px, color, -1) # pylint:disable=no-member
            cv2.circle(frame, (center_x, center_y), self.point_radius_px, (0,0,0), 1) # black outline  # pylint:disable=no-member

        return frame

    def get_loss(self):
        dists = torch.cdist(self.points, self.points, p=2)
        same_mask = self.group_ids.unsqueeze(0) == self.group_ids.unsqueeze(1)
        diff_mask = ~same_mask
        id_mask = torch.eye(self.total_points, dtype=torch.bool, device=self.device)
        same_mask = same_mask & ~id_mask

        repulsion_potential = self.k_repulsion / ( (dists / self.repulsion_dist) + self.eps)
        repulsion = torch.sum(repulsion_potential * same_mask) / 2.0

        attraction_potential = self.k_attraction * (dists**2)
        attraction = torch.sum(attraction_potential * diff_mask) / 2.0

        loss = repulsion + attraction

        lb = torch.relu(-self.points)
        ub = torch.relu(self.points - self.world_size)
        penalty = torch.sum(lb**2) + torch.sum(ub**2)
        loss = loss + 0.1 * penalty

        if self._make_images:
            self.log_image('points', self._make_frame(), to_uint8=False, show_best=True)

        return loss
