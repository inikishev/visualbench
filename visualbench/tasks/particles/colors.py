import cv2
import matplotlib.pyplot as plt  # For a good color palette
import numpy as np
import torch
import torch.nn as nn

from ...benchmark import Benchmark


# Helper to get distinct colors
def get_distinct_colors(n):
    """Gets n distinct BGR colors for OpenCV."""
    cmap = plt.get_cmap('viridis', n) # You can use other cmaps like 'tab10', 'hsv'
    colors_rgba = [cmap(i) for i in range(n)]
    colors_bgr = [(int(r*255), int(g*255), int(b*255)) for b,g,r,_ in colors_rgba] # OpenCV uses BGR
    return colors_bgr

class ColoredParticles(Benchmark):
    def __init__(self, ppg, ngroups,
                 repulsion=1.0, attraction=0.1, repulsion_radius=0.5,
                 world_size=1.0, viz_size=512):
        """
        Particles of same color are repulsed from each other, particles of different colors attract each other.

        Args:
            num_points_per_group (list or int): Number of points in each group.
                                                If int, all groups have this many points.
            num_groups (int): Number of distinct groups/colors.
            k_repulsion (float): Strength of repulsion force.
            k_attraction (float): Strength of attraction force.
            repulsion_radius_factor (float): Factor to scale distances for repulsion.
                                             A smaller value means repulsion is stronger at short distances.
                                             Essentially 1/(d/factor + eps).
            world_size (float): The conceptual size of the world where points exist (e.g., 0 to world_size).
            viz_size (int): The pixel dimension of the square visualization image.
            device (str): PyTorch device ('cpu' or 'cuda').
        """
        super().__init__(log_projections=True)

        if isinstance(ppg, int):
            ppg = [ppg] * ngroups
        if len(ppg) != ngroups:
            raise ValueError("Length of num_points_per_group must match num_groups")

        self.num_groups = ngroups
        self.total_points = sum(ppg)
        self.world_size = world_size
        self.viz_size = viz_size

        # params
        initial_points = torch.rand(self.total_points, 2) * self.world_size
        self.points = nn.Parameter(initial_points)

        # buffers
        self.k_repulsion = nn.Buffer(torch.tensor(repulsion))
        self.k_attraction = nn.Buffer(torch.tensor(attraction))
        self.epsilon = nn.Buffer(torch.tensor(1e-6))
        self.repulsion_radius_factor = nn.Buffer(torch.tensor(repulsion_radius * self.world_size))

        # vis
        group_ids = []
        for i, num_in_group in enumerate(ppg):
            group_ids.extend([i] * num_in_group)
        self.group_ids = nn.Buffer(torch.tensor(group_ids, dtype=torch.long))

        self.cv2_colors = get_distinct_colors(self.num_groups)
        self.point_radius_px = max(3, viz_size // 100) # Radius of points in visualization

    def _generate_frame(self):
        """Generates a uint8 image representing the current solution and appends it to self.frames."""
        # Detach points from computation graph and move to CPU for numpy conversion
        points_np = self.points.data.clone().detach().cpu().numpy()
        group_ids_np = self.group_ids.cpu().numpy()

        # Create a blank white image (BGR)
        frame = np.ones((self.viz_size, self.viz_size, 3), dtype=np.uint8) * 255

        for i in range(self.total_points):
            x, y = points_np[i]
            group_id = group_ids_np[i]
            color = self.cv2_colors[group_id]

            # Scale points from world coordinates to image pixel coordinates
            # Assuming world_size maps to viz_size
            center_x = int(x / self.world_size * self.viz_size)
            center_y = int(y / self.world_size * self.viz_size)

            # Clip coordinates to be within image boundaries for drawing
            center_x = np.clip(center_x, self.point_radius_px, self.viz_size - self.point_radius_px -1)
            center_y = np.clip(center_y, self.point_radius_px, self.viz_size - self.point_radius_px -1)

            cv2.circle(frame, (center_x, center_y), self.point_radius_px, color, -1) # pylint:disable=no-member
            cv2.circle(frame, (center_x, center_y), self.point_radius_px, (0,0,0), 1) # Black outline  # pylint:disable=no-member

        return frame

    def get_loss(self):
        """
        Calculates the loss based on repulsion and attraction.
        Generates and stores a frame for visualization.
        """
        dists = torch.cdist(self.points, self.points, p=2)
        same_group_mask = self.group_ids.unsqueeze(0) == self.group_ids.unsqueeze(1)
        different_group_mask = ~same_group_mask
        identity_mask = torch.eye(self.total_points, dtype=torch.bool, device=self.device)
        valid_same_group_mask = same_group_mask & ~identity_mask
        valid_different_group_mask = different_group_mask # Already excludes diagonal due to same_group_mask

        repulsion_potential = self.k_repulsion / ( (dists / self.repulsion_radius_factor) + self.epsilon)
        loss_repulsion = torch.sum(repulsion_potential * valid_same_group_mask) / 2.0

        attraction_potential = self.k_attraction * (dists**2)
        loss_attraction = torch.sum(attraction_potential * valid_different_group_mask) / 2.0

        total_loss = loss_repulsion + loss_attraction

        out_of_bounds_lower = torch.relu(-self.points) # positive if points < 0
        out_of_bounds_upper = torch.relu(self.points - self.world_size) # positive if points > world_size
        boundary_penalty = torch.sum(out_of_bounds_lower**2) + torch.sum(out_of_bounds_upper**2)
        total_loss += 0.1 * boundary_penalty


        if self._make_images:
            self.log('image', self._generate_frame(), log_test=False, to_uint8=False)

        return total_loss
