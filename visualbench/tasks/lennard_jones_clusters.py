import matplotlib.pyplot as plt

import torch
import numpy as np
import cv2
from ..benchmark import Benchmark

#  function from https://docs.evotorch.ai/v0.1.0/examples/notebooks/Minimizing_Lennard-Jones_Atom_Cluster_Potentials/#benchmarking-snes
def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    # Assume positions has shape [B, 3N] where B is the batch size and N is the number of atoms
    # Reshaping to get individual atoms' positions of shape [B, N, 3]
    positions = positions.view(positions.shape[0], -1, 3)
    # Compute the pairwise differences
    # Subtracting [B, 1, N, 3] from [B, N, 1, 3] gives [B, N, N, 3]
    deltas = (positions.unsqueeze(2) - positions.unsqueeze(1))
    # Norm the differences gives [B, N, N]
    distances = torch.norm(deltas, dim = -1)
    return distances

#  function from https://docs.evotorch.ai/v0.1.0/examples/notebooks/Minimizing_Lennard-Jones_Atom_Cluster_Potentials/#benchmarking-snes
def cluster_potential(positions: torch.Tensor) -> torch.Tensor:
    # Compute the pairwise distances of atoms
    distances = pairwise_distances(positions)

    # Compute the pairwise cost (1 / dist)^12 - (1 / dist)^ 6
    pairwise_cost = (1 / distances).pow(12) - (1 / distances).pow(6.)

    # Get the upper triangle matrix (ignoring the diagonal)
    ut_pairwise_cost = torch.triu(pairwise_cost, diagonal = 1)

    # 4 * Summutation of the upper triangle of pairwise costs gives potential
    potential = 4 * ut_pairwise_cost.sum(dim = (1, 2))
    return potential



def visualize_cluster(
    positions: torch.Tensor,
    img_size: int = 512,
    padding: int = 50,
    dot_radius: int = 10,
    dot_color: tuple[int, int, int] = (0, 255, 0), # BGR for OpenCV (Green)
    bg_color: tuple[int, int, int] = (0, 0, 0),   # Black background
    depth_cueing: bool = True, # Scale size/color by Z-coordinate
    depth_strength: float = 0.7 # How much Z affects size/color (0 to 1)
) -> np.ndarray:
    """
    Creates an image visualization of a single atom cluster configuration.

    Args:
        positions: Tensor of shape [B, 3N] containing atom positions.
        batch_index: The index of the cluster within the batch to visualize.
        img_size: The width and height of the output square image in pixels.
        padding: Padding around the border of the image in pixels.
        dot_radius: The base radius of the dots representing atoms.
        dot_color: The base color (BGR tuple) of the dots.
        bg_color: The background color (BGR tuple) of the image.
        depth_cueing: If True, scale dot size and intensity based on Z-coordinate.
        depth_strength: Controls the intensity of depth cueing (0=none, 1=max).

    """
    # 1. Select the specific cluster and reshape
    pos_single = positions.nan_to_num() # Shape [3N]
    if pos_single.shape[0] % 3 != 0:
        raise ValueError("Input tensor second dimension must be divisible by 3 (3N)")
    num_atoms = pos_single.shape[0] // 3
    if num_atoms == 0: # Handle case with no atoms
        img = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)
        return img

    pos_xyz = pos_single.view(num_atoms, 3) # Shape [N, 3]

    # 2. Convert to NumPy array (detaching if it has gradients)
    pos_np = pos_xyz.detach().cpu().numpy() # Shape [N, 3]

    # 3. Get coordinate ranges for scaling (using XY for projection)
    x_coords = pos_np[:, 0]
    y_coords = pos_np[:, 1]
    z_coords = pos_np[:, 2]

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    min_z, max_z = z_coords.min(), z_coords.max()

    # Add small epsilon to range calculation to avoid division by zero if all points coincide
    epsilon = 1e-6
    range_x = max(max_x - min_x, epsilon)
    range_y = max(max_y - min_y, epsilon)
    range_z = max(max_z - min_z, epsilon)

    # 4. Calculate scaling factor to fit within padding
    draw_area_size = img_size - 2 * padding
    if draw_area_size <= 0:
        raise ValueError("Image size must be larger than 2 * padding")

    scale_factor = draw_area_size / max(range_x, range_y)

    # 5. Calculate center offsets for centering the drawing
    center_x_world = (min_x + max_x) / 2.0
    center_y_world = (min_y + max_y) / 2.0
    img_center = img_size / 2.0

    # 6. Create blank image
    img = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

    # 7. Prepare atom data (including Z for sorting/depth cueing)
    atom_data = []
    for i in range(num_atoms):
        x, y, z = pos_np[i]

        # --- Calculate image coordinates ---
        img_x = int(img_center + scale_factor * (x - center_x_world))
        img_y = int(img_center + scale_factor * (y - center_y_world)) # Y axis often inverted in image coords, but centering handles it.

        # --- Calculate depth effect (optional) ---
        current_radius = dot_radius
        current_color = dot_color
        if depth_cueing and range_z > epsilon:
            # Normalize Z: 0 (closest) to 1 (farthest)
            norm_z = (z - min_z) / range_z
            # Clamp depth strength
            clamped_strength = max(0.0, min(1.0, depth_strength))

            # Scale radius: smaller for farther atoms
            radius_scale = 1.0 - clamped_strength * norm_z * 0.8 # Max 80% reduction
            current_radius = max(1, int(dot_radius * radius_scale))

            # Scale color intensity: darker for farther atoms
            color_scale = 1.0 - clamped_strength * norm_z * 0.6 # Max 60% reduction
            current_color = tuple(int(c * color_scale) for c in dot_color)

        atom_data.append({'img_x': img_x, 'img_y': img_y, 'z': z, 'radius': current_radius, 'color': current_color})

    # 8. Sort atoms by Z-coordinate (draw farthest first for better occlusion)
    if depth_cueing:
        atom_data.sort(key=lambda item: item['z'], reverse=True) # Sort descending Z

    # 9. Draw atoms
    for atom in atom_data:
        cv2.circle( # pylint:disable=no-member
            img,
            (atom['img_x'], atom['img_y']),
            atom['radius'],
            atom['color'],
            thickness=-1, # Filled circle
            lineType=cv2.LINE_AA # Anti-aliasing# pylint:disable=no-member
        )

    return img


class LennardJonesClusters(Benchmark):
    """Lennard-Jones atom cluster potential minimisation. Notoriously difficult.
    I have not been able to find a point where gradients are not nan, so you probably need gradient free methods for this.

    Renders:
        visualization of the cluster.

    Here are some known minima https://doye.chem.ox.ac.uk/jon/structures/LJ.html"""
    def __init__(self, n):
        super().__init__()
        self.n = n

        self.positions = torch.nn.Parameter(torch.randn((n, 3), generator=self.rng.torch('cpu'))*10)

    def get_loss(self):
        if self._make_images:
            frame = visualize_cluster(self.positions.detach().cpu().view(-1)) # pylint:disable=not-callable
            self.log_image('cluster', frame, to_uint8=False)

        return cluster_potential(self.positions.unsqueeze(0))[0]