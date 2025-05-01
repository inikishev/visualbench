import math
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark
from .._utils import CUDA_IF_AVAILABLE

# Helper function for drawing (remains the same)
def draw_state(
    width: int,
    height: int,
    ball_positions: np.ndarray,
    radius: float,
    obstacles: list[dict],
    zones: list[dict],
    bg_color: tuple[int, int, int] = (240, 240, 240),
    ball_color: tuple[int, int, int] = (0, 0, 200),
    obstacle_color: tuple[int, int, int] = (50, 50, 50),
    zone_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Renders the current state of the marble race."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    if zone_colors is None:
        zone_colors = {
            "accelerator": (200, 255, 200),
            "liquid": (200, 200, 255),
            # Add more zone types and colors here
        }

    # Draw Zones first (background elements)
    for zone in zones:
        zone_type = zone.get('type', 'unknown')
        color = zone_colors.get(zone_type, (200, 200, 200)) # Default gray for unknown
        if 'rect' in zone:
            x_min, y_min, x_max, y_max = map(int, zone['rect'])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, -1) # pylint:disable=no-member
        # Add drawing for other zone shapes if needed

    # Draw Obstacles
    for obs in obstacles:
        obs_type = obs.get('type')
        if obs_type == 'circle':
            center = tuple(map(int, obs['center']))
            rad = int(obs['radius'])
            cv2.circle(img, center, rad, obstacle_color, -1) # pylint:disable=no-member
        elif obs_type == 'rect':
            x_min, y_min, x_max, y_max = map(int, obs['rect'])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), obstacle_color, -1) # pylint:disable=no-member
        # Add drawing for other obstacle types if needed

    # Draw Balls (ensure they are drawn last/on top)
    for i in range(ball_positions.shape[0]):
        center = tuple(map(int, ball_positions[i]))
        cv2.circle(img, center, int(radius), ball_color, -1) # pylint:disable=no-member
        # Optional: Draw black outline for better visibility
        cv2.circle(img, center, int(radius), (0,0,0), 1) # pylint:disable=no-member

    return img

# Define custom obstacles and zones (optional)
MARBLE_COURSE: dict[str, Any] = dict(
    obstacles = [
        {'type': 'rect', 'rect': [50, 150, 250, 170]},
        {'type': 'circle', 'center': [150, 300], 'radius': 40},
        {'type': 'rect', 'rect': [0, 400, 100, 420]},
        {'type': 'rect', 'rect': [200, 400, 300, 420]},
    ],
    zones = [
            {'type': 'accelerator', 'rect': [100, 400, 200, 450], 'accel': [0, 1.0], 'strength_key': 'boost'} # Give specific key
    ],
    zone_strengths = {
        "boost": 150.0 # Corresponds to the 'strength_key' above
    }
)
class MarbleRace(Benchmark):
    """
    Gemini coded this.

    A differentiable marble race simulation.
    """
    def __init__(
        self,
        num_marbles: int = 30,
        width: int = 300,
        height: int = 500,
        radius: float = 8.0,
        gravity_strength: float = -80.0, # NOTE: Negative strength for downward force
        repulsion_strength_bb: float = 50.0, # Ball-ball
        repulsion_strength_boundary: float = 100.0,
        repulsion_strength_obstacle: float = 100.0,
        obstacles: list[dict] | None = None,
        zones: list[dict] | None = None,
        zone_strengths: dict[str, float] | None = None,
        initial_positions: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device = CUDA_IF_AVAILABLE,
    ):
        super().__init__()
        self._device = device
        self._dtype = dtype

        self.num_marbles = num_marbles
        self.width = width
        self.height = height
        self.radius = torch.tensor(radius, device=device, dtype=dtype)

        # --- Simulation Parameters ---
        # IMPORTANT: Negative gravity_strength makes balls fall DOWN (increase y coord)
        # because potential U = strength * y, so Force = -dU/dy = -strength.
        # If strength is negative, Force is positive (downwards).
        self.gravity_strength = gravity_strength
        self.repulsion_strength_bb = repulsion_strength_bb
        self.repulsion_strength_boundary = repulsion_strength_boundary
        self.repulsion_strength_obstacle = repulsion_strength_obstacle
        self.zone_strengths = zone_strengths if zone_strengths is not None else {
            "accelerator": 100.0
        }

        # --- Geometry ---
        self.obstacles = obstacles if obstacles is not None else self._default_obstacles()
        self.zones = zones if zones is not None else self._default_zones()
        # Pre-process obstacles/zones to tensors on the correct device
        self._preprocess_geometry()

        # --- State Variables (Using Parameterlist) ---
        if initial_positions is None:
            # Initialize randomly, away from borders
            pos_np = np.random.rand(num_marbles, 2)
            pos_np[:, 0] = pos_np[:, 0] * (width - 4 * radius) + 2 * radius
            pos_np[:, 1] = pos_np[:, 1] * (height / 4) + 2 * radius # Start near top
            initial_positions = torch.tensor(pos_np, device=device, dtype=dtype)
        else:
            # Ensure provided tensor has the right shape
            if initial_positions.shape != (num_marbles, 2):
                raise ValueError(f"initial_positions must have shape ({num_marbles}, 2), "
                                 f"but got {initial_positions.shape}")
            initial_positions = initial_positions.to(device=device, dtype=dtype)

        # Create a Parameterlist, one Parameter per ball
        self.ball_parameters = nn.ParameterList(
            [nn.Parameter(initial_positions[i].clone()) for i in range(num_marbles)]
        )

        self.frames: list[np.ndarray] = [] # Stores rendered frames

        self.to(device)

    def _get_stacked_positions(self) -> torch.Tensor:
        """Stacks the individual ball Parameters into a single tensor."""
        return torch.stack(list(self.ball_parameters), dim=0)

    def _default_obstacles(self) -> list[dict]:
        """Provides some default obstacles if none are specified."""
        # (Same as before)
        return [
            {'type': 'rect', 'rect': [100, 200, 300, 220]},
            {'type': 'circle', 'center': [200, 350], 'radius': 30},
            {'type': 'rect', 'rect': [0, 450, 150, 470]},
            {'type': 'rect', 'rect': [250, 450, 400, 470]},
        ]

    def _default_zones(self) -> list[dict]:
        """Provides some default zones if none are specified."""
        # (Same as before)
        return [
            {'type': 'accelerator', 'rect': [150, 450, 250, 500], 'accel': [0, 1.0]}
        ]

    def _preprocess_geometry(self):
        """Converts obstacle/zone geometry parameters to tensors."""
        for obs in self.obstacles:
            if obs['type'] == 'circle':
                obs['center_t'] = torch.tensor(obs['center'], device=self._device, dtype=self._dtype)
                obs['radius_t'] = torch.tensor(obs['radius'], device=self._device, dtype=self._dtype)
            elif obs['type'] == 'rect':
                obs['rect_t'] = torch.tensor(obs['rect'], device=self._device, dtype=self._dtype)
        for zone in self.zones:
            if zone['type'] == 'accelerator':
                zone['rect_t'] = torch.tensor(zone['rect'], device=self._device, dtype=self._dtype)
                # REMOVED .T - accel is 1D, tensor(...) handles it correctly.
                zone['accel_t'] = torch.tensor(zone['accel'], device=self._device, dtype=self._dtype)
                zone['strength'] = self.zone_strengths.get('accelerator', 0.0)
             # Add preprocessing for other zone types

    def reset(self, initial_positions: torch.Tensor | None = None):
        """Resets ball positions and clears frames."""
        with torch.no_grad():
            if initial_positions is None:
                # Initialize randomly, away from borders
                pos_np = np.random.rand(self.num_marbles, 2)
                pos_np[:, 0] = pos_np[:, 0] * (self.width - 4 * self.radius.item()) + 2 * self.radius.item()
                pos_np[:, 1] = pos_np[:, 1] * (self.height / 4) + 2 * self.radius.item() # Start near top
                new_pos_tensor = torch.tensor(pos_np, device=self._device, dtype=self._dtype)
            else:
                # Ensure provided tensor has the right shape
                if initial_positions.shape != (self.num_marbles, 2):
                    raise ValueError(f"initial_positions must have shape ({self.num_marbles}, 2), "
                                    f"but got {initial_positions.shape}")
                new_pos_tensor = initial_positions.to(device=self._device, dtype=self._dtype)

            # Update each Parameter in the Parameterlist
            for i, param in enumerate(self.ball_parameters):
                param.data.copy_(new_pos_tensor[i])

            # If using an optimizer like Adam, consider resetting its state too:
            # optimizer.state = {} # Or re-initialize the optimizer
        self.frames = []

    def _calculate_potential(self) -> torch.Tensor:
        """Calculates the total potential energy of the system."""
        # Get positions as a single tensor
        pos = self._get_stacked_positions() # Shape: (N, 2)

        total_potential = torch.tensor(0.0, device=self._device, dtype=self._dtype)

        # 1. Gravity Potential: U = m*g*y (summed over balls, m=1)
        # Using negative gravity_strength makes balls fall down (increase y)
        potential_gravity = self.gravity_strength * torch.sum(pos[:, 1])
        total_potential += potential_gravity

        # 2. Ball-Ball Repulsion Potential
        if self.num_marbles > 1 and self.repulsion_strength_bb > 0:
            diffs = pos.unsqueeze(1) - pos.unsqueeze(0)
            dists_sq = torch.sum(diffs**2, dim=-1)
            dists = torch.sqrt(dists_sq + 1e-9)
            penetration = torch.relu(2 * self.radius - dists)

            # --- Create mask to remove self-interaction (diagonal) ---
            # Use torch.ones_like to ensure same device and dtype as penetration
            mask = torch.ones_like(penetration)
            # Set diagonal elements to 0 without in-place modification affecting graph
            mask = mask.scatter_(0, torch.arange(self.num_marbles, device=self._device).unsqueeze(1), 0.0)
            # Alternatively, and perhaps cleaner for diagonal:
            # mask = 1.0 - torch.eye(self.num_marbles, device=self._device, dtype=self.dtype)

            # Apply mask
            penetration_masked = penetration * mask

            # Potential U ~ penetration^2 (gives linear force)
            # Sum over all pairs (automatically handles the factor of 0.5 due to mask)
            potential_bb = self.repulsion_strength_bb * 0.5 * torch.sum(penetration_masked**2)
            total_potential += potential_bb

        # 3. Boundary Repulsion Potential (Walls)
        if self.repulsion_strength_boundary > 0:
            x, y = pos[:, 0], pos[:, 1]
            pen_left = torch.relu(self.radius - x)
            pen_right = torch.relu(x - (self.width - self.radius))
            pen_top = torch.relu(self.radius - y) # Top wall is y=0
            pen_bottom = torch.relu(y - (self.height - self.radius))
            potential_boundary = self.repulsion_strength_boundary * torch.sum(
                pen_left**2 + pen_right**2 + pen_top**2 + pen_bottom**2
            )
            total_potential += potential_boundary

        # 4. Obstacle Repulsion Potential
        if self.repulsion_strength_obstacle > 0:
            potential_obstacles = torch.tensor(0.0, device=self._device, dtype=self._dtype)
            for obs in self.obstacles:
                obs_type = obs.get('type')
                if obs_type == 'circle':
                    obs_center = obs['center_t']
                    obs_radius = obs['radius_t']
                    dists = torch.norm(pos - obs_center.unsqueeze(0), dim=1)
                    penetration = torch.relu(self.radius + obs_radius - dists)
                    potential_obstacles += torch.sum(penetration**2)
                elif obs_type == 'rect':
                    rect = obs['rect_t']
                    ball_x, ball_y = pos[:, 0], pos[:, 1]
                    clamped_x = torch.clamp(ball_x, min=rect[0], max=rect[2])
                    clamped_y = torch.clamp(ball_y, min=rect[1], max=rect[3])
                    closest_point = torch.stack([clamped_x, clamped_y], dim=-1)
                    dist_sq = torch.sum((pos - closest_point)**2, dim=-1)
                    dist = torch.sqrt(dist_sq + 1e-9)
                    penetration = torch.relu(self.radius - dist)
                    potential_obstacles += torch.sum(penetration**2)
                # Add other obstacle types here
            total_potential += self.repulsion_strength_obstacle * potential_obstacles

        # 5. Special Zones Potential (Forces modeled as -grad(U))
        potential_zones = torch.tensor(0.0, device=self._device, dtype=self._dtype)
        for zone in self.zones:
            zone_type = zone.get('type')
            if zone_type == 'accelerator':
                rect = zone['rect_t']
                accel_vec = zone['accel_t'] # Shape (2,)
                strength = zone['strength']

                in_zone_x = (pos[:, 0] > rect[0]) & (pos[:, 0] < rect[2])
                in_zone_y = (pos[:, 1] > rect[1]) & (pos[:, 1] < rect[3])
                in_zone_mask = in_zone_x & in_zone_y

                if torch.any(in_zone_mask):
                    # Potential U = - F_dot_pos = - strength * (accel_vec . pos)
                    # Negative gradient: -grad(U) = + strength * accel_vec (the desired force)
                    # pos[in_zone_mask] is (num_in, 2), accel_vec is (2,)
                    # Result of matmul is (num_in,)
                    dot_products = pos[in_zone_mask] @ accel_vec
                    potential_accel = -strength * torch.sum(dot_products)
                    potential_zones += potential_accel
            # Add other zone types
        total_potential += potential_zones

        return total_potential

    def _render_current_state(self):
        """Renders the current state and appends to self.frames."""
        # Stack positions, detach from graph, move to CPU for numpy/cv2
        ball_pos_tensor = self._get_stacked_positions().nan_to_num()
        ball_pos_np = ball_pos_tensor.nan_to_num().detach().cpu().numpy()

        # Convert obstacles/zones back for drawing (same as before)
        obstacles_draw = []
        for obs in self.obstacles:
            obs_draw = {'type': obs['type']}
            if obs['type'] == 'circle':
                obs_draw['center'] = obs['center_t'].cpu().numpy().tolist()
                obs_draw['radius'] = obs['radius_t'].item()
            elif obs['type'] == 'rect':
                obs_draw['rect'] = obs['rect_t'].cpu().numpy().tolist()
            obstacles_draw.append(obs_draw)

        zones_draw = []
        for zone in self.zones:
            zone_draw = {'type': zone['type']}
            if 'rect' in zone:
                zone_draw['rect'] = zone['rect_t'].cpu().numpy().tolist()
            zones_draw.append(zone_draw)

        frame = draw_state(
            width=self.width,
            height=self.height,
            ball_positions=ball_pos_np,
            radius=self.radius.item(),
            obstacles=obstacles_draw,
            zones=zones_draw
        )

        return frame

    def get_loss(self) -> torch.Tensor:
        """
        Calculates the potential energy ('loss') for the current state
        and renders the state to a frame.
        """
        loss = self._calculate_potential()

        if self._make_images:
            frame = self._render_current_state()
            self.log('image', frame, log_test=False, to_uint8=False)

        return loss
