from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from ..benchmark import Benchmark
from ..utils import CUDA_IF_AVAILABLE

WALLS1 = (
            # Perimeter walls
            (0.0, 5.0, 0.0, 0.2),    # Bottom
            (0.0, 5.0, 4.8, 5.0),    # Top
            (0.0, 0.2, 0.0, 5.0),    # Left
            (4.8, 5.0, 0.0, 5.0),    # Right

            # Internal walls
            (1.0, 4.0, 1.0, 1.2),
            (2.0, 2.2, 1.0, 4.0),
            (3.5, 4.5, 3.0, 3.2),
            (0.5, 1.5, 3.0, 3.2)
)

class OptimalControl(Benchmark):
    """Optimize controls of an agent solving a maze.

    Renders:
        trajectory of the agent.

    Args:
        walls(Sequence, optional):
            a sequence of tuples of 4 values: (x_start, x_end, y_start, y_end).
            Make sure to include perimeter walls if you don't want the agent to go outside.
        dt (float, optional):
            time step, lower is more accurate. Anything higher than 0.1 seems unstable. Defaults to 0.1.
        T (int, optional):
            number of time steps to run simulation for. 100 is enough to solve the maze with dt=0.1. Defaults to 100.
        collision_weight (float, optional): collision loss weight. Defaults to 10..
        control_weight (float, optional): control loss weight. Defaults to 0.1.
        k (float, optional): collision sensitivity. Defaults to 20.0.
    """
    initial_state: torch.Tensor
    target: torch.Tensor
    walls_tensor: torch.Tensor

    def __init__(
        self,
        walls=WALLS1,
        dt=0.1,
        T=100,
        mode: Literal['force', 'velocity', 'position'] = "force",
        collision_weight=10.0,
        control_weight=4.0,
        k=20.0,
        initial_pos=(0.5, 0.5),
        initial_vel=(0.0, 0.0),
        target_pos=(4.5, 4.5),
    ):
        super().__init__()
        self.walls = walls
        self.dt = dt
        self.T = T
        self.collision_weight = collision_weight
        self.control_weight = control_weight
        self.k = k
        self.img_size = 500
        self.scale = self.img_size / 5.0
        self.mode = mode.lower()

        if self.mode not in ["force", "velocity", "position"]:
            raise ValueError("optimize_mode must be 'force', 'velocity', or 'position'")

        # Buffers
        initial_state = torch.tensor([*initial_pos, *initial_vel], dtype=torch.float32)
        self.initial_state = nn.Buffer(initial_state)
        self.target = nn.Buffer(torch.tensor(target_pos, dtype=torch.float32))
        self.walls_tensor = nn.Buffer(torch.tensor(self.walls, dtype=torch.float32))

        if self.mode == "force":
            self.controls = nn.Parameter(torch.zeros(self.T, 2))

        elif self.mode == "velocity":
            self.controls = nn.Parameter(torch.zeros(self.T, 2))

        elif self.mode == "position":
            initial_p = self.initial_state[:2].unsqueeze(0).repeat(self.T, 1)
            self.controls = nn.Parameter(initial_p + torch.randn(self.T, 2) * 0.01) # Add small noise

        self._create_background()

    def _create_background(self):
        """pre-renders static maze elements."""
        self.background = Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255))
        draw = ImageDraw.Draw(self.background)

        walls_cpu = self.walls
        for wall in walls_cpu:
            x1, y1 = wall[0] * self.scale, self.img_size - wall[3] * self.scale
            x2, y2 = wall[1] * self.scale, self.img_size - wall[2] * self.scale
            draw.rectangle([x1, y1, x2, y2], fill='#808080', outline='black')

        self._draw_marker(draw, self.initial_state[:2].cpu().numpy(), '#00ff00')
        self._draw_marker(draw, self.target.cpu().numpy(), '#ff0000')

    def _draw_marker(self, draw, pos, color, size=10):
        x, y = pos[0] * self.scale, self.img_size - pos[1] * self.scale
        draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)

    def get_loss(self):
        control_loss = 0

        P0 = self.initial_state[:2]  # Initial position [px0, py0]
        V0 = self.initial_state[2:]  # Initial velocity [vx0, vy0]

        # --- Simulate Trajectory based on mode ---
        if self.mode == "force":
            U = self.controls * self.dt
            V_increments = torch.cumsum(U, dim=0)
            V = torch.cat([V0.unsqueeze(0), V0.unsqueeze(0) + V_increments], dim=0)

            V_dt = V[:-1] * self.dt
            P_increments = torch.cumsum(V_dt, dim=0)
            P = torch.cat([P0.unsqueeze(0), P0.unsqueeze(0) + P_increments], dim=0)

            control_loss = self.control_weight * torch.mean(self.controls**2)

        elif self.mode == "velocity":
            V = torch.cat([V0.unsqueeze(0), self.controls], dim=0) # Shape [T+1, 2]

            V_dt = V[:-1] * self.dt
            P_increments = torch.cumsum(V_dt, dim=0)
            P = torch.cat([P0.unsqueeze(0), P0.unsqueeze(0) + P_increments], dim=0)

            control_loss = self.control_weight * torch.mean(self.controls**2)

        elif self.mode == "position":
            P = torch.cat([P0.unsqueeze(0), self.controls], dim=0) # Shape [T+1, 2]

            V_derived = (P[1:] - P[:-1]) / self.dt # Shape [T, 2]
            V = torch.cat([V_derived, V_derived[-1:]], dim=0) # Shape [T+1, 2]

            control_loss = self.control_weight * torch.mean(V_derived**2)

        else:
            raise ValueError(f"Invalid optimize_mode: {self.mode}")


        collision_loss = self._collision_loss(P[1:])
        target_loss = torch.sum((P[-1] - self.target)**2)

        loss = target_loss + control_loss + collision_loss

        if self._make_images:
            with torch.no_grad():
                if not hasattr(self, 'background'):
                    self._create_background()

                img = self.background.copy()
                draw = ImageDraw.Draw(img)

                trajectory_pos = P.detach().cpu().numpy()
                points = [(x * self.scale, self.img_size - y * self.scale) for x, y in trajectory_pos]
                if len(points) > 1:
                    for i in range(len(points)-1):
                        draw.line([points[i], points[i+1]], fill='#0000ff', width=3)

                self.log_image('trajectory', np.asarray(img), to_uint8=False, show_best=True)

        return loss

    def _collision_loss(self, trajectory):
        """collision penalty"""
        pos = trajectory[1:] # T, 2

        walls = self.walls_tensor
        x_min, x_max = walls[:, 0], walls[:, 1]
        y_min, y_max = walls[:, 2], walls[:, 3]

        x = pos[:, 0].unsqueeze(1)
        y = pos[:, 1].unsqueeze(1)

        inside_x = torch.sigmoid((x - x_min) * self.k) * torch.sigmoid((x_max - x) * self.k)
        inside_y = torch.sigmoid((y - y_min) * self.k) * torch.sigmoid((y_max - y) * self.k)
        return self.collision_weight * (inside_x * inside_y).sum()
