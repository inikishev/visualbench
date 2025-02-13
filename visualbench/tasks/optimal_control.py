import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw
from ..benchmark import Benchmark
from ..utils import CUDA_IF_AVAILABLE

SIMPLE_MAZE = (
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
    """Optimize controls of an agent solving a maze. Quasi-newton optimizers like LBFGS are significantly faster on this.

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
    initial_state: torch.nn.Buffer
    target: torch.nn.Buffer
    walls_tensor: torch.nn.Buffer
    def __init__(self, walls=None, dt=0.1, T=100, collision_weight=10.0, control_weight=0.1, k=20.):
        super().__init__()
        self.walls = walls if walls is not None else SIMPLE_MAZE  # Ensure walls are defined
        self.dt = dt
        self.T = T
        self.collision_weight = collision_weight
        self.control_weight = control_weight
        self.k = k
        self.img_size = 500
        self.scale = self.img_size / 5.0

        self.register_buffer('initial_state', torch.tensor([0.5, 0.5, 0.0, 0.0]))
        self.register_buffer('target', torch.tensor([4.5, 4.5]))
        self.register_buffer('walls_tensor', torch.tensor(self.walls, dtype=torch.float32))
        self.controls = nn.Parameter(torch.zeros(self.T, 2))
        self._create_background()
        self.frames = []

    def _create_background(self):
        """Pre-render static maze elements."""
        self.background = Image.new('RGB', (self.img_size, self.img_size), (255, 255, 255))
        draw = ImageDraw.Draw(self.background)
        for wall in self.walls:
            x1, y1 = wall[0] * self.scale, self.img_size - wall[3] * self.scale
            x2, y2 = wall[1] * self.scale, self.img_size - wall[2] * self.scale
            draw.rectangle([x1, y1, x2, y2], fill='#808080', outline='black')
        self._draw_marker(draw, self.initial_state[:2].numpy(), '#00ff00')
        self._draw_marker(draw, self.target.numpy(), '#ff0000')

    def _draw_marker(self, draw, pos, color, size=10):
        """Draw position markers."""
        x, y = pos[0] * self.scale, self.img_size - pos[1] * self.scale
        draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=color)

    def get_loss(self):
        """Calculate total loss and visualize trajectory."""
        # Simulate trajectory with precomputed controls*dt
        states = torch.zeros((self.T + 1, 4), device=self.initial_state.device)
        states[0] = self.initial_state
        controls_dt = self.controls * self.dt  # Precompute control adjustments

        for t in range(self.T):
            vel = states[t, 2:] + controls_dt[t]
            pos = states[t, :2] + states[t, 2:] * self.dt
            states[t+1] = torch.cat([pos, vel])

        # Calculate losses
        collision_loss = self._collision_loss(states[:, :2])
        control_loss = self.control_weight * torch.sum(self.controls**2)
        target_loss = torch.sum((states[-1, :2] - self.target)**2)
        total_loss = target_loss + control_loss + collision_loss

        # Render trajectory
        with torch.no_grad():
            img = self.background.copy()
            draw = ImageDraw.Draw(img)
            trajectory = states[:, :2].detach().cpu().numpy()
            points = [(x * self.scale, self.img_size - y * self.scale) for x, y in trajectory]
            for i in range(len(points)-1):
                draw.line([points[i], points[i+1]], fill='#0000ff', width=3)

        return total_loss, {"image path": np.array(img).astype(np.uint8)}

    def _collision_loss(self, trajectory):
        """collision penalty"""
        pos = trajectory[1:]  # Exclude initial position
        walls = self.walls_tensor
        x_min, x_max = walls[:, 0], walls[:, 1]
        y_min, y_max = walls[:, 2], walls[:, 3]

        # Expand trajectory for vectorized operations
        x = pos[:, 0].unsqueeze(1)
        y = pos[:, 1].unsqueeze(1)

        # Compute collision probabilities
        inside_x = torch.sigmoid((x - x_min) * self.k) * torch.sigmoid((x_max - x) * self.k)
        inside_y = torch.sigmoid((y - y_min) * self.k) * torch.sigmoid((y_max - y) * self.k)
        return self.collision_weight * (inside_x * inside_y).sum()