# pylint:disable=no-member
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
from ....benchmark import Benchmark
from ....utils import totensor

DEFAULT_GRID = ([[1,1],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,-1]],[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[0,-1]],[[0,1],[-1,0],[0,1],[0,1],[0,1],[-1,0],[0,0],[0,1],[0,1],[0,1],[0,1],[0,1],[-1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,-1],[0,-1],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,0],[1,0],[0,0],[-1,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,-1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[-1,0],[0,0],[1,0],[0,0],[-1,0],[0,-1],[0,-1],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[0,0],[0,1],[1,0],[0,0],[0,0],[0,0],[0,0],[-1,0],[0,0],[0,1],[1,0],[0,0],[0,0],[0,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[0,1],[0,1],[0,1],[1,0],[0,0],[0,0],[0,0],[-1,0],[0,0],[0,0],[0,1],[0,1],[0,1],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[-1,0],[-1,0],[0,0],[0,1],[1,0],[0,0],[0,0],[-1,0],[0,-1],[0,0],[0,0],[0,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[-1,0],[-1,0],[0,0],[0,0],[0,1],[1,0],[0,0],[0,0],[-1,0],[0,-1],[0,-1],[0,-1],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[-1,0],[-1,0],[0,0],[0,0],[0,0],[0,1],[1,0],[0,0],[0,0],[0,0],[0,0],[-1,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[-1,0],[0,-1],[0,-1],[0,0],[0,0],[0,0],[0,1],[0,1],[0,1],[0,1],[0,1],[-1,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,0],[1,0],[0,0],[0,0],[-1,0],[-1,0],[0,-1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[1,0],[0,-1]],[[0,1],[-1,0],[0,-1],[0,-1],[0,0],[0,0],[0,0],[-1,0],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1],[0,-1]],[[-1,1],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,-1]])

DEFAULT_WALLS = (
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
)

DEFAULT_INIT = (1.0, 6.0)


class RacingTrack(Benchmark):
    def __init__(self, grid=DEFAULT_GRID, walls=DEFAULT_WALLS, init=DEFAULT_INIT, cell_size=16):
        super().__init__()
        self.grid = nn.Buffer(totensor(grid).float())
        self.walls = nn.Buffer(totensor(walls).bool().float())
        self.h, self.w, force_dim = self.grid.shape
        assert force_dim == 2
        self.cell_size = cell_size

        # 3. Initialization
        # Start the "car" on the left side of the track
        self.position = nn.Parameter(torch.tensor(init).float())
        self.integral_loss = nn.Buffer(torch.tensor(0.0))
        self.history = []
        self.step_history = []

        # Pre-render environment
        self.bg_image = self._render_background()

    def reset(self):
        self.history = []
        self.step_history = []
        return super().reset()

    def _sample(self, img, pos):
        """Differentiable bilinear sampling at pos=(y, x)"""
        H, W = img.shape[:2]
        # Grid sample expects x, y in [-1, 1]
        norm_pos = torch.stack([
            (pos[1] / (W - 1)) * 2 - 1,
            (pos[0] / (H - 1)) * 2 - 1
        ]).view(1, 1, 1, 2)

        input_tensor = img.permute(2, 0, 1).unsqueeze(0) if img.ndim == 3 else img.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(input_tensor, norm_pos, align_corners=True, mode='bilinear', padding_mode='border')
        return sampled.view(-1)

    def pre_step(self):
        self.g_ = self._sample(self.grid, self.position)

        with torch.no_grad():

            self.target_ = (self.position + self.g_).detach()

            if len(self.step_history) > 0:
                self.prev_pos_ = torch.tensor(self.step_history[-1]).to(self.position.device)
            else:
                self.prev_pos_ = None

            self.init_integral_loss_ = self.integral_loss.clone()
            self.step_history.append(self.position.detach().cpu().numpy().copy()) # pylint:disable=not-callable

    def get_loss(self):
        if self.g_ is not None: # avoid redundant computation of 1st g
            g = self.g_
            self.g_ = None
        else:
            g = self._sample(self.grid, self.position)

        move_loss = F.mse_loss(self.position, self.target_)

        if self.prev_pos_ is not None:
            with torch.no_grad():
                delta = self.position - self.prev_pos_
                self.integral_loss.set_(self.init_integral_loss_ - torch.dot(delta, g)) # type:ignore

        w = self._sample(self.walls, self.position)
        barrier_loss = 0.4 * torch.exp(5.0 * (w - 0.5)) if w > 0.1 else 0.0

        oob = (F.relu(self.position[0].neg()) + F.relu(self.position[0] - (self.h-1)) +
               F.relu(self.position[1].neg()) + F.relu(self.position[1] - (self.w-1)))

        total_loss = move_loss + self.integral_loss + barrier_loss + oob*10

        # Log metrics
        self.log("path_value", self.integral_loss)
        self.log("wall_penalty", barrier_loss)
        self.history.append(self.position.detach().cpu().numpy().copy()) # pylint:disable=not-callable

        if self.make_images:
            self.log_image(name='race', image=self._render_frame(), to_uint8=False)

        return total_loss

    def _render_background(self):
        H, W = self.h, self.w
        img = np.full((H * self.cell_size, W * self.cell_size, 3), 30, dtype=np.uint8)

        # Draw track surface (Dark Gray) and Walls (Black)
        walls_np = self.walls.cpu().numpy()
        for y in range(H):
            for x in range(W):
                if walls_np[y, x] < 0.5: # Inside track
                    cv2.rectangle(img, (x*self.cell_size, y*self.cell_size),
                                  ((x+1)*self.cell_size, (y+1)*self.cell_size), (80, 80, 80), -1)
                else: # Wall
                    cv2.rectangle(img, (x*self.cell_size, y*self.cell_size),
                                  ((x+1)*self.cell_size, (y+1)*self.cell_size), (20, 20, 25), -1)

        # Draw Vectors (Every 2nd cell for clarity)
        grid_np = self.grid.cpu().numpy()
        for y in range(0, H):
            for x in range(0, W):
                if walls_np[y, x] < 0.5:
                    p1 = (int((x + 0.5) * self.cell_size), int((y + 0.5) * self.cell_size))
                    dy, dx = grid_np[y, x]
                    p2 = (int(p1[0] + dx * self.cell_size * 0.5), int(p1[1] + dy * self.cell_size * 0.5))
                    cv2.arrowedLine(img, p1, p2, (120, 120, 150), 1, tipLength=0.2)
        return img

    def _render_frame(self):
        img = self.bg_image.copy()
        if len(self.history) < 2: return img

        # Draw Trajectory (fading line)
        # Add 0.5 to align with the center of the grid cells
        pts = (np.array(self.history[-100:])[:, ::-1] + 0.5) * self.cell_size
        pts = pts.astype(np.int32)
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], False, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Car
        pos = self.position.detach().cpu().numpy() # pylint:disable=not-callable
        # Add 0.5 to align with the center of the grid cells
        center = (int((pos[1] + 0.5) * self.cell_size), int((pos[0] + 0.5) * self.cell_size))
        cv2.circle(img, center, self.cell_size//4, (0, 0, 255), -1) # Car
        cv2.circle(img, center, self.cell_size//4, (255, 255, 255), 1)
        return img

