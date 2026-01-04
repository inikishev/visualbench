import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
import visualbench as vb

class RacingTrack(vb.Benchmark):
    def __init__(self, grid, walls, init, cell_size=16):
        super().__init__()
        self.grid = nn.Buffer(vb.totensor(grid).float())
        self.walls = nn.Buffer(vb.totensor(walls).bool().float())
        self.h, self.w, _ = self.grid.shape
        self.cell_size = cell_size

        # 3. Initialization
        # Start the "car" on the left side of the track
        self.position = nn.Parameter(torch.tensor(init).float())
        self.value = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.history = []

        # Pre-render environment
        self.bg_image = self._render_background()

    def _sample(self, img, pos):
        """Differentiable bilinear sampling at pos=(y, x)"""
        H, W = img.shape[:2]
        # Grid sample expects x, y in [-1, 1]
        norm_pos = torch.stack([
            (pos[1] / (W - 1)) * 2 - 1,
            (pos[0] / (H - 1)) * 2 - 1
        ]).view(1, 1, 1, 2)

        input_tensor = img.permute(2, 0, 1).unsqueeze(0) if img.ndim == 3 else img.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(input_tensor, norm_pos, align_corners=True, mode='bilinear')
        return sampled.view(-1)

    def get_loss(self):
        # Sample environment
        g = self._sample(self.grid, self.position)
        w = self._sample(self.walls, self.position)

        # 1. Flow Loss: Move in direction of g
        # This creates a "target" slightly ahead in the current
        with torch.no_grad():
            target = (self.position + g).detach()
        move_loss = F.mse_loss(self.position, target)

        # 2. Path Integral: To keep it looping, we reduce the "score"
        # as we move along the vector field.
        if len(self.history) > 0:
            prev_pos = torch.tensor(self.history[-1]).to(self.position.device)
            delta = self.position - prev_pos
            # Moving along g reduces the global value
            self.value.data -= torch.dot(delta, g)

        # 3. Barrier Function: Exponential penalty for walls
        # Using a smooth barrier helps L-BFGS/Adam steer away before hitting
        barrier_loss = 2.0 * torch.exp(5.0 * (w - 0.5)) if w > 0.1 else 0.0

        # 4. Out of bounds safety
        oob = (F.relu(-self.position[0]) + F.relu(self.position[0] - (self.h-1)) +
               F.relu(-self.position[1]) + F.relu(self.position[1] - (self.w-1)))

        total_loss = move_loss + self.value + barrier_loss + oob*10

        # Log metrics
        self.log("path_value", self.value)
        self.log("wall_penalty", barrier_loss)
        self.history.append(self.position.detach().cpu().numpy().copy())

        if self._make_images:
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
        pos = self.position.detach().cpu().numpy()
        # Add 0.5 to align with the center of the grid cells
        center = (int((pos[1] + 0.5) * self.cell_size), int((pos[0] + 0.5) * self.cell_size))
        cv2.circle(img, center, self.cell_size//4, (0, 0, 255), -1) # Car
        cv2.circle(img, center, self.cell_size//4, (255, 255, 255), 1)
        return img


# track = torch.zeros(5, 5, 2)
# track[1, 1:4] = torch.tensor([0., 1.])
# track[3, 1:4] = torch.tensor([0., -1.])
# track[1:4, 1] = torch.tensor([-1., 0.])
# track[1:4, 3] = torch.tensor([1., 0.])

# track[1, 1] = torch.tensor([-1., 1.])
# track[1, 3] = torch.tensor([1., 1.])
# track[3, 1] = torch.tensor([-1., -1.])
# track[3, 3] = torch.tensor([1., -1.])

# walls = torch.zeros(5,5).bool()
# walls[0]=True
# walls[-1]=True
# walls[:,0]=True
# walls[:,-1]=True
# walls[2,2]=True

# bench = RacingTrack(-track, walls, (1,1))
# opt = torch.optim.SGD(bench.parameters(), 1e-1)
# bench.run(opt,1000).plot()