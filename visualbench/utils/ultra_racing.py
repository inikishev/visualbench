import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2

class MouseRacingTrack(nn.Module):
    def __init__(self, walls = None, init_pos=None, criterion = F.mse_loss, cell_size=40):
        super().__init__()
        if walls is None:
            # Setup environment (15x15 grid)
            walls = np.zeros((15, 15))
            walls[0, :] = 1; walls[-1, :] = 1; walls[:, 0] = 1; walls[:, -1] = 1
            # Add some obstacles
            walls[4:11, 7] = 1
            walls[7, 3:8] = 1

        if isinstance(walls, str):
            from .format import to_HW, normalize_to_uint8
            walls = normalize_to_uint8(to_HW(walls)) < 127

        self.walls = nn.Buffer(torch.as_tensor(walls).float())
        self.h, self.w = self.walls.shape
        self.cell_size = cell_size
        self.criterion = criterion

        # Store initial position for resetting
        if init_pos is None: init_pos = torch.tensor([self.h/2, self.w/2])
        self.init_pos = nn.Buffer(torch.as_tensor(init_pos).float())

        # Parameters
        self.position = nn.Parameter(self.init_pos.clone().float())
        self.value = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        # State
        self.mouse_pos = np.array([self.h/2, self.w/2])
        self.history = []
        self.bg_image = self._render_background()

    def reset_position(self):
        """Resets the car to its starting state"""
        with torch.no_grad():
            self.position.copy_(self.init_pos)
            self.value.zero_()
            self.history.clear()

    def _sample_walls(self, pos):
        norm_pos = torch.stack([
            (pos[1] / (self.w - 1)) * 2 - 1,
            (pos[0] / (self.h - 1)) * 2 - 1
        ]).view(1, 1, 1, 2)

        input_tensor = self.walls.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(input_tensor, norm_pos, align_corners=True, mode='bilinear')
        return sampled.view(-1)

    def get_loss(self):
        target = torch.tensor(self.mouse_pos, dtype=torch.float32, device=self.position.device)
        direction = (target - self.position).detach()
        norm = torch.norm(direction) + 1e-6
        unit_dir = direction / norm

        move_loss = self.criterion(self.position, target)

        if len(self.history) > 0:
            prev_pos = torch.tensor(self.history[-1], device=self.position.device)
            delta = self.position - prev_pos
            self.value.data -= torch.dot(delta, unit_dir)

        w = self._sample_walls(self.position)
        barrier_loss = 5.0 * torch.exp(7.0 * (w - 0.4)) if w > 0.1 else 0.0

        oob = (F.relu(-self.position[0]) + F.relu(self.position[0] - (self.h-1)) +
               F.relu(-self.position[1]) + F.relu(self.position[1] - (self.w-1)))

        total_loss = move_loss + self.value + barrier_loss + oob * 50
        return total_loss

    def _render_background(self):
        img = np.full((self.h * self.cell_size, self.w * self.cell_size, 3), 30, dtype=np.uint8)
        walls_np = self.walls.cpu().numpy()
        for y in range(self.h):
            for x in range(self.w):
                color = (20, 20, 25) if walls_np[y, x] > 0.5 else (60, 60, 60)
                cv2.rectangle(img, (x*self.cell_size, y*self.cell_size),
                              ((x+1)*self.cell_size, (y+1)*self.cell_size), color, -1)
                cv2.rectangle(img, (x*self.cell_size, y*self.cell_size),
                              ((x+1)*self.cell_size, (y+1)*self.cell_size), (40, 40, 40), 1)
        return img

    def render(self):
        img = self.bg_image.copy()
        if len(self.history) > 2:
            pts = (np.array(self.history[-50:])[:, ::-1] + 0.5) * self.cell_size
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], False, (0, 255, 255), 2, cv2.LINE_AA)

        m_pos = (int((self.mouse_pos[1] + 0.5) * self.cell_size), int((self.mouse_pos[0] + 0.5) * self.cell_size))
        cv2.drawMarker(img, m_pos, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        pos = self.position.detach().nan_to_num(0,1,-1).cpu().numpy()
        center = (int((pos[1] + 0.5) * self.cell_size), int((pos[0] + 0.5) * self.cell_size))
        cv2.circle(img, center, self.cell_size//3, (0, 0, 255), -1)
        cv2.circle(img, center, self.cell_size//3, (255, 255, 255), 2)
        return img

    def run_simulation(self, optimizer, history_size=100):
        win_name = "Torch Mouse Control (ESC to quit)"
        cv2.namedWindow(win_name)

        def mouse_callback(event, x, y, flags, param):
            # Left click to reset
            if event == cv2.EVENT_LBUTTONDOWN:
                self.reset_position()

            # Update mouse target position
            self.mouse_pos = np.array([y / self.cell_size - 0.5, x / self.cell_size - 0.5])

        cv2.setMouseCallback(win_name, mouse_callback)

        print("Control the red dot with your mouse.")
        print("LEFT CLICK to reset position.")

        while True:
            stop = False

            def closure(backward=True):
                loss = self.get_loss()
                if backward:
                    optimizer.zero_grad()
                    loss.backward()

                self.history.append(self.position.detach().cpu().numpy().copy())
                if len(self.history) > history_size: self.history.pop(0)

                frame = self.render()
                cv2.imshow(win_name, frame)

                if cv2.waitKey(16) & 0xFF == 27:
                    nonlocal stop
                    stop = True
                return loss

            optimizer.step(closure)
            if stop: break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    track = MouseRacingTrack()
    opt = torch.optim.Adam(track.parameters(), 3e-1)
    track.run_simulation(opt)