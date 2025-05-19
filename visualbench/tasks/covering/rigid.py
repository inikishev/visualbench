# pylint:disable=no-member
import cv2
import numpy as np
import torch
from torch import nn
from ...benchmark import Benchmark
from ..packing.rigid_box import CONTAINER1

class RigidBoxCovering(Benchmark):
    def __init__(
        self,
        area_size=CONTAINER1[0],
        rectangle_dims=CONTAINER1[1],
        px_per_unit=20.0,
        T=0.1,
        w_coverage=1.0,
        w_overlap=0.5,
        w_bounds=1.0,
    ):
        super().__init__()
        self.area_H_units, self.area_W_units = area_size
        self.num_rects = len(rectangle_dims)
        self.T = T # In area_size units
        self.px_per_unit = px_per_unit

        max_widths_init = torch.tensor([d[1] for d in rectangle_dims])
        max_heights_init = torch.tensor([d[0] for d in rectangle_dims])

        # Attempt to initialize within a central 60% region, respecting bounds
        spawn_range_x_start = self.area_W_units * 0.2
        spawn_range_x_width = self.area_W_units * 0.6
        spawn_range_y_start = self.area_H_units * 0.2
        spawn_range_y_width = self.area_H_units * 0.6

        initial_positions_x = spawn_range_x_start + torch.rand(self.num_rects) * spawn_range_x_width
        initial_positions_y = spawn_range_y_start + torch.rand(self.num_rects) * spawn_range_y_width

        # Clamp initial positions to ensure rectangles start within bounds
        # (tx >= 0, ty >= 0, tx + w <= W_area, ty + h <= H_area)
        zero = torch.tensor(0.) # styupid clamp
        initial_positions_x = torch.clamp(initial_positions_x, zero, torch.relu(self.area_W_units - max_widths_init))
        initial_positions_y = torch.clamp(initial_positions_y, zero, torch.relu(self.area_H_units - max_heights_init))

        self.positions = nn.Parameter(
            torch.stack([initial_positions_x, initial_positions_y], dim=1) # Shape: (num_rects, 2) -> [x, y]
        )

        self.rect_dims_tensor = nn.Buffer(torch.tensor(rectangle_dims)) # (num_rects, 2) [h, w]

        # --- Rendering grid for soft rasterization ---
        # Grid resolution in pixels
        self.render_H_pixels = max(1, int(self.area_H_units * self.px_per_unit))
        self.render_W_pixels = max(1, int(self.area_W_units * self.px_per_unit))

        # Pixel coordinates in 'area_size' units.
        # Each element in y_coords_units/x_coords_units is the "unit" coordinate of a pixel center/edge.
        y_coords_units = torch.linspace(0, self.area_H_units, self.render_H_pixels, device=self.device)
        x_coords_units = torch.linspace(0, self.area_W_units, self.render_W_pixels, device=self.device)

        # self.grid_Y_units: (render_H_pixels, render_W_pixels), self.grid_X_units: (render_H_pixels, render_W_pixels)
        grid_Y_units, grid_X_units = torch.meshgrid(y_coords_units, x_coords_units, indexing='ij')
        self.grid_Y_units = nn.Buffer(grid_Y_units)
        self.grid_X_units = nn.Buffer(grid_X_units)

        # --- Loss weights ---
        self.w_coverage = w_coverage
        self.w_overlap = w_overlap
        self.w_bounds = w_bounds

        # --- Distinct colors for visualization ---
        self.colors = []
        for i in range(self.num_rects):
            hue = int((i / max(1, self.num_rects)) * 180) # Hue ranges 0-179 in OpenCV
            hsv_color = np.uint8([[[hue, 255, 220]]]) # pyright: ignore[reportArgumentType]
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0] # pyright: ignore[reportCallIssue,reportArgumentType]
            self.colors.append((int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])))


    def _soft_rasterize_rects(self):
        """Vectorized soft rasterization of all rectangles."""
        txs = self.positions[:, 0]  # (num_rects,) top-left x in units
        tys = self.positions[:, 1]  # (num_rects,) top-left y in units
        hs = self.rect_dims_tensor[:, 0]  # (num_rects,) height in units
        ws = self.rect_dims_tensor[:, 1]  # (num_rects,) width in units

        grid_X_exp = self.grid_X_units.unsqueeze(0)  # (1, H_pix, W_pix)
        grid_Y_exp = self.grid_Y_units.unsqueeze(0)  # (1, H_pix, W_pix)

        txs_exp = txs.view(-1, 1, 1)  # (num_rects, 1, 1)
        tys_exp = tys.view(-1, 1, 1)
        ws_exp = ws.view(-1, 1, 1)
        hs_exp = hs.view(-1, 1, 1)

        # Temperature for sigmoid, ensuring it's positive
        T = self.T + 1e-7

        # Horizontal presence: sigmoid((px - x_min)/T) * sigmoid((x_max - px)/T)
        # px = grid_X_exp, x_min = txs_exp, x_max = txs_exp + ws_exp
        sig_val1_x = (grid_X_exp - txs_exp) / T
        sig_val2_x = (txs_exp + ws_exp - grid_X_exp) / T
        presence_x = torch.sigmoid(sig_val1_x) * torch.sigmoid(sig_val2_x)

        # Vertical presence: sigmoid((py - y_min)/T) * sigmoid((y_max - py)/T)
        # py = grid_Y_exp, y_min = tys_exp, y_max = tys_exp + hs_exp
        sig_val1_y = (grid_Y_exp - tys_exp) / T
        sig_val2_y = (tys_exp + hs_exp - grid_Y_exp) / T
        presence_y = torch.sigmoid(sig_val1_y) * torch.sigmoid(sig_val2_y)

        all_rect_masks = presence_x * presence_y  # (num_rects, H_pix, W_pix)
        coverage_map = torch.sum(all_rect_masks, dim=0)  # (H_pix, W_pix)

        return coverage_map, all_rect_masks

    def get_loss(self):
        coverage_map, all_rect_masks = self._soft_rasterize_rects() # all_rect_masks not used by loss directly here
        loss_coverage = torch.mean((1.0 - torch.clamp(coverage_map, max=1.0))**2)
        loss_overlap = torch.mean(torch.relu(coverage_map - 1.0)**2)

        # bounds
        txs = self.positions[:, 0]
        tys = self.positions[:, 1]
        hs = self.rect_dims_tensor[:, 0]
        ws = self.rect_dims_tensor[:, 1]

        # Penalize tx < 0, ty < 0 (rect starting outside left/top)
        loss_b_neg_x = torch.relu(-txs) # pylint:disable = invalid-unary-operand-type
        loss_b_neg_y = torch.relu(-tys) # pylint:disable = invalid-unary-operand-type

        # Penalize tx + w > W_area, ty + h > H_area (rect ending outside right/bottom)
        loss_b_pos_x = torch.relu(txs + ws - self.area_W_units)
        loss_b_pos_y = torch.relu(tys + hs - self.area_H_units)

        # Sum of squared penalties, averaged.
        loss_bounds = (torch.mean(loss_b_neg_x**2) + torch.mean(loss_b_neg_y**2) +
                       torch.mean(loss_b_pos_x**2) + torch.mean(loss_b_pos_y**2)) / 4.0

        total_loss = (self.w_coverage * loss_coverage +
                      self.w_overlap * loss_overlap +
                      self.w_bounds * loss_bounds)

        if self._make_images:
            with torch.no_grad():
                # --- Generate Visualization Frame (using cv2) ---
                image = np.full((self.render_H_pixels, self.render_W_pixels, 3), (240, 240, 240), dtype=np.uint8)
                cv2.rectangle(image, (0, 0), (self.render_W_pixels - 1, self.render_H_pixels - 1), (50, 50, 50), 1)

                current_positions_np = self.positions.detach().cpu().numpy() # pylint:disable=not-callable
                current_dims_np = self.rect_dims_tensor.cpu().numpy() # [h, w]

                scale_x = self.render_W_pixels / self.area_W_units
                scale_y = self.render_H_pixels / self.area_H_units

                for i in range(self.num_rects):
                    rect_x_unit, rect_y_unit = current_positions_np[i]
                    rect_h_unit, rect_w_unit = current_dims_np[i]

                    draw_x1 = int(round(rect_x_unit * scale_x))
                    draw_y1 = int(round(rect_y_unit * scale_y))
                    draw_x2 = int(round((rect_x_unit + rect_w_unit) * scale_x))
                    draw_y2 = int(round((rect_y_unit + rect_h_unit) * scale_y))

                    color = self.colors[i]
                    overlay = image.copy()
                    cv2.rectangle(overlay, (draw_x1, draw_y1), (draw_x2, draw_y2), color, thickness=-1)
                    alpha = 0.6 # Transparency
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                    cv2.rectangle(image, (draw_x1, draw_y1), (draw_x2, draw_y2), (30,30,30), 1)


                self.log_image('hard', image, to_uint8=False, show_best=True)
                while all_rect_masks.shape[0] % 3 != 0: all_rect_masks = torch.cat([all_rect_masks, all_rect_masks[None, 0]])
                self.log_image('soft', all_rect_masks.view(-1,3,*all_rect_masks.shape[1:]).mean(0), to_uint8=True)

        return total_loss
