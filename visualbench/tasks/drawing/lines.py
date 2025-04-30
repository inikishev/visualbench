import math

import torch
from torch import nn
import torch.nn.functional as F
from myai.transforms import normalize

from ..._utils import _make_float_hw3_tensor, _normalize_to_uint8
from ...benchmark import Benchmark


def _point_segment_distance_sq(p, a, b, epsilon=1e-6):
    v = b - a
    w = p - a
    dot_wv = torch.sum(w * v, dim=-1)
    dot_vv = torch.sum(v * v, dim=-1)
    t = torch.clamp(dot_wv / (dot_vv + epsilon), 0.0, 1.0)
    closest_point_on_seg = a + t.unsqueeze(-1) * v
    dist_sq = torch.sum((p - closest_point_on_seg)**2, dim=-1)
    return dist_sq


class LinesDrawer(Benchmark):
    """
    Args:
        num_lines (int): The number of lines to use.
        height (int): The height of the target image.
        width (int): The width of the target image.
        initial_line_thickness (float): Initial value for line thickness (sigma).
                                    Sigma is now learnable and per-line.
        sigma_penalty_weight (float): Coefficient for the sigma penalty term in the loss.
                                    Positive value penalizes *large* sigmas (promotes sharpness).
        init_range (float): Range (+/-) for random initialization of raw coordinates.
                            Colors are initialized with randn.
    """

    def __init__(
        self,
        target_image,
        num_lines: int,
        loss=F.mse_loss,
        per_line_thickness: bool = False,
        initial_line_thickness: float = 1.5,
        thinkness_penalty_weight: float = 0.01,
        min_log_sigma = 0,
        init_range: float = 0.1,
    ):
        super().__init__(log_projections=True)

        target_image = normalize(_make_float_hw3_tensor(target_image)).moveaxis(-1, 0)
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', target_image, to_uint8=True)

        self.num_lines = num_lines
        _, self.height, self.width = target_image.shape
        self.thickness_penalty_weight = thinkness_penalty_weight
        self.per_line_thickness = per_line_thickness
        self.epsilon = 1e-6
        self.loss = loss
        self.min_log_sigma = min_log_sigma

        # raw coords
        self.raw_start_points = nn.Parameter(torch.rand(num_lines, 2) * 2 * init_range - init_range)
        self.raw_end_points = nn.Parameter(torch.rand(num_lines, 2) * 2 * init_range - init_range)
        # raw colors
        self.raw_colors = nn.Parameter(torch.randn(num_lines, 3) * 0.5)
        # sigmas
        if self.per_line_thickness:
            self.log_sigmas = nn.Parameter(torch.full((num_lines,), math.log(initial_line_thickness)))
        else:
            self.log_sigmas = nn.Parameter(torch.tensor(math.log(initial_line_thickness), dtype=torch.float32))

        # precompute grid
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0.5, self.width - 0.5, self.width),
            torch.linspace(0.5, self.height - 0.5, self.height),
            indexing='xy'
        )
        pixel_grid = torch.stack((grid_x, grid_y), dim=-1).permute(1, 0, 2) # [H, W, 2]
        self.pixel_grid = nn.Buffer(pixel_grid)

        # plotting
        self.set_display_best('image reconstructed')

    def _get_sigmoid_params(self):
        """maps params with sigmoid"""
        # positions
        start_points_x = torch.sigmoid(self.raw_start_points[:, 0:1]) * self.width
        start_points_y = torch.sigmoid(self.raw_start_points[:, 1:2]) * self.height
        start_points = torch.cat([start_points_x, start_points_y], dim=1) # N, 2

        end_points_x = torch.sigmoid(self.raw_end_points[:, 0:1]) * self.width
        end_points_y = torch.sigmoid(self.raw_end_points[:, 1:2]) * self.height
        end_points = torch.cat([end_points_x, end_points_y], dim=1) # N, 2

        # colors and sigmas
        colors = torch.sigmoid(self.raw_colors) # N, 3
        sigmas = torch.exp(self.log_sigmas) # N

        return start_points, end_points, colors, sigmas

    def _render(self):
        """render (differentiably)"""
        start_points, end_points, colors, sigmas = self._get_sigmoid_params()

        P = self.pixel_grid.unsqueeze(2) # pixels H, W, 1, 2
        A = start_points.view(1, 1, self.num_lines, 2) # starts 1, 1, N, 2
        B = end_points.view(1, 1, self.num_lines, 2) # enmds 1, 1, N, 2
        Col = colors.view(1, 1, self.num_lines, 3) # colors 1, 1, N, 3

        sigmas_view = sigmas.view(1, 1, sigmas.numel())

        # calculate influence  of each line on each pixel and convert to weight via gaussian kernel
        dist_sq = _point_segment_distance_sq(P, A, B, self.epsilon) # H, W, N

        variance = 2 * sigmas_view**2 + self.epsilon # 1, 1, N
        weights = torch.exp(-dist_sq / variance)     # H, W, N

        # multiply weights by line colors
        weighted_colors = weights.unsqueeze(-1) * Col # H, W, N, 3

        # sum all lines
        rendered_image_flat = torch.sum(weighted_colors, dim=2) # H, W, 3

        # clamp and permute
        rendered_image = torch.clamp(rendered_image_flat, 0.0, 1.0)
        rendered_image_chw = rendered_image.permute(2, 0, 1) # 3, H, W
        return rendered_image_chw, sigmas

    def get_loss(self):
        target_image = self.target_image
        rendered_image, sigmas = self._render()
        reconstruction_loss = self.loss(rendered_image, target_image)

        # penalize small and large sigmas
        sigma_penalty = torch.mean(sigmas) * self.thickness_penalty_weight
        small_sigmas = self.log_sigmas[self.log_sigmas<self.min_log_sigma]
        if len(small_sigmas) > 0:
            sigma_penalty = sigma_penalty + small_sigmas.pow(2).mean()


        loss = reconstruction_loss + sigma_penalty

        if self._make_images:
            img = rendered_image.detach() * 255
            self.log('image reconstructed', img.cpu().to(torch.uint8), log_test=False, to_uint8=False)
            self.log_difference("image difference", rendered_image, to_uint8=True)

        return loss

