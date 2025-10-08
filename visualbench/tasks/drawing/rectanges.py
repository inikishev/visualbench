from typing import Any
from collections.abc import Callable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import to_HW3, CUDA_IF_AVAILABLE, normalize


class RectanglesDrawer(Benchmark):
    """Reconstructs the passed colored image with soft semi-transparent rectangles.

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_rectangles (int):
            number of rectangles for image reconstruction.
            Each rectangle has 8 parameters - 4 coordinates, 3 color values and alpha.
            There are also always 3 parameters for background color and 1 for shapness.
        initial_sharpness (float, optional):
            Initial sharpness (it is a learnable parameter and will get optimized). Defaults to 150.
        exp_sharpness (bool, optional):
            Applies exp to sharpness. Defaults to False.
        min_sharpness (float, optional):
            Applies squared penalty for when shapness is below this. Defaults to 100.
        penalty (bool, optional):
            Multiplier to penalty for when sharness is too low. Defaults to 1.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """
    x_grid: torch.nn.Buffer
    y_grid: torch.nn.Buffer
    target_image: torch.nn.Buffer
    def __init__(
        self,
        target_image,
        num_rectangles: int = 100,
        initial_sharpness: float = 150,
        exp_sharpness = False,
        min_sharpness: float = 100,
        penalty: float = 1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.register_buffer('target_image', target_image)
        self.add_reference_image('target', (target_image*255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_rectangles = num_rectangles
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Rectangle parameters (cx, cy, w, h, r, g, b, a)
        self.rect_params = nn.Parameter(torch.rand(num_rectangles, 8, generator = self.rng.torch()))

        # Learnable background color (RGB)
        self.bg_color = nn.Parameter(torch.zeros(3))  # Initialized to black

        # Coordinate grid
        H, W = target_image.shape[-2:]

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing='ij'
        )
        self.x_grid = torch.nn.Buffer(x_grid)
        self.y_grid = torch.nn.Buffer(y_grid)

        self._show_titles_on_video = False

    def get_loss(self):
        # Normalize parameters to (0,1)
        p = torch.sigmoid(self.rect_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        # Extract rectangle components
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        colors = p[:, 4:7]  # (N, 3)
        alpha = p[:, 7]     # (N,)

        # Calculate boundaries with constraints
        left = cx - w/2
        right = cx + w/2
        top = cy - h/2
        bottom = cy + h/2

        # Reshape for broadcasting
        left = left.view(-1, 1, 1)
        right = right.view(-1, 1, 1)
        top = top.view(-1, 1, 1)
        bottom = bottom.view(-1, 1, 1)

        if self.exp_sharpness: sharpness = torch.exp(self.sharpness)
        else: sharpness = self.sharpness
        if sharpness < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty

        else:
            penalty = 0

        # Calculate coverage
        x_coverage = (
            torch.sigmoid((self.x_grid - left) * sharpness) -
            torch.sigmoid((self.x_grid - right) * sharpness)
        )
        y_coverage = (
            torch.sigmoid((self.y_grid - top) * sharpness) -
            torch.sigmoid((self.y_grid - bottom) * sharpness)
        )
        mask = x_coverage * y_coverage  # (N, H, W)

        # Calculate contributions
        alpha = alpha.view(-1, 1, 1, 1)
        colors = colors.view(-1, 3, 1, 1)
        mask = mask.view(-1, 1, *mask.shape[1:])

        # Rectangle contributions
        rect_contributions = colors * alpha * mask  # (N, 3, H, W)
        total_rect_contrib = torch.sum(rect_contributions, dim=0)  # (3, H, W)

        # Background contribution (visible where rectangles don't cover)
        total_alpha = torch.sum(alpha * mask, dim=0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - total_alpha)  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_rect_contrib
        loss = self.loss_fn(reconstructed, self.target_image)

        if self._make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0)*255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty