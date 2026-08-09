import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ....benchmark import Benchmark
from ....utils import to_HW3, normalize


class GaborFiltersDrawer(Benchmark):
    """Reconstructs the passed colored image with Gabor filters.

    Gabor filters are sinusoidal gratings modulated by a Gaussian envelope.
    They are biologically-inspired and resemble receptive fields of simple cells
    in the primary visual cortex.

    Each Gabor filter has parameters:
        - position (x, y): center of the filter
        - orientation (theta): rotation angle of the sinusoidal grating
        - frequency (lambda): spatial frequency of the sinusoid
        - phase (psi): phase offset of the sinusoid
        - sigma: standard deviation of the Gaussian envelope
        - aspect_ratio: ellipticity of the Gaussian envelope
        - color (RGB): color of the filter
        - alpha: transparency/weight of the filter

    Args:
        target_image (Any):
            target image, either path to image, numpy array or torch tensor.
            Can be channel first or channel last or 2D.
        num_filters (int):
            number of Gabor filters for image reconstruction.
            There are also always 3 parameters for background color.
        initial_sharpness (float, optional):
            Initial sharpness parameter. Defaults to 50.
        exp_sharpness (bool, optional):
            Applies exp to sharpness. Defaults to False.
        min_sharpness (float, optional):
            Applies squared penalty for when sharpness is below this. Defaults to 10.
        penalty (float, optional):
            Multiplier to penalty for when sharpness is too low. Defaults to 0.1.
        loss_fn (Callable):
            loss function between reconstructed and target image. Defaults to F.mse_loss.
    """

    def __init__(
        self,
        target_image: Any,
        num_filters: int = 100,
        initial_sharpness: float = 50,
        exp_sharpness: bool = False,
        min_sharpness: float = 10,
        penalty: float = 0.1,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
    ):
        super().__init__()
        target_image = normalize(to_HW3(target_image, generator=self.rng.torch()).float(), 0, 1).moveaxis(-1, 0)
        # 3HW image
        self.target_image = nn.Buffer(target_image)
        self.add_reference_image('target', (target_image * 255).detach().cpu().numpy().astype(np.uint8), to_uint8=False)

        self.num_filters = num_filters
        self.loss_fn = loss_fn
        self.min_sharpness = min_sharpness
        self.penalty = penalty

        # learnable sharpness (controls edge sharpness of filters)
        self.exp_sharpness = exp_sharpness
        if exp_sharpness:
            self.sharpness = nn.Parameter(torch.log(torch.tensor(initial_sharpness, dtype=torch.float32)))
        else:
            self.sharpness = nn.Parameter(torch.tensor(initial_sharpness, dtype=torch.float32))

        # Gabor filter parameters (11 per filter):
        # - x, y: center position (2)
        # - theta: orientation angle (1)
        # - log_lambda: log of spatial frequency (1)
        # - psi: phase offset (1)
        # - log_sigma: log of Gaussian std (1)
        # - log_aspect: log of aspect ratio (1)
        # - RGB color (3)
        # - alpha: transparency (1)
        self.filter_params = nn.Parameter(torch.rand(num_filters, 11, generator=self.rng.torch()).logit())

        # Learnable background color (RGB)
        self.bg_color = nn.Parameter(torch.zeros(3))

        # Coordinate grid
        H, W = target_image.shape[-2:]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        self.x_grid = nn.Buffer(x_grid)
        self.y_grid = nn.Buffer(y_grid)

        self._show_titles_on_video = False

    def _gabor_response(self, x: torch.Tensor, y: torch.Tensor,
                        cx: torch.Tensor, cy: torch.Tensor,
                        theta: torch.Tensor, log_lambda: torch.Tensor,
                        psi: torch.Tensor, log_sigma: torch.Tensor,
                        log_aspect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gabor filter response at given coordinates.

        Args:
            x, y: Coordinate grids
            cx, cy: Filter center positions
            theta: Orientation (radians)
            log_lambda: Log spatial frequency
            psi: Phase offset
            log_sigma: Log Gaussian std
            log_aspect: Log aspect ratio

        Returns:
            Tuple of (gaussian envelope, sinusoid response)
        """
        # Transform coordinates to filter's rotated frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Translate to filter center
        x_prime = x - cx.unsqueeze(-1).unsqueeze(-1)
        y_prime = y - cy.unsqueeze(-1).unsqueeze(-1)

        # Rotate coordinates
        x_rot = x_prime * cos_theta.unsqueeze(-1).unsqueeze(-1) + y_prime * sin_theta.unsqueeze(-1).unsqueeze(-1)
        y_rot = -x_prime * sin_theta.unsqueeze(-1).unsqueeze(-1) + y_prime * cos_theta.unsqueeze(-1).unsqueeze(-1)

        # Gaussian envelope
        sigma_x = torch.exp(log_sigma).unsqueeze(-1).unsqueeze(-1)
        sigma_y = sigma_x * torch.exp(log_aspect).unsqueeze(-1).unsqueeze(-1)

        gaussian = torch.exp(
            -0.5 * (x_rot ** 2 / (sigma_x ** 2 + 1e-8) + y_rot ** 2 / (sigma_y ** 2 + 1e-8))
        )

        # Sinusoidal grating
        lambda_val = torch.exp(log_lambda).unsqueeze(-1).unsqueeze(-1)
        sinusoid = torch.cos(2 * math.pi * x_rot / (lambda_val + 1e-8) + psi.unsqueeze(-1).unsqueeze(-1))

        return gaussian, sinusoid

    def get_loss(self):
        # Normalize parameters
        p = torch.sigmoid(self.filter_params)
        bg_color = torch.sigmoid(self.bg_color)  # (3,)

        # Get device for creating constant tensors
        device = self.filter_params.device

        # Extract filter components
        # Position (normalized to image coordinates -1 to 1)
        cx = (p[:, 0] * 2 - 1)  # x position in [-1, 1]
        cy = (p[:, 1] * 2 - 1)  # y position in [-1, 1]

        # Orientation (0 to pi)
        theta = p[:, 2] * math.pi

        # Spatial frequency (log scale, range roughly 0.02 to 0.5)
        log_lambda = torch.lerp(torch.tensor(math.log(0.02), device=device), torch.tensor(math.log(0.5), device=device), p[:, 3])

        # Phase offset (0 to 2pi)
        psi = p[:, 4] * 2 * math.pi

        # Gaussian std (log scale, range roughly 0.01 to 0.5)
        log_sigma = torch.lerp(torch.tensor(math.log(0.01), device=device), torch.tensor(math.log(0.5), device=device), p[:, 5])

        # Aspect ratio (log scale, range roughly 0.2 to 1.0, i.e., elongated ellipses)
        log_aspect = torch.lerp(torch.tensor(math.log(0.2), device=device), torch.tensor(math.log(1.0), device=device), p[:, 6])

        # Color and alpha
        colors = p[:, 7:10]  # (N, 3) RGB
        alpha = p[:, 10]  # (N,) transparency

        if self.exp_sharpness:
            sharpness = torch.exp(self.sharpness)
        else:
            sharpness = self.sharpness

        if sharpness < self.min_sharpness:
            penalty = ((self.min_sharpness - sharpness) ** 2) * self.penalty
        else:
            penalty = 0

        # Compute Gabor responses for all filters
        # Shape: (num_filters, H, W) for both gaussian and sinusoid
        gaussian, sinusoid = self._gabor_response(
            self.x_grid, self.y_grid,
            cx, cy, theta, log_lambda, psi, log_sigma, log_aspect
        )

        # Use gaussian as the alpha mask (spatial localization)
        # Use sinusoid to modulate the intensity/color
        # Sinusoid ranges from -1 to 1, we map it to 0 to 1 for color modulation
        sinusoid_normalized = (sinusoid + 1) / 2  # Now ranges from 0 to 1

        # Weight responses by alpha and color
        alpha_expanded = alpha.view(-1, 1, 1, 1)  # (N, 1, 1, 1)
        colors_expanded = colors.view(-1, 3, 1, 1)  # (N, 3, 1, 1)
        gaussian_expanded = gaussian.unsqueeze(1)  # (N, 1, H, W)
        sinusoid_expanded = sinusoid_normalized.unsqueeze(1)  # (N, 1, H, W)

        # Filter contributions: (N, 3, H, W)
        # color * alpha * gaussian_envelope * sinusoid_modulation
        filter_contributions = colors_expanded * alpha_expanded * gaussian_expanded * sinusoid_expanded
        total_filter_contrib = torch.sum(filter_contributions, dim=0)  # (3, H, W)

        # Background contribution
        total_alpha = torch.sum(alpha_expanded * gaussian_expanded, dim=(0, 1)).unsqueeze(0)  # (1, H, W)
        bg_contrib = bg_color.view(3, 1, 1) * (1 - torch.clamp(total_alpha, 0, 1))  # (3, H, W)

        # Final reconstruction
        reconstructed = bg_contrib + total_filter_contrib
        reconstructed = torch.clamp(reconstructed, 0, 1)
        loss = self.loss_fn(reconstructed, self.target_image)

        if self.make_images:
            with torch.no_grad():
                img = reconstructed.detach().clamp(0, 1).permute(1, 2, 0) * 255
                self.log_image('reconstructed', img.cpu().numpy().astype(np.uint8), to_uint8=False, show_best=True)

        return loss + penalty
