import math
from typing import Literal, cast

import cv2
import numpy as np
import torch

from ...benchmark import Benchmark


def poly_from_roots(roots):
    """
    Computes polynomial coefficients from roots using iterative convolution.
    Returns tensor of complex coefficients [a_n, a_{n-1}, ..., a_0]
    """
    # Start with P(z) = 1 (coefficient of z^0 is 1)
    coeffs = torch.tensor([1.0 + 0.0j], device=roots.device, dtype=roots.dtype)
    for r in roots:
        # Multiply current poly by (z - r)
        # poly_new = [coeffs, 0] - [0, r * coeffs]
        new_coeffs = torch.zeros(coeffs.shape[0] + 1, device=roots.device, dtype=roots.dtype)
        new_coeffs[:-1] += coeffs
        new_coeffs[1:] -= r * coeffs
        coeffs = new_coeffs
    return coeffs

def eval_poly(coeffs, z):
    """Evaluates polynomial at complex points z using Horner's method."""
    res = torch.zeros_like(z)
    for c in coeffs:
        res = res * z + c
    return res

def get_derivative_coeffs(coeffs):
    """Returns complex coefficients of the derivative polynomial."""
    n = coeffs.shape[0] - 1
    if n <= 0:
        return torch.tensor([0.0 + 0.0j], device=coeffs.device, dtype=coeffs.dtype)

    powers = torch.arange(n, 0, -1, device=coeffs.device, dtype=torch.float32)

    return coeffs[:-1] * powers

def nan_to_num(x):
    if math.isfinite(x): return x
    return 0

class CasasAlvero(Benchmark):
    """optimize some points doing something"""
    def __init__(self, n: int = 6, min_method: Literal["min", "prod", "softmin"] = "min"):
        super().__init__()
        self.n = n

        initial_roots = torch.randn(n - 2, 2) * 0.5
        self.free_roots = torch.nn.Parameter(initial_roots)

        self.fixed_roots = torch.nn.Buffer(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
        self.min_method = min_method

    def get_roots(self):
        roots_raw = torch.cat([self.fixed_roots, self.free_roots], dim=0)
        return torch.complex(roots_raw[:, 0], roots_raw[:, 1])

    def get_loss(self):
        roots = self.get_roots()
        coeffs = poly_from_roots(roots)

        total_loss = cast(torch.Tensor, 0)
        current_coeffs = coeffs

        # For each derivative k = 1 to n-1
        for k in range(1, self.n):
            current_coeffs = get_derivative_coeffs(current_coeffs)

            # Condition: P^(k) and P must share a root.
            # We evaluate P^(k) at all roots of P and take the minimum magnitude.
            vals = eval_poly(current_coeffs, roots)

            if self.min_method == "min": shared_root_loss = torch.min(torch.abs(vals)**2)
            elif self.min_method == "prod": shared_root_loss = torch.prod(torch.abs(vals)**2 + 1e-8).pow(1/self.n)
            elif self.min_method == "softmin": shared_root_loss = -0.1 * torch.logsumexp(-torch.abs(vals)**2 / 0.1, dim=0)
            else: raise ValueError(self.min_method)

            total_loss = total_loss + shared_root_loss
            self.log(f"deriv_{k}_loss", shared_root_loss)

        if self._make_images:
            self.log_image(name="complex_plane", image=self.render_plane(roots), to_uint8=False)

        return total_loss

    @torch.no_grad
    def render_plane(self, roots, size=512):
        rng = 2.0
        # Create coordinate grid
        x = np.linspace(-rng, rng, size)
        y = np.linspace(-rng, rng, size)
        grid_x, grid_y = np.meshgrid(x, y)
        z_grid = torch.tensor(grid_x + 1j * grid_y, device=roots.device, dtype=roots.dtype)

        # Background: log magnitude of the polynomial
        coeffs = poly_from_roots(roots)
        vals = eval_poly(coeffs, z_grid)
        mag = torch.log1p(torch.abs(vals)).cpu().numpy()

        # Normalize for visualization
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        img = (mag * 150).astype(np.uint8) # Darker background
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS) # pylint:disable=no-member

        # Draw roots
        roots_np = roots.cpu().numpy()
        for i, r in enumerate(roots_np):
            px = int((nan_to_num(r.real) + rng) / (2 * rng) * (size - 1))
            py = int((nan_to_num(r.imag) + rng) / (2 * rng) * (size - 1))

            # Green for fixed roots, Cyan for variables
            color = (0, 255, 0) if i < 2 else (255, 255, 0)
            cv2.circle(img, (px, py), 4, color, -1, cv2.LINE_AA) # pylint:disable=no-member

        return img
