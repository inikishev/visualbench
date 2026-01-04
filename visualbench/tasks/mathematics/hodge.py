from typing import Any

import cv2
import numpy as np
import torch

from ...benchmark import Benchmark
from ...utils import to_CHW, to_square


class HodgeConjecture(Benchmark):
    """Basically looks kinda cool

    Numerical optimization inspired by the Hodge Conjecture on a Complex Torus.

    Goal:
    Find an algebraic cycle (the zero set of a complex function f) whose
    fundamental class matches a target Hodge class.

    Mathematical Formulation:
    1. We represent f as a sum of Fourier harmonics f(z) = Σ c_k e^{i k z}.
    2. The 'density' of the zero set is given by the Poincaré-Lelong formula:
       Density ≈ Δ log|f|.
    3. We optimize the coefficients c_k such that the density matches the target.

    Args:
        target (torch.Tensor, optional): Target Hodge class of shape (C, H, W).
            C is treated as a batch dimension (independent classes).
            If None, a default 'three circles' pattern is generated.
        n_freqs (int): Number of Fourier frequencies (complexity of the cycle).
    """
    def __init__(self, target: Any | None, n_freqs: int = 16):
        super().__init__()

        if target is None:
            # Generate default "Three Circles" target
            grid_size = 256
            y, x = torch.meshgrid(torch.linspace(0, 1, grid_size), torch.linspace(0, 1, grid_size), indexing='ij')
            c1 = ((x - 0.3)**2 + (y - 0.3)**2 < 0.015).float()
            c2 = ((x - 0.7)**2 + (y - 0.4)**2 < 0.02).float()
            c3 = ((x - 0.5)**2 + (y - 0.8)**2 < 0.018).float()
            target = (c1 + c2 + c3).clamp(0, 1).unsqueeze(0) # Shape (1, 256, 256)

        else:
            target = to_square(to_CHW(target))

        self.C, self.H, self.W = target.shape
        self.target_class = torch.nn.Buffer(target.float())

        # Parameters: Fourier coefficients for each channel
        # Shape: (Batch, Freq, Freq, Real/Imag)
        freq_size = n_freqs * 2 + 1
        self.coeffs = torch.nn.Parameter(torch.randn(self.C, freq_size, freq_size, 2) * 0.01)

        self.add_reference_image("Target Hodge Class", self.target_class, to_uint8=True)

    def get_f(self):
        """Reconstructs the complex function f(z) from Fourier coefficients."""
        complex_coeffs = torch.view_as_complex(self.coeffs)

        # Calculate padding to reach target resolution (H, W)
        pad_h = self.H - complex_coeffs.shape[1]
        pad_w = self.W - complex_coeffs.shape[2]

        p_t, p_b = pad_h // 2, pad_h - (pad_h // 2)
        p_l, p_r = pad_w // 2, pad_w - (pad_w // 2)

        # pad logic: (last_dim_front, last_dim_back, prev_dim_front, prev_dim_back)
        padded = torch.nn.functional.pad(complex_coeffs, (p_l, p_r, p_t, p_b))

        # Batch IFFT across channels
        f_spatial = torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(-2, -1))) * (self.H * self.W) # pylint:disable=not-callable
        return f_spatial

    def get_loss(self):
        f = self.get_f()

        # |f|^2 per channel
        f_sq_mag = torch.view_as_real(f).pow(2).sum(-1)
        log_abs_f = torch.log(f_sq_mag + 1e-6)

        # 2D Laplacian for the batch (C, H, W)
        laplacian = (
            torch.roll(log_abs_f, 1, dims=1) + torch.roll(log_abs_f, -1, dims=1) +
            torch.roll(log_abs_f, 1, dims=2) + torch.roll(log_abs_f, -1, dims=2) - 4 * log_abs_f
        )

        # Map Laplacian to a normalized density [0, 1]
        # High Laplacian = high curvature = location of zeros
        current_cycle_density = torch.sigmoid(laplacian * 15.0)

        loss = torch.nn.functional.mse_loss(current_cycle_density, self.target_class)
        self.log("MSE Loss", loss)

        if self._make_images:
            # 1. Visualize Magnitude (where it is dark, zeros exist)
            f_abs = f_sq_mag.sqrt()
            self.log_image("Function Magnitude |f|", f_abs, to_uint8=True)

            # 2. Visualize Current Cycle with Colormapping
            if self.C == 1:
                # Single channel: apply colormap
                cycle_np = current_cycle_density[0].detach().cpu().numpy()
                cycle_uint8 = (cycle_np * 255).astype(np.uint8)
                colored = cv2.applyColorMap(cycle_uint8, cv2.COLORMAP_JET) # pylint:disable=no-member
                # cv2 (H, W, BGR) -> visualbench (RGB, H, W)
                cycle_vis = torch.from_numpy(colored).permute(2, 0, 1).flip(0)
            else:
                # Multi-channel: treat as RGB or generic batch
                cycle_vis = current_cycle_density

            self.log_image("Algebraic Cycle Reconstruction", cycle_vis, to_uint8=(self.C > 1))

            # 3. Error map
            error = torch.abs(current_cycle_density - self.target_class)
            self.log_image("Matching Error", error, to_uint8=True)

        return loss
