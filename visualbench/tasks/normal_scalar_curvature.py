import cv2  # For image generation
import matplotlib.pyplot as plt  # For colormap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..benchmark import Benchmark


# Helper function to create convolution kernels for derivatives
def get_derivative_kernels(dtype=torch.float32): # Device will be handled by .to(device) later
    # First derivatives (central difference)
    kernel_fx = torch.tensor([[[[-1, 0, 1]]]], dtype=dtype) # For d/dx
    kernel_fy = torch.tensor([[[[-1], [0], [1]]]], dtype=dtype) # For d/dy

    # Second derivatives (central difference)
    kernel_fxx = torch.tensor([[[[1, -2, 1]]]], dtype=dtype) # For d^2/dx^2
    kernel_fyy = torch.tensor([[[[1], [-2], [1]]]], dtype=dtype) # For d^2/dy^2

    return {
        'fx': kernel_fx, 'fy': kernel_fy,
        'fxx': kernel_fxx, 'fyy': kernel_fyy
    }

class NormalScalarCurvature(Benchmark):
    """Creates cool images on some optimizers"""
    def __init__(self, grid_size=128, domain_size=16.0, target_positive_curvature=0.3, cmap='coolwarm'):
        super().__init__(log_projections=True)
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.target_positive_curvature = target_positive_curvature

        self.dx = self.domain_size / (self.grid_size -1)
        self.dy = self.dx

        self.z_field = nn.Parameter(torch.randn(grid_size, grid_size) * 0.05)

        # Store host-based kernels (templates). They will be moved to the correct device
        # and cast to the correct dtype in _get_kernels_on_device.
        self._kernels_on_host = get_derivative_kernels(dtype=torch.float32)
        self.kernels = {}
        self.colormap = plt.get_cmap(cmap)

    def _get_kernels_on_device(self):
        device = self.z_field.device
        dtype = self.z_field.dtype

        # Check if 'kernels' attribute exists and if its contents are on the correct device/dtype
        if not self.kernels or \
           self.kernels['fx'].device != device or \
           self.kernels['fx'].dtype != dtype:
            # print(f"Updating kernels for device: {device}, dtype: {dtype}") # For debugging
            self.kernels = {name: k_host.to(device=device, dtype=dtype)
                            for name, k_host in self._kernels_on_host.items()}
        return self.kernels

    def get_loss(self):
        kernels = self._get_kernels_on_device()

        z = self.z_field.unsqueeze(0).unsqueeze(0) # (B, C, H, W) = (1, 1, H, W)
        padding_mode = 'replicate' # Or 'circular', 'reflect'

        # --- fx ---
        k_fx = kernels['fx'] # Kernel shape (1,1,kH,kW), e.g. (1,1,1,3)
        # Padding for F.conv2d would be ( (kH-1)//2, (kW-1)//2 )
        # For F.pad, it's (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
        Ph_fx = (k_fx.shape[2] - 1) // 2 # Padding for Height dimension
        Pw_fx = (k_fx.shape[3] - 1) // 2 # Padding for Width dimension
        pad_tuple_fx = (Pw_fx, Pw_fx, Ph_fx, Ph_fx)
        z_padded_fx = F.pad(z, pad_tuple_fx, mode=padding_mode)
        fx = F.conv2d(z_padded_fx, k_fx, padding=0) / (2 * self.dx) # pylint:disable=not-callable

        # --- fy ---
        k_fy = kernels['fy'] # Kernel shape (1,1,3,1)
        Ph_fy = (k_fy.shape[2] - 1) // 2
        Pw_fy = (k_fy.shape[3] - 1) // 2
        pad_tuple_fy = (Pw_fy, Pw_fy, Ph_fy, Ph_fy)
        z_padded_fy = F.pad(z, pad_tuple_fy, mode=padding_mode)
        fy = F.conv2d(z_padded_fy, k_fy, padding=0) / (2 * self.dy) # pylint:disable=not-callable

        # --- fxx ---
        k_fxx = kernels['fxx'] # Kernel shape (1,1,1,3)
        Ph_fxx = (k_fxx.shape[2] - 1) // 2
        Pw_fxx = (k_fxx.shape[3] - 1) // 2
        pad_tuple_fxx = (Pw_fxx, Pw_fxx, Ph_fxx, Ph_fxx)
        z_padded_fxx = F.pad(z, pad_tuple_fxx, mode=padding_mode)
        fxx = F.conv2d(z_padded_fxx, k_fxx, padding=0) / (self.dx**2) # pylint:disable=not-callable

        # --- fyy ---
        k_fyy = kernels['fyy'] # Kernel shape (1,1,3,1)
        Ph_fyy = (k_fyy.shape[2] - 1) // 2
        Pw_fyy = (k_fyy.shape[3] - 1) // 2
        pad_tuple_fyy = (Pw_fyy, Pw_fyy, Ph_fyy, Ph_fyy)
        z_padded_fyy = F.pad(z, pad_tuple_fyy, mode=padding_mode)
        fyy = F.conv2d(z_padded_fyy, k_fyy, padding=0) / (self.dy**2) # pylint:disable=not-callable

        # --- fxy = d/dy (fx) ---
        # Input is fx, kernel is k_fy. We use pad_tuple_fy calculated for k_fy.
        fx_padded_for_fxy = F.pad(fx, pad_tuple_fy, mode=padding_mode)
        # Note: fx already has 1/(2*dx) scaling. Resulting fxy has 1/(4*dx*dy) scaling.
        fxy = F.conv2d(fx_padded_for_fxy, k_fy, padding=0) / (2 * self.dy) # pylint:disable=not-callable

        # Scalar Curvature K
        fx2 = fx**2
        fy2 = fy**2

        numerator = 2 * ( (1 + fy2) * fxx - 2 * fx * fy * fxy + (1 + fx2) * fyy )
        denominator = (1 + fx2 + fy2)**2

        epsilon = 1e-8
        scalar_curvature_K = numerator / (denominator + epsilon)

        # Loss function
        loss = torch.mean((torch.relu(self.target_positive_curvature - scalar_curvature_K))**2)

        # --- Visualization ---
        if self._make_images:
            K_detached = scalar_curvature_K.detach().squeeze().cpu().numpy()

            # Normalize K for visualization.
            vis_abs_bound = max(0.1, 2.0 * abs(self.target_positive_curvature)) # Ensure a minimum visual range
            vis_min = -vis_abs_bound
            vis_max = vis_abs_bound

            if self.target_positive_curvature > 0: # Center slightly towards positive
                vis_min = -vis_abs_bound / 2
                vis_max = vis_abs_bound * 1.5
            elif self.target_positive_curvature < 0: # Center slightly towards negative
                vis_min = -vis_abs_bound * 1.5
                vis_max = vis_abs_bound / 2

            # Ensure target is within visualized range for clarity
            vis_min = min(vis_min, self.target_positive_curvature - 0.1*abs(self.target_positive_curvature) if self.target_positive_curvature!=0 else -0.05)
            vis_max = max(vis_max, self.target_positive_curvature + 0.1*abs(self.target_positive_curvature) if self.target_positive_curvature!=0 else 0.05)
            if vis_min == vis_max: # Avoid division by zero if target is 0 and bounds collapse
                vis_min -= 0.1
                vis_max += 0.1


            K_norm = np.clip((K_detached - vis_min) / (vis_max - vis_min + epsilon), 0, 1)
            colored_K = self.colormap(K_norm)[:, :, :3] # Take RGB, discard Alpha
            frame_cv = (colored_K * 255).astype(np.uint8)

            self.log('image', frame_cv, log_test=False, to_uint8=False)
            self.log_difference('image difference', frame_cv, to_uint8=False)

        return loss