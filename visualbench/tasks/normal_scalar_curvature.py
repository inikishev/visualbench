import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..benchmark import Benchmark


def _get_derivative_kernels(dtype=torch.float32):
    kernel_fx = torch.tensor([[[[-1, 0, 1]]]], dtype=dtype) # For d/dx
    kernel_fy = torch.tensor([[[[-1], [0], [1]]]], dtype=dtype) # For d/dy

    kernel_fxx = torch.tensor([[[[1, -2, 1]]]], dtype=dtype) # For d^2/dx^2
    kernel_fyy = torch.tensor([[[[1], [-2], [1]]]], dtype=dtype) # For d^2/dy^2

    return {
        'fx': kernel_fx, 'fy': kernel_fy,
        'fxx': kernel_fxx, 'fyy': kernel_fyy
    }

class NormalScalarCurvature(Benchmark):
    """TODO

    Renders:
        z field and solution.
    """
    def __init__(self, grid_size=128, domain=16.0, target_curvature=0.3, cmap='coolwarm'):
        super().__init__()
        self.grid_size = grid_size
        self.domain = domain
        self.target_curvature = target_curvature

        self.dx = self.domain / (self.grid_size -1)
        self.dy = self.dx

        self.z_field = nn.Parameter(torch.randn(grid_size, grid_size) * 0.05)

        self._kernels = _get_derivative_kernels(dtype=torch.float32)
        self.kernels = {}
        self.colormap = plt.get_cmap(cmap)

    def _get_kernels(self):
        device = self.z_field.device
        dtype = self.z_field.dtype

        # Check if 'kernels' attribute exists and if its contents are on the correct device/dtype
        if not self.kernels or \
           self.kernels['fx'].device != device or \
           self.kernels['fx'].dtype != dtype:

            self.kernels = {name: k_host.to(device=device, dtype=dtype)
                            for name, k_host in self._kernels.items()}
        return self.kernels

    def get_loss(self):
        kernels = self._get_kernels()

        z = self.z_field.unsqueeze(0).unsqueeze(0) # (B, C, H, W) = (1, 1, H, W)
        padding_mode = 'replicate' # Or 'circular', 'reflect'

        # ------------------------------------ fx ------------------------------------ #
        k_fx = kernels['fx'] # Kernel shape (1,1,3,3)
        Ph_fx = (k_fx.shape[2] - 1) // 2
        Pw_fx = (k_fx.shape[3] - 1) // 2
        z_fx = F.pad(z, (Pw_fx, Pw_fx, Ph_fx, Ph_fx), mode=padding_mode)
        fx = F.conv2d(z_fx, k_fx, padding=0) / (2 * self.dx) # pylint:disable=not-callable

        # ------------------------------------ fy ------------------------------------ #
        k_fy = kernels['fy'] # Kernel shape (1,1,3,1)
        Ph_fy = (k_fy.shape[2] - 1) // 2
        Pw_fy = (k_fy.shape[3] - 1) // 2
        pad_fy = (Pw_fy, Pw_fy, Ph_fy, Ph_fy)
        z_fy = F.pad(z, pad_fy, mode=padding_mode)
        fy = F.conv2d(z_fy, k_fy, padding=0) / (2 * self.dy) # pylint:disable=not-callable

        # ------------------------------------ fxx ----------------------------------- #
        k_fxx = kernels['fxx'] # Kernel shape (1,1,1,3)
        Ph_fxx = (k_fxx.shape[2] - 1) // 2
        Pw_fxx = (k_fxx.shape[3] - 1) // 2
        z_fxx = F.pad(z, (Pw_fxx, Pw_fxx, Ph_fxx, Ph_fxx), mode=padding_mode)
        fxx = F.conv2d(z_fxx, k_fxx, padding=0) / (self.dx**2) # pylint:disable=not-callable

        # ------------------------------------ fyy ----------------------------------- #
        k_fyy = kernels['fyy'] # Kernel shape (1,1,3,1)
        Ph_fyy = (k_fyy.shape[2] - 1) // 2
        Pw_fyy = (k_fyy.shape[3] - 1) // 2
        z_padded_fyy = F.pad(z, (Pw_fyy, Pw_fyy, Ph_fyy, Ph_fyy), mode=padding_mode)
        fyy = F.conv2d(z_padded_fyy, k_fyy, padding=0) / (self.dy**2) # pylint:disable=not-callable

        # ------------------------------ fxy = d/dy (fx) ----------------------------- #
        fxy = F.conv2d(F.pad(fx, pad_fy, mode=padding_mode), k_fy, padding=0) / (2 * self.dy) # pylint:disable=not-callable

        # scalar curvature K
        fx2 = fx**2
        fy2 = fy**2

        num = 2 * ( (1 + fy2) * fxx - 2 * fx * fy * fxy + (1 + fx2) * fyy )
        denom = (1 + fx2 + fy2)**2

        epsilon = 1e-8
        K = num / (denom + epsilon)

        # Loss function
        loss = torch.mean((torch.relu(self.target_curvature - K))**2)

        # Frame
        if self._make_images:
            with torch.no_grad():
                K_detached = K.detach().squeeze().cpu().numpy()

                vis_abs_bound = max(0.1, 2.0 * abs(self.target_curvature)) # Ensure a minimum visual range
                vis_min = -vis_abs_bound
                vis_max = vis_abs_bound

                if self.target_curvature > 0:
                    vis_min = -vis_abs_bound / 2
                    vis_max = vis_abs_bound * 1.5
                elif self.target_curvature < 0:
                    vis_min = -vis_abs_bound * 1.5
                    vis_max = vis_abs_bound / 2

                vis_min = min(vis_min, self.target_curvature - 0.1*abs(self.target_curvature) if self.target_curvature!=0 else -0.05)
                vis_max = max(vis_max, self.target_curvature + 0.1*abs(self.target_curvature) if self.target_curvature!=0 else 0.05)
                if vis_min == vis_max:
                    vis_min -= 0.1
                    vis_max += 0.1


                K_norm = np.clip((K_detached - vis_min) / (vis_max - vis_min + epsilon), 0, 1)
                colored_K = self.colormap(K_norm)[:, :, :3]
                frame = (colored_K * 255).astype(np.uint8)

                self.log_image('z field', self.z_field, to_uint8=True)
                self.log_image('solution', frame, to_uint8=False, log_difference=True, show_best=True)

        return loss