import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .._utils import _make_float_3hw_tensor
from ..benchmark import Benchmark


class KatosProblem(Benchmark):
    """goal is to find image whose laplacian is target, and that's pretty hard"""
    def __init__(self, target):
        super().__init__(log_projections=True)
        self.f_target = nn.Buffer(_make_float_3hw_tensor(target).unsqueeze(0))

        self.u = nn.Parameter(torch.randn_like(self.f_target) * 0.1)

        # discrete Laplacian kernel
        laplacian_kernel_np = np.array([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=np.float32)

        self.laplacian_kernel = nn.Buffer(torch.tensor(laplacian_kernel_np).unsqueeze(0).unsqueeze(0).repeat_interleave(self.f_target.size(1), 0))#.repeat_interleave(self.f_target.size(1), 1))

        self.add_reference_image('target', image = self.f_target, to_uint8=True)

    def apply_laplacian(self, x):
        return F.conv2d(x, self.laplacian_kernel, groups=x.size(1), padding='same')# pylint:disable=not-callable

    def get_loss(self):
        laplacian_u = self.apply_laplacian(self.u)
        loss = F.mse_loss(laplacian_u, self.f_target)

        if self._make_images:
            with torch.no_grad():
                self.log("image laplacian u", self.u, log_test=False, to_uint8=True)
                self.log('image ∇²u', laplacian_u, log_test=False, to_uint8=True)
                self.log_difference("image u difference", self.u, to_uint8=True)

        return loss

    def get_target_f_image(self):
        f_display_np = self.f_target.detach().cpu().squeeze(0).moveaxis(0,-1).numpy()
        min_val, max_val = f_display_np.min(), f_display_np.max()
        if max_val - min_val > 1e-6:
            f_norm = (f_display_np - min_val) / (max_val - min_val)
        else:
            f_norm = np.zeros_like(f_display_np)
        f_uint8 = (f_norm * 255).astype(np.uint8)
        return f_uint8

    def get_laplacian_u_image(self):
        with torch.no_grad():
            lap_u = self.apply_laplacian(self.u)
            lap_u_np = lap_u.detach().cpu().squeeze(0).moveaxis(0,-1).numpy()
            min_val, max_val = lap_u_np.min(), lap_u_np.max()
            if max_val - min_val > 1e-6:
                lap_u_norm = (lap_u_np - min_val) / (max_val - min_val)
            else:
                lap_u_norm = np.zeros_like(lap_u_np)
            lap_u_uint8 = (lap_u_norm * 255).astype(np.uint8)
            return lap_u_uint8

