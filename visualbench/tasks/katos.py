import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import to_CHW
from ..benchmark import Benchmark


class KatosProblem(Benchmark):
    """goal is to find image whose laplacian is target, and that's pretty hard"""
    def __init__(self, target):
        super().__init__()
        self.f_target = nn.Buffer(to_CHW(target).float().unsqueeze(0))

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
                self.log_image("laplacian u", self.u, to_uint8=True, log_difference=True)
                self.log_image('image ∇²u', laplacian_u, to_uint8=True)

        return loss
