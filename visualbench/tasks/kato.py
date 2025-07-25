import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark import Benchmark
from ..utils import to_CHW


class Kato(Benchmark):
    """goal is to find image whose laplacian is target, and that's pretty hard.

    Renders:
        image and its laplacian.

    """
    def __init__(self, target):
        super().__init__()
        self.target = nn.Buffer(to_CHW(target).float().unsqueeze(0))

        self.u = nn.Parameter(torch.randn_like(self.target) * 0.1)

        # discrete Laplacian kernel
        laplacian = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32)

        self.kernel = nn.Buffer(laplacian.unsqueeze(0).unsqueeze(0).repeat_interleave(self.target.size(1), 0))#.repeat_interleave(self.f_target.size(1), 1))

        self.add_reference_image('target', image = self.target, to_uint8=True)

    def get_loss(self):
        u_laplacian = F.conv2d(self.u, self.kernel, groups=self.u.size(1), padding='same')# pylint:disable=not-callable
        loss = F.mse_loss(u_laplacian, self.target)

        if self._make_images:
            with torch.no_grad():
                self.log_image("laplacian u", self.u, to_uint8=True, log_difference=True)
                self.log_image('image ∇²u', u_laplacian, to_uint8=True)

        return loss
