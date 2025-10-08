import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...benchmark import Benchmark
from ...utils import to_HW3


class PartitionDrawer(Benchmark):
    """
    """

    def __init__(
        self,
        image,
        num_points=100,
        points_init="random",
        colors_init="random",
        min_softmax_beta=1000.0,
        loss=F.mse_loss,
        make_images=True,
    ):
        super().__init__()

        self.num_points = num_points
        self.softmax_beta = nn.Parameter(torch.tensor(min_softmax_beta, dtype=torch.float32))
        self.min_softmax_beta = min_softmax_beta

        image = to_HW3(image, generator=self.rng.torch()).float()
        image = image - image.min()
        image = image / image.max()
        self.target = nn.Buffer(image)
        self.height, self.width, _ = self.target.shape

        self.loss = loss
        self._make_images = make_images

        # init points
        if points_init == 'random':
            points_init = torch.rand(self.num_points, 2, device=self.device, generator=self.rng.torch())
        elif points_init == 'uniform_grid':
            grid_size = int(np.ceil(np.sqrt(num_points)))
            x = torch.linspace(0.05, 0.95, grid_size, device=self.device)
            y = torch.linspace(0.05, 0.95, grid_size, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            points_init = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            points_init = points_init[:num_points]
            points_init += (torch.rand(points_init.size(), generator=self.rng.torch()) - 0.5) * (1.0 / grid_size) * 0.5
            points_init.clamp_(0, 1)
        else:
            raise ValueError(f"Unknown init_points_strategy: {points_init}")
        self.points = nn.Parameter(points_init) # num_points, 2

        # init colors
        if colors_init == 'random':
            colors_init = torch.rand(self.num_points, 3, device=self.device, generator=self.rng.torch())
        elif colors_init == 'target_sample':
            points_pixel = (points_init.data.clamp(0, 1) * torch.tensor([self.width - 1, self.height - 1], device=self.device)).round().long()
            points_pixel[:, 0].clamp_(0, self.width - 1)
            points_pixel[:, 1].clamp_(0, self.height - 1)
            colors_init = self.target[points_pixel[:, 1], points_pixel[:, 0]]
            colors_init += torch.randn(colors_init.size(), generator=self.rng.torch()) * 0.01
            colors_init.clamp_(0, 1)
        else:
            raise ValueError(f"Unknown init_colors_strategy: {colors_init}")
        self.colors = nn.Parameter(colors_init) # num_points, 3

        # precompute pixel coords
        y_coords = torch.linspace(0, 1, self.height, device=self.device)
        x_coords = torch.linspace(0, 1, self.width, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.pixel_coords = nn.Buffer(torch.stack([grid_x, grid_y], dim=-1)) # H, W, 2
        self.flat_pixel_coords = nn.Buffer(self.pixel_coords.view(-1, 2)) # height * width, 2

        self.frames = []

        self.add_reference_image('target', self.target, to_uint8=True)

        self._show_titles_on_video = False

    def get_loss(self):
        pixels = self.flat_pixel_coords.unsqueeze(1) # H*W, 1, 2
        points = self.points.unsqueeze(0) # 1, num_points, 2
        dist_sq = torch.sum((pixels - points)**2, dim=2) # H*W, num_points
        weights = F.softmax(-self.softmax_beta * dist_sq, dim=1) # H*W, num_points
        assigned_colors = weights @ self.colors
        render = assigned_colors.view(self.height, self.width, 3).clamp(0, 1)
        loss = self.loss(render, self.target)

        # penalty for blurry image
        if self.softmax_beta < self.min_softmax_beta:
            loss = loss + (self.min_softmax_beta - self.softmax_beta) ** 2

        # make images
        if self._make_images:
            with torch.no_grad():
                self.log_image('reconstructed', render, to_uint8=True, show_best=True)
                self.log_image('residual', (render - self.target).abs_(), to_uint8=True)

        return loss
