import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ..benchmark import Benchmark
from .._utils import _make_float_hw3_tensor, _normalize_to_uint8



class PartitionReconstructor(Benchmark):
    """
    """

    def __init__(
        self,
        target_image,
        num_points=100,
        init_points_strategy="random",
        init_colors_strategy="random",
        min_softmax_beta=300.0,
        loss=F.mse_loss,
        make_images=True,
    ):
        super().__init__(log_projections=True)

        self.num_points = num_points
        self.softmax_beta = nn.Parameter(torch.tensor(min_softmax_beta, dtype=torch.float32))
        self.min_softmax_beta = min_softmax_beta

        target_image = _make_float_hw3_tensor(target_image)
        target_image = target_image - target_image.min()
        target_image = target_image / target_image.max()
        self.target_image_float = nn.Buffer(target_image)
        self.height, self.width, _ = self.target_image_float.shape

        self.loss = loss
        self._make_images = make_images

        # init points
        if init_points_strategy == 'random':
            points_init = torch.rand(self.num_points, 2, device=self.device)
        elif init_points_strategy == 'uniform_grid':
            grid_size = int(np.ceil(np.sqrt(num_points)))
            x = torch.linspace(0.05, 0.95, grid_size, device=self.device)
            y = torch.linspace(0.05, 0.95, grid_size, device=self.device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            points_init = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            points_init = points_init[:num_points]
            points_init += (torch.rand_like(points_init) - 0.5) * (1.0 / grid_size) * 0.5
            points_init.clamp_(0, 1)
        else:
            raise ValueError(f"Unknown init_points_strategy: {init_points_strategy}")
        self.points = nn.Parameter(points_init) # num_points, 2

        # init colors
        if init_colors_strategy == 'random':
            colors_init = torch.rand(self.num_points, 3, device=self.device)
        elif init_colors_strategy == 'target_sample':
            points_pixel = (points_init.data.clamp(0, 1) * torch.tensor([self.width - 1, self.height - 1], device=self.device)).round().long()
            points_pixel[:, 0].clamp_(0, self.width - 1)
            points_pixel[:, 1].clamp_(0, self.height - 1)
            colors_init = self.target_image_float[points_pixel[:, 1], points_pixel[:, 0]]
            colors_init += torch.randn_like(colors_init) * 0.01
            colors_init.clamp_(0, 1)
        else:
            raise ValueError(f"Unknown init_colors_strategy: {init_colors_strategy}")
        self.colors = nn.Parameter(colors_init) # num_points, 3

        # precompute pixel coords
        y_coords = torch.linspace(0, 1, self.height, device=self.device)
        x_coords = torch.linspace(0, 1, self.width, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.pixel_coords = nn.Buffer(torch.stack([grid_x, grid_y], dim=-1)) # H, W, 2
        self.flat_pixel_coords = nn.Buffer(self.pixel_coords.view(-1, 2)) # height * width, 2

        self.frames = []

        self.add_reference_image('target', self.target_image_float, to_uint8=True)
        self.set_display_best('image reconstructed')

    def get_loss(self):
        """
        Calculates the loss using softmax weighting and generates the current image.
        """
        pixels = self.flat_pixel_coords.unsqueeze(1) # H*W, 1, 2
        points = self.points.unsqueeze(0) # 1, num_points, 2
        dist_sq = torch.sum((pixels - points)**2, dim=2) # H*W, num_points
        weights = F.softmax(-self.softmax_beta * dist_sq, dim=1) # H*W, num_points
        assigned_colors = weights @ self.colors
        rendered_image_float = assigned_colors.view(self.height, self.width, 3).clamp(0, 1)
        loss = self.loss(rendered_image_float, self.target_image_float)

        # penalty for blurry image
        if self.softmax_beta < self.min_softmax_beta:
            loss = loss + (self.min_softmax_beta - self.softmax_beta) ** 2

        # make images
        if self._make_images:
            self.log('image reconstructed', rendered_image_float, log_test=False, to_uint8=True)
            self.log_difference('image difference', rendered_image_float, to_uint8=True)

        return loss
