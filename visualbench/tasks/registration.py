import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..benchmark import Benchmark
from ..utils import format


def _affine_move(fixed: torch.Tensor, moving: torch.Tensor, affine: torch.Tensor, mode, padding_mode, is_2d:bool):
    grid = F.affine_grid(affine.unsqueeze(0), [1, *fixed.shape], align_corners=True) # (N, H, W, 2)

    moving = moving.unsqueeze(0)
    if not is_2d: moving = moving.unsqueeze(0) # (D, H, W) to (N, C, D, H, W)

    if moving.size(0) > 1:
        grid = grid.repeat_interleave(moving.size(0), 0)

    return F.grid_sample(moving, grid, mode=mode, padding_mode=padding_mode, align_corners=True)[0]


def _initialize(fixed, moving, generator, mode, padding_mode):

    # source and target are allowed to be 2D or 3D, if first or last dim is 4- assumed to be 2D
    fixed = format.totensor(fixed).squeeze()

    random_affine = False
    if moving is None:
        random_affine = True
        moving = fixed

    moving = format.totensor(moving).squeeze()

    if moving.ndim == 3 and fixed.ndim == 3:
        moving_2d = moving.size(0) <= 3 or moving.size(-1) <= 3
        fixed_2d = fixed.size(0) <= 3 or fixed.size(-1) <= 3

    else:
        assert moving.ndim == 2, moving.shape
        assert fixed.ndim == 2, fixed.shape
        moving_2d = fixed_2d = True

    is_2d = moving_2d or fixed_2d
    if is_2d:
        moving = format.to_CHW(moving)
        fixed = format.to_CHW(fixed)

    if is_2d:
        affine = torch.eye(3)[:2]

    else:
        affine = torch.eye(4)[:3]

    if random_affine:
        with torch.no_grad():
            # rotate = torch.tensor([[-0.87, 0.48, 0.5], [-0.48, -0.87, 0.5]])
            # rotate = torch.tensor([[1.5, 0, 0.25], [0, 1.5, 0.25]])
            # noise = torch.randn(affine.size(), dtype=affine.dtype, generator=generator) * 0.5
            mat = torch.tensor([[ 2.2705, -0.1467, -0.8394], [ 0.2842,  0.9577, -0.4493]])
            moving = _affine_move(fixed=fixed, moving=moving, affine=mat,
                                mode=mode, padding_mode=padding_mode, is_2d=is_2d)

    return affine, fixed, moving, is_2d



class AffineRegistration(Benchmark):
    """Affine registration, the goal is to find affine transform that alligns ``moving`` to ``fixed``.

    Args:
        fixed: fixed image, the target to align moving image to. Supports 2D and 3D (volumetric) images.
        moving: moving image, if None, uses rotated fixed image. Supports 2D and 3D (volumetric) images. Defaults to None.
        criterion: criterion. Defaults to F.mse_loss.
        mode: interpolation. Defaults to 'bilinear'.
        padding_mode: padding. Defaults to 'border'.
    """
    def __init__(self, fixed, moving=None, criterion=F.mse_loss, mode='bilinear', padding_mode='border'):
        super().__init__()

        affine, fixed, moving, self.is_2d = _initialize(fixed, moving, self.rng.torch(), mode, padding_mode)

        self.affine = nn.Parameter(affine)
        self.fixed = nn.Buffer(fixed)
        self.moving = nn.Buffer(moving)

        if self.is_2d:
            self.add_reference_image('fixed', fixed, to_uint8=True)

        self.moving = nn.Buffer(moving)
        self.fixed = nn.Buffer(fixed)
        self.criterion = criterion
        self.mode = mode
        self.padding_mode = padding_mode
        self._show_titles_on_video = False

    def get_loss(self):
        moved = _affine_move(self.fixed, self.moving, self.affine, self.mode, self.padding_mode, is_2d=self.is_2d)

        loss = self.criterion(moved, self.fixed)

        if self._make_images:
            with torch.no_grad():
                if self.is_2d:
                    self.log_image("moving", moved, to_uint8=True)
                    self.log_image('overlay', moved+self.fixed, to_uint8=True)

                else:
                    d, h, w = moved.size()
                    self.log_image("moving D", moved[d//2], to_uint8=True)
                    self.log_image("moving H", moved[:, h//2], to_uint8=True)
                    self.log_image("moving W", moved[:, :, w//2], to_uint8=True)

                    self.log_image('overlay D', moved[d//2]+self.fixed[d//2], to_uint8=True)
                    self.log_image('overlay H', moved[:, h//2]+self.fixed[:, h//2], to_uint8=True)
                    self.log_image('overlay W', moved[:, :, w//2]+self.fixed[:, :, w//2], to_uint8=True)

        return loss



class DeformableRegistration(Benchmark):
    """Deformable registration, the goal is to find a deformation field that alligns ``moving`` to ``fixed``.

    Args:
        fixed: fixed image, the target to align moving image to. Supports 2D and 3D (volumetric) images.
        moving: moving image, if None, uses rotated fixed image. Supports 2D and 3D (volumetric) images. Defaults to None.
        grid_size: size of the deformation grid. Defaults to (10,10).
        criterion: criterion. Defaults to F.mse_loss.
        mode: interpolation. Defaults to 'bilinear'.
        padding_mode: padding. Defaults to 'border'.
    """
    def __init__(self, fixed, moving=None, grid_size: int | Sequence[int]=(10,10), criterion=F.mse_loss, mode='bilinear', padding_mode='border'):
        super().__init__()
        affine, fixed, moving, self.is_2d = _initialize(fixed, moving, self.rng.torch(), mode, padding_mode)

        if self.is_2d:
            if isinstance(grid_size, int): grid_size = (grid_size, grid_size)
            x = torch.linspace(-1, 1, grid_size[0])
            y = torch.linspace(-1, 1, grid_size[1])
            grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), 0).unsqueeze(1)

            x = torch.linspace(-1, 1, fixed.size(1))
            y = torch.linspace(-1, 1, fixed.size(2))
            sampler_grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1).unsqueeze(0).repeat_interleave(2, 0)

        else:
            if isinstance(grid_size, int): grid_size = (grid_size, grid_size, grid_size)
            x = torch.linspace(-1, 1, grid_size[0])
            y = torch.linspace(-1, 1, grid_size[1])
            z = torch.linspace(-1, 1, grid_size[1])
            grid = torch.stack(torch.meshgrid(x, y, z, indexing='xy'), 0).unsqueeze(1)

            x = torch.linspace(-1, 1, fixed.size(1))
            y = torch.linspace(-1, 1, fixed.size(2))
            z = torch.linspace(-1, 1, fixed.size(3))
            sampler_grid = torch.stack(torch.meshgrid(x, y, z, indexing='xy'), -1).unsqueeze(0).repeat_interleave(3, 0)

        self.grid = nn.Parameter(grid)
        self.sampler_grid = nn.Buffer(sampler_grid)
        self.fixed = nn.Buffer(fixed)
        self.moving = nn.Buffer(moving)

        if self.is_2d:
            self.add_reference_image('fixed', fixed, to_uint8=True)

        self.moving = nn.Buffer(moving)
        self.fixed = nn.Buffer(fixed)
        self.criterion = criterion
        self.mode = mode
        self.padding_mode = padding_mode

        self._show_titles_on_video = False

    def get_loss(self):

        grid = F.grid_sample(input=self.grid, grid=self.sampler_grid, mode=self.mode, align_corners=True)

        moving = self.moving.unsqueeze(0)
        if not self.is_2d: moving = moving.unsqueeze(0) # (D, H, W) to (N, C, D, H, W)

        if moving.size(0) > 1:
            grid = grid.repeat_interleave(moving.size(0), 0)

        moved = F.grid_sample(moving, grid.moveaxis(0,-1), mode=self.mode, padding_mode=self.padding_mode, align_corners=True)[0]
        loss = self.criterion(moved, self.fixed)

        if self._make_images:
            with torch.no_grad():
                if self.is_2d:
                    self.log_image("moving", moved, to_uint8=True)
                    self.log_image('overlay', moved+self.fixed, to_uint8=True)

                else:
                    d, h, w = moved.size()
                    self.log_image("moving D", moved[d//2], to_uint8=True)
                    self.log_image("moving H", moved[:, h//2], to_uint8=True)
                    self.log_image("moving W", moved[:, :, w//2], to_uint8=True)

                    self.log_image('overlay D', moved[d//2]+self.fixed[d//2], to_uint8=True)
                    self.log_image('overlay H', moved[:, h//2]+self.fixed[:, h//2], to_uint8=True)
                    self.log_image('overlay W', moved[:, :, w//2]+self.fixed[:, :, w//2], to_uint8=True)

        return loss
