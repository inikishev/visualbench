import math
from collections.abc import Sequence
from importlib.util import find_spec
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from ...benchmark import Benchmark
from ...models.basic import MLP
from ...utils import normalize, to_CHW, maybe_per_sample_loss
from ...utils.padding import pad_to_shape


def _raise_torchvision(*args, **kwargs):
    raise ModuleNotFoundError("torchvision is required for linear layer visualization")

if find_spec("torchvision") is not None:
    from torchvision.utils import make_grid
else:
    make_grid = _raise_torchvision

class NeuralDrawer(Benchmark):
    """inputs - 2, output - n_channels"""
    def __init__(self, image, model, batch_size: int | None = None, criterion = F.mse_loss, expand: int = 0):
        super().__init__()
        self.image = nn.Buffer(to_CHW(image))
        self.targets = nn.Buffer(self.image.flatten(1, -1).T) # (pixels, channels)
        self.shape = [self.image.shape[1], self.image.shape[2], self.image.shape[0]]

        x = torch.arange(self.image.size(1))
        y = torch.arange(self.image.size(2))
        X, Y = torch.meshgrid(x,y, indexing='xy')
        self.coords = nn.Buffer(torch.stack([X, Y], -1).flatten(0, -2).to(self.image)) # (pixels, 2)

        self.min = self.image.min()
        self.max = self.image.max()

        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size

        self.expand = expand
        self.expanded_shape = [self.image.shape[1] + expand*2, self.image.shape[2] + expand*2, self.image.shape[0]]
        x = torch.arange(-expand, self.image.size(1)+expand)
        y = torch.arange(-expand, self.image.size(2)+expand)
        X, Y = torch.meshgrid(x,y, indexing='xy')
        self.expanded_coords = nn.Buffer(torch.stack([X, Y], -1).flatten(0, -2).to(self.image)) # (pixels, 2)

        mask1 = (self.expanded_coords >= 0).all(-1)
        mask2 = self.expanded_coords[:, 0] < self.image.size(1)
        mask3 = self.expanded_coords[:, 1] < self.image.size(2)
        self.loss_mask = nn.Buffer(mask1 & mask2 & mask3)

        self.add_reference_image("target", self.image, to_uint8=True)
        self._show_titles_on_video = False

        self.set_multiobjective_func(torch.mean)

    def get_loss(self):

        with torch.no_grad():
            mask = None
            idxs = None
            if self._make_images:
                if self.expand != 0:
                    inputs = self.expanded_coords
                    targets = self.targets
                    mask = self.loss_mask

                else:
                    inputs = self.coords
                    targets = self.targets

                if self.batch_size is not None:
                    idxs = torch.randperm(self.targets.size(0))[:self.batch_size]
                    targets = self.targets[idxs]

            else:
                batch_idxs = torch.randperm(self.targets.size(0))[:self.batch_size]
                inputs = self.coords[batch_idxs]
                targets = self.targets[batch_idxs]
                mask = None


        full_preds: torch.Tensor = self.model(inputs) # (pixels, channels)
        if mask is None: preds = full_preds
        else: preds = full_preds[mask]
        if idxs is not None: preds = preds[idxs]

        loss = maybe_per_sample_loss(self.criterion, (preds, targets), per_sample=self._multiobjective)

        with torch.no_grad():
            if self._make_images:
                if self.expand != 0: full_preds = full_preds.view(self.expanded_shape)
                else: full_preds = full_preds.view(self.shape)
                self.log_image('prediction', full_preds, to_uint8=True, min=self.min, max=self.max, show_best=True)

        return loss

@torch.no_grad
def _features_to_grid(features:torch.Tensor, bw_shape):
    stacked = features.moveaxis(-1,0).view(features.size(-1), *bw_shape).unsqueeze(1) # (n_features, 1, H, W)
    return make_grid(stacked, nrow=math.ceil(math.sqrt(stacked.size(0))), padding=1, normalize=True, scale_each=True, pad_value=features.mean().item())

class LayerwiseNeuralDrawer(Benchmark):
    """inputs - 2, output - n_channels"""
    def __init__(self, image, layers=(12,12,12,12,12,12,12), act_cls = nn.LeakyReLU, bn:bool=True, batch_size: int | None = None, criterion = F.mse_loss, expand: int = 0):
        super().__init__()
        self.image = nn.Buffer(to_CHW(image))
        self.targets = nn.Buffer(self.image.flatten(1, -1).T) # (pixels, channels)
        self.shape = [self.image.shape[1], self.image.shape[2], self.image.shape[0]]

        x = torch.arange(self.image.size(1))
        y = torch.arange(self.image.size(2))
        X, Y = torch.meshgrid(x,y, indexing='xy')
        self.coords = nn.Buffer(torch.stack([X, Y], -1).flatten(0, -2).to(self.image)) # (pixels, 2)

        self.min = self.image.min()
        self.max = self.image.max()

        layers = [2] + list(layers)

        BatchNorm = nn.BatchNorm1d if bn else nn.Identity
        modules: list[nn.Module] = [
            nn.Sequential(nn.Linear(i,o), act_cls(), BatchNorm(o)) for i,o in zip(layers[:-1], layers[1:])
        ]
        modules.append(nn.Sequential(nn.Linear(layers[-1], self.image.size(0))))
        self.model = nn.ModuleList(modules)

        self.criterion = criterion
        self.batch_size = batch_size

        self.expand = expand
        self.expanded_shape = [self.image.shape[1] + expand*2, self.image.shape[2] + expand*2, self.image.shape[0]]
        x = torch.arange(-expand, self.image.size(1)+expand)
        y = torch.arange(-expand, self.image.size(2)+expand)
        X, Y = torch.meshgrid(x,y, indexing='xy')
        self.expanded_coords = nn.Buffer(torch.stack([X, Y], -1).flatten(0, -2).to(self.image)) # (pixels, 2)

        mask1 = (self.expanded_coords >= 0).all(-1)
        mask2 = self.expanded_coords[:, 0] < self.image.size(1)
        mask3 = self.expanded_coords[:, 1] < self.image.size(2)
        self.loss_mask = nn.Buffer(mask1 & mask2 & mask3)

        self.add_reference_image("target", self.image, to_uint8=True)
        self._show_titles_on_video = False

    def get_loss(self):

        with torch.no_grad():
            mask = None
            idxs = None
            if self._make_images:
                if self.expand != 0:
                    inputs = self.expanded_coords
                    targets = self.targets
                    mask = self.loss_mask

                else:
                    inputs = self.coords
                    targets = self.targets

                if self.batch_size is not None:
                    idxs = torch.randperm(self.targets.size(0))[:self.batch_size]
                    targets = self.targets[idxs]

            else:
                batch_idxs = torch.randperm(self.targets.size(0))[:self.batch_size]
                inputs = self.coords[batch_idxs]
                targets = self.targets[batch_idxs]
                mask = None

        # inputs is (n_pixels, n_features)
        shape = self.expanded_shape if self.expand != 0 else self.shape # (H, W, C)
        bw_shape = shape[:-1] # (H, W)
        images = []
        for layer in self.model:
            layer = cast(nn.Sequential, layer)

            # pass through linear
            inputs: torch.Tensor = layer[0](inputs)
            if self._make_images:
                images.append(_features_to_grid(inputs, bw_shape))

            if len(layer) > 1:
                # pass through activation and batchnorm
                inputs: torch.Tensor = layer[2](layer[1](inputs))
                if self._make_images:
                    images.append(_features_to_grid(inputs, bw_shape))

        full_preds: torch.Tensor = inputs # (n_pixels, n_channels)
        if mask is None: preds = full_preds
        else: preds = full_preds[mask]
        if idxs is not None: preds = preds[idxs]

        loss = self.criterion(preds, targets)

        with torch.no_grad():
            if self._make_images:
                cur_layer = 1
                is_act = False
                for grid in images:
                    if is_act: name = f"{cur_layer} - activation"
                    else: name = f"{cur_layer}"
                    self.log_image(name, grid, to_uint8=True)

                    if is_act: cur_layer += 1
                    is_act = not is_act

        return loss