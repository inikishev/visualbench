import math

import torch
from torch import nn
import torch.nn.functional as F

from ...utils import to_CHW, normalize
from ...benchmark import Benchmark
from ...models.basic import MLP
from ...utils.padding import pad_to_shape

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

        loss = self.criterion(preds, targets)

        with torch.no_grad():
            if self._make_images:
                if self.expand != 0: full_preds = full_preds.view(self.expanded_shape)
                else: full_preds = full_preds.view(self.shape)
                self.log_image('prediction', full_preds, to_uint8=True, min=self.min, max=self.max, show_best=True)

        return loss