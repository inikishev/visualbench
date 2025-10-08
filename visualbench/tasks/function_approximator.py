from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import visualbench as vb
from visualbench.benchmark import Benchmark
from visualbench.utils import totensor, maybe_per_sample_loss


@torch.no_grad
def _plot_vecs(
    vectors: list[torch.Tensor],
    ymin,
    ymax,
    width: int = 512,
    height: int = 256,
    line_thickness: int = 1,
) -> np.ndarray:
    if not vectors:
        return np.zeros((height, width, 3), dtype=np.uint8)
    try:
        all_vectors = torch.stack(vectors)
    except Exception as e:
        raise ValueError("All tensors in the list must have the same length.") from e

    num_vectors, num_points = all_vectors.shape
    device = all_vectors.device

    normalized_vectors = ((all_vectors - ymin) / (ymax - ymin)).clip(min=ymin, max=ymax)

    x_coords = torch.linspace(0, width - 1, steps=num_points, device=device)
    y_coords = (height - 1) * (1 - normalized_vectors)
    points_tensor = torch.stack((x_coords.expand_as(y_coords), y_coords), dim=-1)

    points_np = points_tensor.numpy(force=True).astype(np.int32)

    # ----------------------------------- FRAME ---------------------------------- #
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(num_vectors):
        hue = int((i * 180 / num_vectors) % 180)
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0] # pylint:disable=no-member
        color_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        pts = points_np[i]
        if i == num_vectors - 2: line_thickness = 2
        cv2.polylines(frame, [pts], isClosed=False, color=color_tuple, thickness=line_thickness, lineType=cv2.LINE_AA)# pylint:disable=no-member

    return frame

class FunctionApproximator(Benchmark):
    """Approximate ``target`` with a deep neural network with skip connections, where each layer outputs 1 value.

    Each layer is a linear layer which takes in outputs of ``n`` previous layers, where ``n`` is 0 to ``n_skip`` depending on position of the layer, and outputs a single value.

    Renders:
        output of each layer.

    Inspired by https://www.youtube.com/watch?v=3BwRVpciD3k

    """
    def __init__(self, target: Any, depth:int=7, n_skip:int=4, act=F.tanh, batch_size=None, criterion=F.mse_loss, resolution=(256,512)):
        super().__init__()

        self.target = nn.Buffer(totensor(target, dtype=torch.float32))
        self.input = nn.Buffer(torch.linspace(-1, 1, len(self.target), dtype=torch.float32))
        self.min = self.target.amin().item()
        self.max=self.target.amax().item()
        d = self.max-self.min
        self.min -= d
        self.max += d

        self.depth = depth
        self.criterion = criterion
        self.act = act
        self.n_skip = n_skip
        self.batch_size = batch_size

        self.layers = nn.ModuleList(nn.Linear(min(i+1, max(n_skip, 1)), 1) for i in range(depth))
        with torch.no_grad():
            for l in self.layers:
                torch.nn.init.orthogonal_(l.weight, generator=self.rng.torch())
                l.bias.zero_() # type:ignore

        self.resolution = resolution
        self._show_titles_on_video = False
        self.set_multiobjective_func(torch.mean)

    @staticmethod
    def SINE(periods=4, n=1024):
        return torch.sin(torch.linspace(0, torch.pi*periods*2, n))

    def get_loss(self):
        x = self.input

        xs: list[torch.Tensor] = [x]

        for i, layer in enumerate(self.layers):
            if len(xs) > 0 and self.n_skip > 0:
                x = torch.stack(xs[-self.n_skip:], -1)

            x = layer(x.unsqueeze(0)).squeeze()
            if i != self.depth - 1:
                x = self.act(x)

            xs.append(x)

        if self.batch_size is None:
            loss = maybe_per_sample_loss(self.criterion, (x, self.target), per_sample=self._multiobjective)
        else:
            idxs = torch.randperm(self.target.size(0))[:self.batch_size]
            loss = maybe_per_sample_loss(self.criterion, (x[idxs], self.target[idxs]), per_sample=self._multiobjective)
            with torch.no_grad():
                self.log('test loss', self.criterion(x, self.target))

        if self._make_images:
            with torch.no_grad():
                xs.append(self.target)
                frame = _plot_vecs(xs, self.min, self.max, self.resolution[1], self.resolution[0])
                self.log_image("outputs", frame, to_uint8=False, show_best=True)
        return loss