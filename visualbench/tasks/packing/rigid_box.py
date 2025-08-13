
# pylint:disable=not-callable
import itertools
import random
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ...benchmark import Benchmark
from ...utils import tonumpy, totensor


def _put_alpha(x: np.ndarray, other:np.ndarray, alpha1: float, alpha2: float = 1):
    return x - (x - other)*(alpha1*alpha2)

def _softrect2d_(array: np.ndarray, x1, x2, color, alpha: float, add_fn = _put_alpha) -> None:
    """same as array[x1[0]:x2[0], x1[1]:x2[1]] = color, but with a soft edge

    Args:
        array (np.ndarray): a (H, W, 3) array
        x1 (_type_): coord of first point.
        x2 (_type_): coord of second point.
        color (_type_): color - 3 floats
        alpha (float): alpha
        add_fn (_type_, optional): function that adds. Defaults to put_alpha.
    """
    x1 = np.clip(tonumpy(x1), 0, array.shape[:-1])
    x1low = np.floor(x1,).astype(int)
    x1high = np.ceil(x1,).astype(int)
    x1dist_from_low = x1 - x1low

    x2 = np.clip(tonumpy(x2), 0, array.shape[:-1])
    x2low = np.floor(x2,).astype(int)
    x2high = np.ceil(x2,).astype(int)
    x2dist_from_low = x2 - x2low

    color = tonumpy(color)

    if x1dist_from_low[0] > 0:
        array[x1low[0], x1high[1]:x2low[1]] = add_fn(array[x1low[0], x1high[1]:x2low[1]], color, 1-x1dist_from_low[0], alpha)
    if x1dist_from_low[1] > 0:
        array[x1high[0]:x2low[0], x1low[1]] = add_fn(array[x1high[0]:x2low[0], x1low[1]], color, 1-x1dist_from_low[1], alpha)
    if x2dist_from_low[0] > 0:
        array[x2high[0]-1, x1high[1]:x2low[1]] = add_fn(array[x2high[0]-1, x1high[1]:x2low[1]], color, x2dist_from_low[0], alpha)
    if x2dist_from_low[1] > 0:
        array[x1high[0]:x2low[0], x2high[1]-1] = add_fn(array[x1high[0]:x2low[0], x2high[1]-1], color, x2dist_from_low[1], alpha)

    # fill main rectangle
    array[x1high[0]:x2low[0], x1high[1]:x2low[1]] = add_fn(array[x1high[0]:x2low[0], x1high[1]:x2low[1]], color, alpha, 1)

CONTAINER1 = (10,10), ((1,10), (1,10), (1,9), (1,9), (3,3), (2,4), (4,1), (3,4), (3,2), (2,2), (1,1), (1,1), (6,1), (8,1))

def uniform_container(box_size:tuple[float,float], num_boxes:tuple[int,int]):
    """makes a container filled with same sized boxes.

    Args:
        box_size (tuple[float,float]): _description_
        num_boxes (tuple[int,int]): _description_

    Returns:
        _type_: _description_
    """
    container_size = (box_size[0] * num_boxes[0], box_size[1] * num_boxes[1])
    boxes = [box_size for _ in range(num_boxes[0] * num_boxes[1])]
    return container_size, boxes

def _make_colors(n,seed):
    # generate colors for boxes
    colors = []
    i = 2
    while len(colors) < i:
        colors = list(itertools.product(np.linspace(0, 255, n).tolist(), repeat=3)) # type:ignore
        if (255., 255., 255.) in colors: colors.remove((255., 255., 255.))
        i+=1

    rng = random.Random(seed)
    rgbs = rng.sample(colors, k = n)

    # remove almost white colors
    for i in range(len(rgbs)): # pylint:disable=consider-using-enumerate
        while sum(rgbs[i]) > 600: # type:ignore
            rgbs[i] = rng.sample(colors, k = 1)[0]

    return rgbs

class RigidBoxPacking(Benchmark):
    """Box packing without rotation benchmark, can be rendered as a video.

    If an optimizer accepts bounds, pass (0, 1) for all parameters.

    Args:
        container_size (_type_, optional): tuple of two numbers - size of the container to fit boxes into. Defaults to CONTAINER1[0].
        box_sizes (_type_, optional): list of tuples of two numbers per box - its x and y size. Defaults to CONTAINER1[1].
        npixels (float | None, optional):
            Number of pixels in the video (product of width and height). Defaults to 100_000.
            Aspect ratio is determined by `container_size`.
        square (bool, optional): if True, overlap in loss function will be squared. Defaults to False.
        penalty (float, optional):
            multiplier to absolute penalty for going outside the edges. Defaults to 0.5.
        sq_penalty (float, optional):
            multiplier to squared penalty for going outside the edges. Defaults to 20.
        init (str, optional):
            initially put boxes in the center, in the corner, or randomly.
            'random' init is seeded and is always the same. Other inits
            also add a very small amount of seeded noise to ensure no two boxes
            spawn in exactly the same place, which would make their gradients
            identical so they will never separate. Defaults to 'random'.
        colors_seed (int, optional): seed for box colors. Defaults to 2.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cpu'.
    """
    def __init__(
        self,
        container_size = CONTAINER1[0],
        box_sizes = CONTAINER1[1],
        npixels: float | None = 100_000,
        square: bool = False,
        penalty: float = 0.5,
        sq_penalty: float = 20,
        init: Literal['center', 'corner', 'random', 'top'] = 'top',
        colors_seed: int | None = 13,
        dtype = torch.float32,
        seed = 0,
    ):
        super().__init__(bounds=(0, 1), seed = seed)
        if npixels is not None: scale = (npixels / np.prod(container_size)) ** (1/2)
        else: scale = 1
        self.scale = scale
        self.container_size_np = (np.array(container_size, dtype = float) * scale).astype(int)
        size = torch.prod(torch.tensor(self.container_size_np, dtype = dtype))
        self.size = nn.Buffer(size)
        container_size = torch.from_numpy(self.container_size_np).to(dtype=dtype)
        self.container_size = nn.Buffer(container_size)
        box_sizes = totensor(box_sizes, dtype = dtype) * scale
        self.box_sizes = nn.Buffer(box_sizes)
        self.box_sizes_np = self.box_sizes.detach().cpu().numpy()
        self.square = square

        self.penalty = penalty
        self.sq_penalty = sq_penalty

        # generate colors for boxes
        self.colors = _make_colors(len(box_sizes), colors_seed)

        # slightly randomize params so that no params overlap which gives them exactly the same gradients
        # so they never detach from each other
        normalized_box_sizes = self.box_sizes / self.container_size.unsqueeze(0) # 0 to 1
        noise = torch.randn((len(box_sizes), 2), dtype = dtype, generator=self.rng.torch())

        if init == 'center':
            self.params = nn.Parameter((1 - normalized_box_sizes) * 0.5 + noise.mul(0.01), requires_grad=True)
        elif init == 'corner':
            self.params = nn.Parameter(noise.uniform_(0, 0.01), requires_grad=True)
        elif init == 'top':
            p = (1 - normalized_box_sizes) * 0.5 + noise.mul(0.01)
            p[:, 0] = noise.uniform_(0, 0.01)[:,0]
            self.params = nn.Parameter(p, requires_grad=True)
        elif init == 'random':
            self.params = nn.Parameter((1 - normalized_box_sizes) * noise.uniform_(0, 1))

        self.set_multiobjective_func(torch.mean)


    @torch.no_grad
    def _make_frame(self):
        # arr = paramvec.detach().cpu().numpy().reshape(self.params.shape)
        arr = self.params.detach().cpu().numpy()
        container = np.full((*self.container_size_np, 3), 255)
        for (y,x), box, c in zip(arr, self.box_sizes_np, self.colors):
            y *= self.container_size_np[0]
            y *= (self.container_size_np[0] - box[0])/self.container_size_np[0]
            x *= self.container_size_np[1]
            x *= (self.container_size_np[1] - box[1])/self.container_size_np[1]
            # if y+box[0] >= self.container_size_np[0]: y = self.container_size_np[0] - box[0]
            # if x+box[1] >= self.container_size_np[1]: x = self.container_size_np[1] - box[1]
            try: _softrect2d_(container, (y,x), (y+box[0], x+box[1]), c, 0.5,)
            except IndexError: pass
        return np.clip(container, 0, 255).astype(np.uint8)

    def get_loss(self):
        # we still need penalty as if box is entirely outside, gradient will be 0
        overflows = [self.params[self.params>1] - 1, self.params[self.params < 0]]
        overflows = [i for i in overflows if i.numel() > 0]
        if len(overflows) > 0:
            penalty = torch.stack([i.abs().mean() for i in overflows]).sum() * self.penalty
            penalty = penalty + torch.stack([i.pow(2).mean() for i in overflows]).sum() * self.sq_penalty
            if not torch.isfinite(penalty): penalty = torch.tensor(torch.inf, device = self.params.device)
        else: penalty = torch.tensor(0, device = self.params.device)

        # create boxes from parameters
        params = self.params
        boxes = torch.zeros(len(self.box_sizes)+4, 4, device = self.params.device)
        for i, ((y,x), box) in enumerate(zip(params, self.box_sizes)):
            y = y * self.container_size[0]
            y = y * (self.container_size[0] - box[0])/self.container_size[0]
            x = x * self.container_size[1]
            x = x * (self.container_size[1] - box[1])/self.container_size[1]

            boxes[i, 0] = y; boxes[i, 1] = y+box[0]; boxes[i, 2] = x; boxes[i, 3] = x+box[1]

        # edge boxes
        for i, edge in enumerate([
            (-1e10, 0, -1e10, 0),
            (-1e10, 0, self.container_size[1], 1e10),
            (self.container_size[0], 1e10, -1e10, 0),
            (self.container_size[0], 1e10, self.container_size[1], 1e10),
        ]):
            ip = i+1
            boxes[-ip, 0] = edge[0]; boxes[-ip, 1] = edge[1]; boxes[-ip, 2] = edge[2]; boxes[-ip, 3] = edge[3]

        # this calculates total overlap between every pair of boxes
        # but in a vectorized way
        ya1, yb1 = torch.meshgrid(boxes[:, 0], boxes[:, 0], indexing = 'ij')
        ya2, yb2 = torch.meshgrid(boxes[:, 1], boxes[:, 1], indexing = 'ij')
        xa1, xb1 = torch.meshgrid(boxes[:, 2], boxes[:, 2], indexing = 'ij')
        xa2, xb2 = torch.meshgrid(boxes[:, 3], boxes[:, 3], indexing = 'ij')

        x_overlap = torch.clamp(torch.minimum(xa2, xb2) - torch.maximum(xa1, xb1), min=0)

        # mask diagonal elements (ovelap with itself), and last four boxes as those are to avoid overflow overedges
        mask = torch.eye(len(boxes), dtype = torch.bool, device = self.params.device).logical_not_()
        mask[-4:] = False
        y_overlap = torch.clamp(torch.minimum(ya2, yb2) - torch.maximum(ya1, yb1), min=0) * mask

        overlap = x_overlap * y_overlap
        if self.square: overlap = overlap ** 2

        loss = overlap#.sum() / self.size
        penalized_loss = loss + penalty

        # code above is equivalent to commented out code below (which was very slow):
        # for i, (ya1, ya2, xa1, xa2) in enumerate(boxes):
        #     for j, (yb1, yb2, xb1, xb2) in enumerate(boxes[:-4]): # skip last 4 edge boxes
        #         if i != j:
        #             x_overlap = max(min(xa2, xb2) - max(xa1, xb1), 0)
        #             y_overlap = max(min(ya2, yb2) - max(ya1, yb1), 0)
        #             overlap = x_overlap * y_overlap
        #             if self.square: overlap = overlap ** 2
        #             loss = loss + overlap / self.size

        if self._make_images:
            self.log_image("boxes", self._make_frame(), to_uint8=False, show_best=True)

        return penalized_loss.ravel()

