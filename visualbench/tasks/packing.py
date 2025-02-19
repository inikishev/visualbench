# pylint:disable=not-callable
import itertools
import math
import random
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from myai.transforms import tonumpy, totensor

from ..benchmark import Benchmark


def put_alpha(x: np.ndarray, other:np.ndarray, alpha1: float, alpha2: float = 1):
    return x - (x - other)*(alpha1*alpha2)

def softrect2d_(array: np.ndarray, x1, x2, color, alpha: float, add_fn = put_alpha) -> None:
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

class BoxPacking(Benchmark):
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
    size: torch.nn.Buffer
    container_size: torch.nn.Buffer
    box_sizes: torch.nn.Buffer
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
        make_images = True,
        seed = 0,
    ):
        super().__init__(log_projections = True, seed = seed)
        if npixels is not None: scale = (npixels / np.prod(container_size)) ** (1/2)
        else: scale = 1
        self.scale = scale
        self.container_size_np = (np.array(container_size, dtype = float) * scale).astype(int)
        size = torch.prod(torch.tensor(self.container_size_np, dtype = dtype))
        self.register_buffer('size', size)
        container_size = torch.from_numpy(self.container_size_np).to(dtype=dtype)
        self.register_buffer('container_size', container_size)
        box_sizes = totensor(box_sizes, dtype = dtype) * scale
        self.register_buffer('box_sizes', box_sizes)
        self.box_sizes_np = self.box_sizes.detach().cpu().numpy()
        self.square = square

        self.penalty = penalty
        self.sq_penalty = sq_penalty

        # generate colors for boxes
        self.colors = _make_colors(len(box_sizes), colors_seed)
        self._make_images = make_images

        # slightly randomize params so that no params overlap which gives them exactly the same gradients
        # so they never detach from each other
        normalized_box_sizes = self.box_sizes / self.container_size.unsqueeze(0) # 0 to 1
        noise = torch.randn((len(box_sizes), 2), dtype = dtype, generator=self.rng.torch())

        if init == 'center':
            self.params = torch.nn.Parameter((1 - normalized_box_sizes) * 0.5 + noise.mul(0.01), requires_grad=True)
        elif init == 'corner':
            self.params = torch.nn.Parameter(noise.uniform_(0, 0.01), requires_grad=True)
        elif init == 'top':
            p = (1 - normalized_box_sizes) * 0.5 + noise.mul(0.01)
            p[:, 0] = noise.uniform_(0, 0.01)[:,0]
            self.params = torch.nn.Parameter(p, requires_grad=True)
        elif init == 'random':
            self.params = torch.nn.Parameter((1 - normalized_box_sizes) * noise.uniform_(0, 1))

        self.set_display_best('image')

    @torch.no_grad
    def _make_solution_image(self):
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
            try: softrect2d_(container, (y,x), (y+box[0], x+box[1]), c, 0.5,)
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

        loss = overlap.sum() / self.size
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

        if self._make_images: self.log("image", self._make_solution_image(), False, to_uint8=False)
        return penalized_loss


TEN_BOX_PROBLEM = (3.5,3.5), (((1,1), )*10)
class RotatingBoxPacking(Benchmark):
    """Box packing with rotation.

    Args:
        container_size (_type_, optional): tuple of two numbers - size of the container to fit boxes into. Defaults to CONTAINER1[0].
        box_sizes (_type_, optional): list of tuples of two numbers per box - its x and y size. Defaults to CONTAINER1[1].
        image_size (int, optional): size of image for rendering. Defaults to 256.
        wall_penalty (float, optional): penalty for overlapping with the container. Defaults to 10.0.
        device (str, optional): device. Defaults to 'cuda'.
    """
    ws: torch.nn.Buffer
    hs: torch.nn.Buffer
    colors: torch.nn.Buffer
    def __init__(self, container_size = TEN_BOX_PROBLEM[0], box_sizes = TEN_BOX_PROBLEM[1], image_size=256, wall_penalty=2.0, make_images=True):
        super().__init__(log_projections = True, seed = 0)
        self.register_buffer('ws', torch.tensor([w for w, h in box_sizes], dtype=torch.float32))
        self.register_buffer('hs', torch.tensor([h for w, h in box_sizes], dtype=torch.float32))
        self.n_rects = len(box_sizes)
        self.container_w, self.container_h = container_size
        self.image_size = image_size
        self.wall_penalty = wall_penalty

        # intiialize in centers
        x = (torch.rand(self.n_rects, generator = self.rng.torch())*0.5) * container_size[0]
        y = (torch.rand(self.n_rects, generator = self.rng.torch())*0.5) * container_size[1]
        theta = torch.randn(self.n_rects, generator = self.rng.torch()) * 0.3  # Small random initial rotations
        self.positions = torch.nn.Parameter(torch.stack([x, y, theta], dim=1))

        self.register_buffer('colors', torch.tensor(_make_colors(len(box_sizes), 0), dtype=torch.float32) / 255)
        self._make_images = make_images
        self.set_display_best('image')

    def get_loss(self):
        x, y, theta = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]

        # Smooth absolute value approximation for gradient flow
        eps = 1e-8
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        abs_cos = torch.sqrt(cos_t**2 + eps)
        abs_sin = torch.sqrt(sin_t**2 + eps)

        # Rotated dimensions with differentiable approximation
        w_rot = self.ws * abs_cos + self.hs * abs_sin
        h_rot = self.ws * abs_sin + self.hs * abs_cos

        # Bounding box coordinates
        min_x, max_x = x - w_rot/2, x + w_rot/2
        min_y, max_y = y - h_rot/2, y + h_rot/2

        # Overlap calculation
        overlap_x = (torch.min(max_x[:, None], max_x) - torch.max(min_x[:, None], min_x)).clamp(min=0)
        overlap_y = (torch.min(max_y[:, None], max_y) - torch.max(min_y[:, None], min_y)).clamp(min=0)
        overlaps = overlap_x * overlap_y
        total_overlap = (overlaps.sum() - torch.diag(overlaps).sum()) / 2

        # Boundary penalties
        boundary_loss = (torch.relu(-min_x) + torch.relu(max_x - self.container_w) +
                         torch.relu(-min_y) + torch.relu(max_y - self.container_h)).sum()

        loss = total_overlap + self.wall_penalty * boundary_loss

        # Visualization
        if self._make_images:
            self.log('image', self._render_frame(x, y, theta).to(torch.uint8), False, to_uint8=False)

        return loss

    @torch.no_grad
    def _render_frame(self, x, y, theta):
        H = W = self.image_size
        img = torch.ones((H, W, 3), device=x.device)
        xx = torch.linspace(0, self.container_w, W, device=x.device)
        yy = torch.linspace(0, self.container_h, H, device=x.device)
        grid_x, grid_y = torch.meshgrid(xx, yy, indexing='xy')

        for i in range(self.n_rects):
            # Rotated rectangle mask
            dx = grid_x - x[i]
            dy = grid_y - y[i]
            cos_t, sin_t = torch.cos(theta[i]), torch.sin(theta[i])
            rot_x = dx * cos_t - dy * sin_t
            rot_y = dx * sin_t + dy * cos_t

            mask = (rot_x.abs() <= self.ws[i]/2) & (rot_y.abs() <= self.hs[i]/2)
            color = self.colors[i % len(self.colors)]
            img[mask] = img[mask] * 0.7 + color * 0.3  # More visible overlaps

        # Add container borders
        img[0,:] = img[-1,:] = img[:,0] = img[:,-1] = 1.0
        return (img * 255).permute(1, 0, 2).clamp(0, 255)



RADII1 = tuple([1,2,3,4,1,2,3,1,2,1]*4 + [10])

class SpherePacking(Benchmark):
    """The goal is to pack 2D spheres as densely as possible.

    The objective is for each sphere to be as close to the origin as possible while penalizing overlaps.

    Args:
        radii (Sequence[float] | np.ndarray | torch.Tensor): list of radii of each sphere

    """
    radii: torch.nn.Buffer
    grid: torch.nn.Buffer
    def __init__(self, radii: Sequence[float] | np.ndarray | torch.Tensor = RADII1, make_images = True):
        super().__init__(log_projections = True)
        self._make_images = make_images
        self.N = len(radii)
        self.register_buffer('radii', totensor(radii, dtype=torch.float32))

        # we arrange spheres in a circle which is quite involved but its good initialization...
        # Calculate sum of consecutive radii (including last to first)
        sum_r = self.radii + torch.roll(self.radii, shifts=-1, dims=0)
        sum_r_max = torch.max(sum_r)

        # Binary search to find optimal circle radius R
        R_low = sum_r_max / 2.0
        R_high = torch.sum(self.radii)  # Initial upper bound

        # Perform binary search
        for _ in range(100):
            R = (R_low + R_high) / 2.0
            theta_total = 2.0 * torch.sum(torch.asin(sum_r / (2.0 * R)))
            if theta_total < 2 * math.pi:
                R_high = R  # Need larger theta_total, decrease R
            else:
                R_low = R

        R = (R_low + R_high) / 2.0

        # Calculate angles between consecutive spheres
        theta_i = 2 * torch.asin(sum_r / (2.0 * R))

        # Compute cumulative angles for sphere positions
        theta_without_last = theta_i[:-1]
        cum_sum = torch.cumsum(theta_without_last, dim=0)
        alpha = torch.zeros(self.N, device=self.radii.device)
        alpha[1:] = cum_sum

        # Initialize positions
        x = R * torch.cos(alpha)
        y = R * torch.sin(alpha)
        self.positions = torch.nn.Parameter(torch.stack([x, y], dim=1))

        self.spread_coeff = 0.01
        self.edge_epsilon = 0.015

        # Visualization setup
        H, W = 256, 256
        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_x, grid_y], -1))  # (H, W, 2)

    def get_loss(self):
        pos, radii = self.positions, self.radii

        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        distances = torch.sqrt((diff**2).sum(-1) + 1e-8)
        i, j = torch.triu_indices(self.N, self.N, 1)
        overlaps = torch.clamp_min(radii[i] + radii[j] - distances[i, j], 0)
        total_loss = (overlaps**2).sum() + self.spread_coeff * (pos**2).sum()

        # Visualization
        if self._make_images:
            with torch.no_grad():
                # Scale to fit spheres in view
                combined = torch.cat([pos - radii.unsqueeze(-1),
                                    pos + radii.unsqueeze(-1)], -1)
                min_val, _ = combined.min(0)[0].min(0)
                max_val, _ = combined.max(0)[0].max(0)
                scale = 2 / (max_val - min_val).max().clamp_min(1e-8)
                offset = (min_val + max_val) / 2

                # Transform positions and radii
                scaled_pos = (pos - offset) * scale
                scaled_radii = radii * scale

                # Calculate distance from grid to all spheres
                scaled_pos_reshaped = scaled_pos.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, N, 2)

                grid_dist = torch.norm(self.grid.unsqueeze(2) - scaled_pos_reshaped, dim=-1)

                # Create masks
                in_sphere = grid_dist <= scaled_radii  # (H, W, N)
                in_edge = (grid_dist >= (scaled_radii - self.edge_epsilon)) & in_sphere

                # Combine across spheres
                any_sphere = in_sphere.any(-1)  # (H, W)
                any_edge = in_edge.any(-1)

                # Create image: white=inside, black=edges, gray=background
                image = torch.full((256, 256), 192, dtype=torch.uint8, device=pos.device)
                image[any_sphere] = 255
                image[any_edge] = 0

                self.log('image', image, False, to_uint8 = False)
        return total_loss