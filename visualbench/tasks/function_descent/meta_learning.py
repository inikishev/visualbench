import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.func
from PIL import Image, ImageDraw
from torch import nn

from ...benchmark import Benchmark
from ...tasks.function_descent.function_descent import _safe_flatten, _UnpackCall
from ...tasks.function_descent.test_functions import TEST_FUNCTIONS, TestFunction
from ...utils.format import tonumpy, totensor
from ...utils.funcplot import funcplot2d


class _PackCall:
    __slots__ = ("f", )
    def __init__(self, f): self.f=f
    def __call__(self, x): return self.f(*x)

class Chainable(nn.Module, ABC):
    @abstractmethod
    def forward(self, x:torch.Tensor, f:torch.Tensor, g:torch.Tensor, state:dict) -> tuple[torch.Tensor, dict]:
        """returns new x"""

class LR(Chainable):
    def __init__(self):
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x, f, g, state):
        return g*self.lr, state

class Decay(Chainable):
    def __init__(self):
        super().__init__()
        self.decay = nn.Parameter(torch.tensor(0.99))

    def forward(self, x, f, g, state):
        lr = state.get('lr', 1)
        g = g*lr
        state['lr'] = lr * self.decay
        return g, state

class HeavyBall(Chainable):
    def __init__(self):
        super().__init__()
        self.momentum = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, f, g, state):
        velocity = state.get('velocity', None)
        if velocity is None: velocity = torch.zeros_like(g)
        velocity = (velocity + g) * self.momentum
        state['velocity'] = velocity
        return velocity, state

class Preconditioner(Chainable):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.eye(2, dtype=torch.float32))

    def forward(self, x, f, g, state):
        return self.P @ g, state

class Adam(Chainable):
    def __init__(self):
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(1e-5))
        self.betas = nn.Parameter(torch.tensor([0.9, 0.999]))
        self.eps = nn.Parameter(torch.tensor(1e-5))

    def forward(self, x, f, g, state):
        exp_avg = state.get('exp_avg', None)
        if exp_avg is None: exp_avg = torch.zeros_like(g)
        exp_avg = exp_avg.lerp(g, 1-self.betas[0])

        exp_avg_sq = state.get('exp_avg_sq', None)
        if exp_avg_sq is None: exp_avg_sq = torch.zeros_like(g)
        exp_avg_sq = exp_avg_sq.lerp(g**2, 1-self.betas[1])

        update = exp_avg / (exp_avg_sq + self.eps.clip(min=1e-10))

        state['exp_avg'] = exp_avg
        state['exp_avg_sq'] = exp_avg_sq

        return update, state

class Schedule(Chainable):
    def __init__(self, n:int):
        super().__init__()
        self.lrs = nn.Parameter(torch.ones(n, dtype=torch.float32))

    def forward(self, x, f, g, state):
        step = state.get('step', 0)
        lr = self.lrs[step]
        state['step'] = step+1
        return g*lr, state

class Normalize(Chainable):
    def __init__(self):
        super().__init__()
    def forward(self, x, f, g, state):
        return g / torch.linalg.vector_norm(g).clip(min=1e-9), state # pylint:disable=not-callable

class Chain(Chainable):
    def __init__(self, *opts: Chainable):
        super().__init__()
        self.opts = nn.ModuleList(opts)

    def forward(self, x, f, g, state):
        states = [state.get(i, {}) for i in range(len(self.opts))]

        for i, (opt, st) in enumerate(zip(self.opts, states)):
            g, st = opt(x, f, g, st)
            state[i] = st

        x = x - g
        return x, state

class MetaLearning(Benchmark):
    LR=LR
    Decay=Decay
    HeavyBall=HeavyBall
    Adam=Adam
    Preconditioner=Preconditioner
    Schedule=Schedule
    Normalize=Normalize
    Chain=Chain

    def __init__(
        self,
        opt: Chain,
        n_steps: int,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        minima = None,
        dtype: torch.dtype = torch.float32,
        unpack=True,
        resolution = 384,
        tune_lr: bool = True,
        log_scale:bool=False,
    ):
        if isinstance(func,str): f = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = dtype)
        else: f = func

        if isinstance(f, TestFunction):
            if x0 is None: x0 = f.x0()
            if domain is None: domain = f.domain()
            if minima is None: minima = f.minima()
            unpack = True

        x0 = totensor(x0, dtype=dtype)
        super().__init__(log_params=True)

        if unpack: f = _PackCall(f)
        self.func: Callable[[torch.Tensor], torch.Tensor] = f
        self.grad_and_value = torch.func.grad_and_value(f)

        if domain is None: raise ValueError("Domain must be provided")
        self._domain = tonumpy(_safe_flatten(domain))

        self.unpack = unpack
        if minima is not None: self.minima = totensor(minima)
        else: self.minima = minima
        self.n_steps = n_steps

        self.x0 = torch.nn.Buffer(x0)
        self.opt = opt

        lr_tensor = None
        for ch in self.opt.opts:
            if isinstance(ch, LR):
                lr_tensor = ch.lr
                break

        # make sure it doesn't nan
        if lr_tensor is not None:
            with torch.no_grad():
                loss = self._get_x_f()[1][-1]
                while math.isnan(loss):
                    lr_tensor.mul_(0.5)
                    loss = self._get_x_f()[1][-1]

                # tune lr
                v_orig = lr_tensor.clone()
                if tune_lr:
                    from scipy.optimize import minimize_scalar
                    def objective(v):
                        lr_tensor.set_(torch.tensor(v, device=lr_tensor.device, dtype=lr_tensor.dtype)) # type:ignore
                        return self._get_x_f()[1][-1].detach().cpu().item()
                    res: Any = minimize_scalar(objective, bracket=(0,lr_tensor.detach().cpu().item()), options=dict(maxiter=10))
                    if math.isfinite(res.fun) and res.fun < loss:
                        lr_tensor.set_(torch.tensor(res.x, device=lr_tensor.device, dtype=lr_tensor.dtype))
                    else:
                        lr_tensor.set_(v_orig)# type:ignore



        # # ------------------------------- visualization ------------------------------ #
        self.resolution = resolution

        xmin, xmax, ymin, ymax = self._domain
        self.vis_domain = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

        # x_coords = torch.linspace(xmin, xmax, self.resolution, dtype=dtype)
        # y_coords = torch.linspace(ymin, ymax, self.resolution, dtype=dtype)
        # grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # grid_points = torch.stack([grid_x, grid_y], dim=-1)
        # flat_points = grid_points.view(-1, 2)

        # with torch.no_grad():
        #     z_values_flat = self.func(flat_points.T) # type:ignore

        # z_values = z_values_flat.view(self.resolution, self.resolution)

        # if log_scale: z_values = torch.log(z_values - z_values.min() + 1e-8)
        # z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min())

        # colormap = plt.get_cmap('viridis')
        # background_rgba = colormap(z_norm.cpu().numpy())
        # background_pixels = (background_rgba[:, :, :3] * 255).astype(np.uint8)

        # self.background_img = Image.fromarray(np.flipud(background_pixels), 'RGB')
        bounds = self._get_domain()
        f = _UnpackCall(self.func)

        # make frame with matplotlib
        fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Use the whole figure  # pyright:ignore[reportCallIssue, reportArgumentType]
        ax.axis('off') # No axes, ticks, or labels

        funcplot2d(
            f, *bounds, num=resolution, cmap='gray',
            levels=20, contour_cmap='binary',
            contour_lw=0.1, contour_alpha=1.0, # Alpha is handled by cv2 later
            log_contour=log_scale, lib='torch', ax=ax
        )
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])

        # render to numpy
        fig.canvas.draw()
        background = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)  # pyright:ignore[reportAttributeAccessIssue]
        background = background.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3].copy()

        # 1st point and minima
        if self.minima is not None:
            minima = self._world_to_pixel(self.minima.numpy(force=True))
            cv2.drawMarker(background, [int(i) for i in minima], (0, 255, 255)) # pylint:disable=no-member

        plt.close(fig)
        self.background = Image.fromarray(background)

    def _world_to_pixel(self, point: np.ndarray) -> tuple[float, float]:
        """Converts a point from function domain to pixel coordinates."""
        d = self.vis_domain
        nx = (point[0] - d['xmin']) / (d['xmax'] - d['xmin'])
        ny = (point[1] - d['ymin']) / (d['ymax'] - d['ymin'])
        px = nx * (self.resolution - 1)
        py = (1 - ny) * (self.resolution - 1)
        return (px, py)

    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def _get_domain(self):
        return np.array([[self._domain[0],self._domain[1]],[self._domain[2],self._domain[3]]])

    def _get_x_f(self):
        x = self.x0
        f = None
        x_list: list[torch.Tensor] = [x]
        f_list = []
        state = {}
        for i in range(self.n_steps):
            g, f = self.grad_and_value(x)
            f_list.append(f)

            x, state = self.opt(x, f, g, state)
            x_list.append(x)

        f = self.func(x) # f with last params
        f_list.append(f)

        return x_list, f_list

    def get_loss(self):
        x_list, f_list = self._get_x_f()

        if self._make_images:
            frame = self.background.copy()
            draw = ImageDraw.Draw(frame)

            pixel_coords = [self._world_to_pixel(tonumpy(p)) for p in x_list]

            if len(pixel_coords) > 1:
                draw.line(pixel_coords, fill=(255, 0, 0, 255), width=1)
                draw.point(pixel_coords)

            radius = 3
            for i, p in enumerate(pixel_coords):
                if i == 0:
                    fill_color = (0, 255, 0, 255) # Start: Green
                elif i == len(pixel_coords) - 1:
                    fill_color = (255, 255, 255, 255) # End: White
                else:
                    continue # Optionally draw intermediate points

                outline_color = (0, 0, 0, 255)
                bbox = [p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius]
                draw.ellipse(bbox, fill=fill_color, outline=outline_color)

            frame = np.array(frame)
            self.log_image("trajectory", frame, to_uint8=False)


        return f_list[-1]
