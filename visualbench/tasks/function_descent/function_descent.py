# pylint:disable=no-member
"""2D function descent"""

import os
from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from ...benchmark import Benchmark
from ...utils._benchmark_video import _maybe_progress, GIF_POST_PBAR_MESSAGE
from ...utils.format import tonumpy, totensor
from ...utils.funcplot import funcplot2d
from ...utils.renderer import OpenCVRenderer
from .test_functions import TEST_FUNCTIONS, TestFunction


class _UnpackCall:
    __slots__ = ("f", )
    def __init__(self, f): self.f=f
    def __call__(self, *x): return self.f(torch.stack(x, 0))

def _safe_flatten(x):
    # stupid 0d tensors are iterable but not
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 0: return x
    if isinstance(x, Iterable): return [_safe_flatten(i) for i in x]
    return x

class FunctionDescent(Benchmark):
    """descend a function.

    Args:
        func (Callable | str):
            function or string name of one of the test functions.
            Use ``FunctionDescent.list_funcs()`` to print all functions.
        x0 (ArrayLike): initial parameters (if None, func must be a string)
        bounds:
            Either ``(xmin, xmax, ymin, ymax)``, or ``((xmin, xmax), (ymin, ymax))`.`
            This is only used for plotting and defines the extent of what is plotted. If None,
            bounds are determined from minimum and maximum values of coords that have been visited.
        minima (tuple[float, float], optional): optinal coords of the minima for plotting. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        device (torch.types.Device, optional): device. Defaults to "cuda".
        unpack (bool, optional):
            if True, function is called as ``func(x[0], x[1])``, otherwise ``func(x)``. Defaults to True.
    """
    _LOGGER_XY_KEY: str = "params"
    _LEARNABLE_XY: bool = True
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        minima = None,
        dtype: torch.dtype = torch.float32,
        mo_func: Callable | None = None,
        unpack=True,
    ):
        if isinstance(func,str): f = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = dtype)
        else: f = func

        if isinstance(f, TestFunction):
            if x0 is None: x0 = f.x0()
            if domain is None: domain = f.domain()
            if minima is None: minima = f.minima()
            if mo_func is None: mo_func = f.mo_func()
            unpack = True

        x0 = totensor(x0, dtype=dtype)
        super().__init__() # log_params will be True by default. But False in neural descent for large nets.

        self.func: Callable[..., torch.Tensor] | TestFunction = f # type:ignore

        if domain is not None: self._domain = tonumpy(_safe_flatten(domain))
        else: self._domain = None

        self.unpack = unpack
        if minima is not None: self.minima = totensor(minima)
        else: self.minima = minima

        if self._LEARNABLE_XY:
            self.xy = torch.nn.Parameter(x0.requires_grad_(True))
        else:
            self.xy = torch.nn.Buffer(x0.requires_grad_(False))

        if mo_func is not None:
            self.set_multiobjective_func(mo_func)



    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def _get_domain(self):
        if self._domain is None:
            params = self.logger.numpy(self._LOGGER_XY_KEY)
            return np.array(list(zip(params.min(0), params.max(0))))
        return np.array([[self._domain[0],self._domain[1]],[self._domain[2],self._domain[3]]])

    def get_loss(self):
        xy = self.xy
        if self.unpack:
            xy = xy.clone()
            loss = self.func(xy[0], xy[1])
        else:
            loss = self.func(xy) # type:ignore
        return loss

    @torch.no_grad
    def plot( # pyright:ignore[reportIncompatibleMethodOverride]
        self,
        cmap = 'gray',
        contour_levels = 25,
        contour_cmap = 'binary',
        marker_cmap="coolwarm",
        contour_lw = 0.5,
        contour_alpha = 0.3,
        marker_size=7.,
        marker_alpha=1.,
        linewidth=0.5,
        line_alpha=1.,
        linecolor="red",
        norm=None,
        log_contour=False,
        ax=None,
    ):
        if ax is None: fig, ax = plt.subplots(figsize=(7,7))
        bounds = self._get_domain()

        if self.unpack: f = self.func
        else: f = _UnpackCall(self.func)

        f_proc = f
        sample_output = f(*torch.tensor([0., 0.]))
        if sample_output.numel() > 1:
            mf = self._multiobjective_func
            assert mf is not None
            f_single = lambda x,y: mf(f(x,y))
            f_proc = f_single

        funcplot2d(f_proc, *bounds, cmap = cmap, levels = contour_levels, contour_cmap = contour_cmap, contour_lw=contour_lw, contour_alpha=contour_alpha, norm=norm, log_contour=log_contour, lib=torch, ax=ax) # type:ignore

        if self._LOGGER_XY_KEY in self.logger:
            params = self.logger.numpy(self._LOGGER_XY_KEY)
            # params = np.clip(params, *bounds.T) # type:ignore
            losses = self.logger.numpy('train loss')

            if len(params) > 0:
                ax.scatter(*params.T, c=losses[:len(params)], cmap=marker_cmap, s=marker_size, alpha=marker_alpha)
                ax.plot(*params.T, alpha=line_alpha, lw=linewidth, c=linecolor)
                ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])

        if self.minima is not None:
            ax.scatter(tonumpy([self.minima[0]]), tonumpy(self.minima[1]), s=16, marker='x', c="red")
        return ax

    def _make_colors(self, s='rgb'):
        # make nice colors from losses
        loss_history = self.logger.numpy('train loss').copy()
        colors = np.array(loss_history, copy=True)
        colors = np.nan_to_num(loss_history, nan = np.nanmax(loss_history), posinf = np.nanmax(loss_history), neginf = np.nanmin(loss_history))
        if colors.min() < 0: colors -= colors.min()
        colors /= colors.max()

        red = np.where(colors > 0.5, 1., colors * 2.)
        green = np.where(colors <= 0.5, 1., (1 - colors) * 2.)
        blue = np.zeros_like(colors)

        d = {"r":red, "g":green, "b":blue}
        colors = np.clip(np.stack([d[c] for c in s], axis=-1) * 255, 0, 255).astype(np.uint8)
        return colors

    @torch.no_grad
    def render( # pyright:ignore[reportIncompatibleMethodOverride] # pylint:disable=arguments-renamed
        self,
        file: str | os.PathLike,
        fps: int = 60,
        resolution: int = 720,
        log_contour: bool = True,
        contour_levels: int = 20,
        cmap: str = 'gray',
        contour_cmap: str = 'binary',
        contour_thickness: float = 0.1,
        line_alpha: float = 0.5,
        progress: bool = True,
        scale: int = 1,
    ):
        import cv2
        bounds = self._get_domain()
        if self.unpack: f = self.func
        else: f = _UnpackCall(self.func)
        f_proc = f

        sample_output = f(*torch.tensor([0., 0.]))
        if sample_output.numel() > 1:
            mf = self._multiobjective_func
            assert mf is not None
            f_single = lambda x,y: mf(f(x,y))
            f_proc = f_single

        # make frame with matplotlib
        fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Use the whole figure  # pyright:ignore[reportCallIssue, reportArgumentType]
        ax.axis('off') # No axes, ticks, or labels

        funcplot2d(
            f_proc, *bounds, num=resolution, cmap=cmap,
            levels=contour_levels, contour_cmap=contour_cmap,
            contour_lw=contour_thickness, contour_alpha=1.0, # Alpha is handled by cv2 later
            log_contour=log_contour, lib='torch', ax=ax
        )
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])

        # render to numpy
        fig.canvas.draw()
        background = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)  # pyright:ignore[reportAttributeAccessIssue]
        background = background.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3].copy()
        # background_bgr = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        # coords to pixel indexes
        coord_history = self.logger.numpy(self._LOGGER_XY_KEY)
        def _world_to_pixel(coords, domain_bounds, image_size):
            """Maps world coordinates to pixel coordinates."""
            coords = np.nan_to_num(coords, nan=0, posinf=1e10, neginf=-1e10)
            pix = np.zeros_like(coords, dtype=np.int32)
            width, height = image_size

            denom = domain_bounds[0, 1] - domain_bounds[0, 0]
            if abs(denom) < 1e-12: denom = 1
            pix[:, 0] = ((coords[:, 0] - domain_bounds[0, 0]) / denom) * (width - 1)

            denom = domain_bounds[1, 1] - domain_bounds[1, 0]
            if abs(denom) < 1e-12: denom = 1
            pix[:, 1] = (1 - (coords[:, 1] - domain_bounds[1, 0]) / denom) * (height - 1)
            return pix

        pixel_coords = _world_to_pixel(coord_history, bounds, (resolution, resolution))

        with OpenCVRenderer(file, fps) as renderer:
            # persistent overlay which is added with transparency
            line_overlay = np.zeros_like(background, dtype=np.uint8)

            # 1st point and minima
            cv2.circle(background, pixel_coords[0], 4, (0,255,0), -1, lineType=cv2.LINE_AA)
            if self.minima is not None:
                cv2.drawMarker(
                    background,
                    _world_to_pixel(self.minima.unsqueeze(0), bounds, (resolution, resolution))[0],
                    (0, 255, 255)
                )

            line_color_bgr = (255, 0, 0)
            marker_color_bgr = (255, 255, 255)

            iterator = range(1, len(pixel_coords))
            colors = self._make_colors('rbg')
            for i in _maybe_progress(iterator, enable=progress):
                p1 = tuple(pixel_coords[i-1])
                p2 = tuple(pixel_coords[i])
                c = colors[i]

                # new line and circle (point is 1 behind so that it is on top)
                cv2.circle(background, p2, 2, c.tolist(), -1, lineType=cv2.LINE_AA)
                cv2.line(line_overlay, p1, p2, line_color_bgr, thickness=1, lineType=cv2.LINE_AA)
                # cv2.circle(background, p2, 4, (255,255,255), -1, lineType=cv2.LINE_AA)

                # blend it
                frame = cv2.addWeighted(background, 1.0, line_overlay, line_alpha, 0)

                # current point
                cv2.circle(frame, p2, 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, p2, 4, marker_color_bgr, -1, lineType=cv2.LINE_AA)

                renderer.write(frame)

            if progress and str(file).lower().endswith(".gif"):
                print(GIF_POST_PBAR_MESSAGE)