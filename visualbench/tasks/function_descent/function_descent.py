# pylint:disable=no-member
"""2D function descent"""


from collections.abc import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from ...benchmark import Benchmark
from ...utils._benchmark_video import _maybe_progress
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
        x0 (ArrayLike): initial parameters
        bounds:
            Only used for 2D functions. Either `(xmin, xmax, ymin, ymax)`, or `((xmin, xmax), (ymin, ymax))`.
            This is only used for plotting and defines the extent of what is plotted. If None,
            bounds are determined from minimum and maximum values of coords that have been visited.
        minima (_type_, optional): optinal coords of the minima. Defaults to None.
        dtype (torch.dtype, optional): dtype. Defaults to torch.float32.
        device (torch.types.Device, optional): device. Defaults to "cuda".
        unpack (bool, optional): if True, function is called as `func(*x)`, otherwise `func(x)`. Defaults to True.
    """
    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float,float,float,float] | Sequence[float] | None = None,
        minima = None,
        dtype: torch.dtype = torch.float32,
        unpack=True,
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

        self.func: Callable[..., torch.Tensor] | TestFunction = f # type:ignore

        if domain is not None: self._domain = tonumpy(_safe_flatten(domain))
        else: self._domain = None

        self.unpack = unpack
        if minima is not None: self.minima = totensor(minima)
        else: self.minima = minima

        self.params = torch.nn.Parameter(x0.requires_grad_(True))
        # self.noise = noise
        # self.noise_tensor = torch.nn.Buffer(torch.randn_like(self.params))
        # self.noise_batch = 0

    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def _get_domain(self):
        if self._domain is None:
            params = self.logger.numpy('params')
            return np.array(list(zip(params.min(0), params.max(0))))
        return np.array([[self._domain[0],self._domain[1]],[self._domain[2],self._domain[3]]])

    def get_loss(self):
        params = self.params
        if self.unpack:
            params = params.clone()
            loss = self.func(params[0], params[1])
        else:
            loss = self.func(params) # type:ignore
        return loss

    @torch.no_grad
    def plot(
        self,
        cmap = 'gray',
        contour_levels = 25,
        contour_cmap = 'binary',
        marker_cmap="coolwarm",
        contour_lw = 0.5,
        contour_alpha = 0.3,
        marker_size=7.,
        marker_alpha=0.5,
        linewidth=0.5,
        line_alpha=0.5,
        linecolor="red",
        norm=None,
        log_contour=False,
        ax=None,
    ):
        if ax is None: fig, ax = plt.subplots(figsize=(7,7))
        bounds = self._get_domain()

        if self.unpack: f = self.func
        else: f = _UnpackCall(self.func)
        funcplot2d(f, *bounds, cmap = cmap, levels = contour_levels, contour_cmap = contour_cmap, contour_lw=contour_lw, contour_alpha=contour_alpha, norm=norm, log_contour=log_contour, lib=torch) # type:ignore

        if 'params' in self.logger:
            params = self.logger.numpy('params')
            # params = np.clip(params, *bounds.T) # type:ignore
            losses = self.logger.numpy('train loss')

            if len(params) > 0:
                ax.scatter(*params.T, c=losses, cmap=marker_cmap, s=marker_size, alpha=marker_alpha)
                ax.plot(*params.T, alpha=line_alpha, lw=linewidth, c=linecolor)
                ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1])

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

    # override render, because get_loss doesnt make frames so that it is even faster
    @torch.no_grad
    def render_fast(self, file: str, fps: int = 60, scale: int | float = 1, progress=True, num: int = 500, marker_size:int|np.ndarray = 2, marker_alpha: float = 0.5, ):
        bounds = self._get_domain()
        xrange = bounds[0]
        yrange = bounds[1]

        # note: variables have "x" and "y" in their names.
        # but I am not sure if they are actually x and y coordinates.
        # I had to suswap some of them by trial and error
        x_spacing = torch.linspace(*xrange, num) # type:ignore
        y_spacing = torch.linspace(*yrange, num) # type:ignore
        X,Y = torch.meshgrid(x_spacing, y_spacing, indexing='xy') # grid of points

        # make function image
        if self.unpack: img: np.ndarray = tonumpy(self.func(X, Y)) # type:ignore
        else: img: np.ndarray = tonumpy(self.func(torch.stack([X,Y], dim=0))).T # type:ignore
        img -= img.min()
        img /= img.max()
        img *= 255

        colors = self._make_colors()

        # turn coords into corresponding image pixels
        coord_history = self.logger.numpy('params').copy()
        x_coords = np.round(((coord_history[:, 0] - xrange[0]) / (xrange[1] - xrange[0])) * num).astype(np.int64)
        y_coords = np.round(((coord_history[:, 1] - yrange[0]) / (yrange[1] - yrange[0])) * num).astype(np.int64)

        sizes = np.asanyarray(marker_size)
        if sizes.ndim < 1:
            sizes = sizes[np.newaxis].repeat(len(x_coords), axis = 0)

        size = img.shape
        img = img[:,:,np.newaxis].repeat(repeats = 3, axis = 2)

        with OpenCVRenderer(file, fps, scale=scale) as renderer:
            frame = img.copy()
            for x, y, c, s in _maybe_progress(list(zip(x_coords, y_coords, colors, sizes)), enable = progress):
                out_of_screen = False
                if any(((not np.isfinite(ii)) for ii in (x, y, s,))): out_of_screen = True
                else:
                    x = int(x); y = int(y); s = int(s)
                    if (x - s < 0) or (y - s < 0) or (x + s >= size[1]) or (y + s >= size[0]):
                        out_of_screen = True
                    else:
                        # add persistent point to frame
                        frame[y - s : y + s, x - s : x + s, :] -= (frame[y - s : y + s, x - s : x + s, :] - c) * marker_alpha

                # clone current frame in order to show elarged current point, which is updated each frame
                cur_frame = np.clip(np.nan_to_num(frame), 0, 255).astype(np.uint8)
                if not out_of_screen:
                    sp = s+1
                    cur_frame[y - sp : y + sp, x - sp : x + sp, :] = 0
                    cur_frame[y - sp : y + sp, x - sp : x + sp, 2] = 255

                renderer.write(cur_frame[::-1])

    @torch.no_grad
    def render( # pyright:ignore[reportIncompatibleMethodOverride] # pylint:disable=arguments-renamed
        self,
        file: str,
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

        # make frame with matplotlib
        fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Use the whole figure  # pyright:ignore[reportCallIssue, reportArgumentType]
        ax.axis('off') # No axes, ticks, or labels

        funcplot2d(
            f, *bounds, num=resolution, cmap=cmap,
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
        coord_history = self.logger.numpy('params')
        def _world_to_pixel(coords, domain_bounds, image_size):
            """Maps world coordinates to pixel coordinates."""
            pix = np.zeros_like(coords, dtype=np.int32)
            width, height = image_size
            pix[:, 0] = ((coords[:, 0] - domain_bounds[0, 0]) / (domain_bounds[0, 1] - domain_bounds[0, 0])) * (width - 1)
            pix[:, 1] = (1 - (coords[:, 1] - domain_bounds[1, 0]) / (domain_bounds[1, 1] - domain_bounds[1, 0])) * (height - 1)
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