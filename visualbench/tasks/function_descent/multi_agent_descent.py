# pylint:disable=no-member
"""Multi-agent function descent - multiple particles descending a 2D function."""

import os
from collections.abc import Callable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ...benchmark import Benchmark
from ...utils._benchmark_video import _maybe_progress, GIF_POST_PBAR_MESSAGE
from ...utils.format import tonumpy, totensor
from ...utils.funcplot import funcplot2d
from ...utils.renderer import OpenCVRenderer
from .function_descent import _safe_flatten
from .test_functions import TEST_FUNCTIONS, TestFunction


class MultiAgentDescent(Benchmark):
    """Multiple agents descend a 2D function simultaneously.

    This benchmark is useful for studying how optimizers handle multiple particles,
    diversity in optimization, and collective behavior. Agents can optionally repel
    each other to encourage exploration of different minima.

    Args:
        func (Callable | str):
            function or string name of one of the test functions.
            Use ``MultiAgentDescent.list_funcs()`` to print all functions.
        n_agents (int): number of agents. Defaults to 16.
        x0 (Sequence | None):
            initial parameters for all agents. If None, agents are initialized
            randomly within the domain.
        domain:
            Either ``(xmin, xmax, ymin, ymax)``, or ``((xmin, xmax), (ymin, ymax))``.
            This is only used for plotting and defines the extent of what is plotted. If None,
            bounds are determined from minimum and maximum values of coords that have been visited.
        minima (Sequence | None): optional coords of minima for plotting. Defaults to None.
        dtype (torch.dtype): dtype. Defaults to torch.float32.
        repulsion_strength (float):
            strength of repulsion between agents. 0 means no repulsion.
            Higher values encourage agents to spread out and find different minima.
            Defaults to 0.0.
        unpack (bool):
            if True, function is called as ``func(x, y)``, otherwise ``func(x)``. Defaults to True.
    """

    _LOGGER_XY_KEY: str = "params"

    def __init__(
        self,
        func: Callable[..., torch.Tensor] | str | TestFunction,
        n_agents: int = 16,
        x0: Sequence | np.ndarray | torch.Tensor | None = None,
        domain: tuple[float, float, float, float] | Sequence[float] | None = None,
        minima=None,
        dtype: torch.dtype = torch.float32,
        repulsion_strength: float = 1,
        unpack: bool = True,
    ):
        if isinstance(func, str):
            f = TEST_FUNCTIONS[func].to(device='cpu', dtype=dtype)
        else:
            f = func

        if isinstance(f, TestFunction):
            if x0 is None:
                # Random initialization within domain
                if domain is None:
                    domain = f.domain()
                x0 = self._random_init(domain, n_agents, dtype)
            if domain is None:
                domain = f.domain()
            if minima is None:
                minima = f.minima()
            unpack = True
        else:
            # Custom callable - require x0 or initialize randomly with default domain
            if x0 is None:
                if domain is None:
                    domain = (-10, 10, -10, 10)  # Default domain for custom functions
                x0 = self._random_init(domain, n_agents, dtype)

        super().__init__()

        self.func: Callable[..., torch.Tensor] | TestFunction = f  # type:ignore
        self.n_agents = n_agents
        self.repulsion_strength = repulsion_strength

        if domain is not None:
            self._domain = tonumpy(_safe_flatten(domain))
        else:
            self._domain = None

        self.unpack = unpack
        if minima is not None:
            self.minima = totensor(minima)
        else:
            self.minima = minima

        x0 = totensor(x0, dtype=dtype)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0).expand(n_agents, -1)
        self.xy = torch.nn.Parameter(x0.requires_grad_(True))

    def _random_init(self, domain, n_agents, dtype):
        """Initialize agents randomly within the domain."""
        domain = _safe_flatten(domain)
        xmin, xmax, ymin, ymax = domain
        x = torch.rand(n_agents, dtype=dtype) * (xmax - xmin) + xmin
        y = torch.rand(n_agents, dtype=dtype) * (ymax - ymin) + ymin
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def list_funcs():
        print(sorted(list(TEST_FUNCTIONS.keys())))

    def _get_domain(self) -> np.ndarray:
        if self._domain is None:
            params = self.logger.to_numpy(self._LOGGER_XY_KEY)
            if len(params) == 0:
                return np.array([[-10, 10], [-10, 10]])
            return np.array(list(zip(params.min(0), params.max(0))))
        return np.array([[self._domain[0], self._domain[1]], [self._domain[2], self._domain[3]]])

    def _compute_repulsion(self):
        """Compute pairwise repulsion between agents."""
        if self.repulsion_strength == 0:
            return torch.tensor(0.0, device=self.xy.device, dtype=self.xy.dtype)

        # Compute xy normalized by domain size
        domain = self._get_domain()
        xrange, yrange = np.abs(domain[:, 0] - domain[:, 1])
        if xrange == 0 or yrange == 0: # can happen when domain is not specified and there is 1 point
            return torch.tensor(0.0, device=self.xy.device, dtype=self.xy.dtype)

        xy = torch.stack([self.xy[:, 0] / xrange, self.xy[:, 1] / yrange], 1)

        diff = xy.unsqueeze(0) - xy.unsqueeze(1)  # (n, n, 2)
        penalty = diff.abs().add(torch.finfo(diff.dtype).eps).reciprocal()

        # Exclude self-interactions
        penalty = penalty * (1 - torch.eye(self.n_agents, device=self.xy.device).unsqueeze(-1))

        # Sum over all pairs and scale
        return self.repulsion_strength * penalty.mean()

    def get_loss(self):
        if self.unpack:
            losses = self.func(self.xy[:, 0], self.xy[:, 1])
        else:
            losses = self.func(self.xy)  # type:ignore

        if not isinstance(losses, torch.Tensor) or losses.numel() == 1:
            losses = losses.repeat(self.n_agents)

        # Average function value + repulsion
        repulsion_loss = self._compute_repulsion()
        return losses.mean() + repulsion_loss

    @torch.no_grad
    def plot(  # pyright:ignore[reportIncompatibleMethodOverride]
        self,
        cmap='gray',
        contour_levels=25,
        contour_cmap='binary',
        marker_cmap="coolwarm",
        contour_lw=0.5,
        contour_alpha=0.5,
        marker_size=7.0,
        marker_alpha=0.4,
        linewidth=0.5,
        line_alpha=0.5,
        linecolor="red",
        norm=None,
        log_contour=False,
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        bounds = self._get_domain()

        if self.unpack:
            f = self.func
        else:
            f = lambda x, y: self.func(torch.stack([x, y])) # pyright:ignore[reportCallIssue]

        f_proc = f
        sample_output = f(*torch.tensor([0., 0.]))
        if sample_output.numel() > 1:
            mf = self._multiobjective_func
            assert mf is not None
            f_single = lambda x, y: mf(f(x, y))
            f_proc = f_single

        funcplot2d(
            f_proc, *bounds, cmap=cmap, levels=contour_levels,
            contour_cmap=contour_cmap, contour_lw=contour_lw,
            contour_alpha=contour_alpha, norm=norm, log_contour=log_contour,
            lib=torch, ax=ax  # type:ignore
        )

        if self._LOGGER_XY_KEY in self.logger:
            params = self.logger.to_numpy(self._LOGGER_XY_KEY)
            losses = self.logger.to_numpy('train loss')

            if len(params) > 0:
                # Reshape params from (steps, n_agents*2) to (steps, n_agents, 2)
                if params.ndim == 2 and params.shape[1] == self.n_agents * 2:
                    params = params.reshape(params.shape[0], self.n_agents, 2)
                elif params.ndim == 1:
                    params = params.reshape(self.n_agents, 2)[None]

                # params shape: (steps, n_agents, 2)
                if params.ndim == 3:
                    # Plot all agents
                    for i in range(params.shape[1]):
                        agent_params = params[:, i]
                        agent_losses = losses[:len(agent_params)] if len(losses) >= len(agent_params) else losses
                        ax.scatter(
                            agent_params[:, 0], agent_params[:, 1],
                            c=agent_losses,
                            cmap=marker_cmap, s=marker_size, alpha=marker_alpha
                        )
                        ax.plot(
                            agent_params[:, 0], agent_params[:, 1],
                            alpha=line_alpha, lw=linewidth, c=linecolor
                        )
                elif params.ndim == 2:
                    # Single agent or flattened
                    if params.shape[1] == 2:
                        ax.scatter(
                            params[:, 0], params[:, 1],
                            c=losses[:len(params)],
                            cmap=marker_cmap, s=marker_size, alpha=marker_alpha
                        )
                        ax.plot(params[:, 0], params[:, 1], alpha=line_alpha, lw=linewidth, c=linecolor)

                ax.set_xlim(*bounds[0])
                ax.set_ylim(*bounds[1])

        if self.minima is not None:
            ax.scatter(
                tonumpy([self.minima[0]]), tonumpy(self.minima[1]),
                s=16, marker='x', c="red"
            )
        return ax

    @torch.no_grad
    def render(  # pyright:ignore[reportIncompatibleMethodOverride]
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
        bounds = self._get_domain()
        if self.unpack:
            f = self.func
        else:
            f = lambda x, y: self.func(torch.stack([x, y]))

        f_proc = f
        sample_output = f(*torch.tensor([0., 0.]))
        if sample_output.numel() > 1:
            mf = self._multiobjective_func
            assert mf is not None
            f_single = lambda x, y: mf(f(x, y))
            f_proc = f_single

        # Make frame with matplotlib
        fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        funcplot2d(
            f_proc, *bounds, num=resolution, cmap=cmap,
            levels=contour_levels, contour_cmap=contour_cmap,
            contour_lw=contour_thickness, contour_alpha=1.0,
            log_contour=log_contour, lib='torch', ax=ax
        )
        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])

        # Render to numpy
        fig.canvas.draw()
        background = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)  # pyright:ignore[reportAttributeAccessIssue]
        background = background.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
        plt.close(fig)

        # Coords to pixel indexes
        coord_history = self.logger.to_numpy(self._LOGGER_XY_KEY)
        # Reshape from (steps, n_agents*2) to (steps, n_agents, 2)
        if coord_history.ndim == 2:
            n_steps = coord_history.shape[0]
            coord_history = coord_history.reshape(n_steps, self.n_agents, 2)
        elif coord_history.ndim == 1:
            coord_history = coord_history.reshape(self.n_agents, 2)[None]

        def _world_to_pixel(coords, domain_bounds, image_size):
            """Maps world coordinates to pixel coordinates."""
            coords = np.nan_to_num(coords, nan=0, posinf=1e10, neginf=-1e10)
            pix = np.zeros_like(coords, dtype=np.int32)
            width, height = image_size

            denom = domain_bounds[0, 1] - domain_bounds[0, 0]
            if abs(denom) < 1e-12:
                denom = 1
            pix[:, 0] = ((coords[:, 0] - domain_bounds[0, 0]) / denom) * (width - 1)

            denom = domain_bounds[1, 1] - domain_bounds[1, 0]
            if abs(denom) < 1e-12:
                denom = 1
            pix[:, 1] = (1 - (coords[:, 1] - domain_bounds[1, 0]) / denom) * (height - 1)
            return pix

        pixel_coords_history = np.array([
            _world_to_pixel(agent_coords, bounds, (resolution, resolution))
            for agent_coords in coord_history
        ])

        with OpenCVRenderer(file, fps) as renderer:
            line_overlay = np.zeros_like(background, dtype=np.uint8)

            # Minima
            if self.minima is not None:
                cv2.drawMarker(
                    background,
                    _world_to_pixel(self.minima.unsqueeze(0), bounds, (resolution, resolution))[0],
                    (0, 255, 255)
                )

            line_color_bgr = (255, 0, 0)

            iterator = range(1, len(pixel_coords_history))
            colors = self._make_colors('rbg')

            for i in _maybe_progress(iterator, enable=progress):
                frame = background.copy()

                for agent_idx in range(pixel_coords_history.shape[1]):
                    p1 = tuple(pixel_coords_history[i - 1, agent_idx])
                    p2 = tuple(pixel_coords_history[i, agent_idx])
                    c = colors[min(i, len(colors) - 1)]

                    # Draw line
                    cv2.line(line_overlay, p1, p2, line_color_bgr, thickness=1, lineType=cv2.LINE_AA)
                    # Draw point
                    cv2.circle(frame, p2, 3, c.tolist(), -1, lineType=cv2.LINE_AA)

                # Blend overlay
                frame = cv2.addWeighted(frame, 1.0, line_overlay, line_alpha, 0)

                renderer.write(frame)

            if progress and str(file).lower().endswith(".gif"):
                print(GIF_POST_PBAR_MESSAGE)

    def _make_colors(self, s='rgb'):
        """Make nice colors from losses."""
        loss_history = self.logger.to_numpy('train loss').copy()
        colors = np.array(loss_history, copy=True)
        colors = np.nan_to_num(
            loss_history,
            nan=np.nanmax(loss_history),
            posinf=np.nanmax(loss_history),
            neginf=np.nanmin(loss_history)
        )
        if colors.min() < 0:
            colors -= colors.min()
        colors /= colors.max()

        red = np.where(colors > 0.5, 1., colors * 2.)
        green = np.where(colors <= 0.5, 1., (1 - colors) * 2.)
        blue = np.zeros_like(colors)

        d = {"r": red, "g": green, "b": blue}
        colors = np.clip(np.stack([d[c] for c in s], axis=-1) * 255, 0, 255).astype(np.uint8)
        return colors
