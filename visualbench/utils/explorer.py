from collections.abc import Callable, Sequence
from functools import partial

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import display
from matplotlib.colors import LogNorm
from torch import nn


class FunctionExplorer:
    """
    Interactive widget for visualizing an n-dimensional function.

    Args:
        func (callable): The function to visualize. It should accept a
            torch.Tensor of shape (n_dims,) or a batched tensor of shape
            (batch_size, n_dims) and return a scalar torch.Tensor or a
            batched tensor of shape (batch_size,).
        n_dims (int): The number of input dimensions for the function.
        param_names (list of str, optional): Names for each dimension.
            If None, defaults to ['x0', 'x1', ...].
        default_range (tuple, optional): The default (min, max) range for all sliders.
    """
    def __init__(self, func, n_dims, param_names=None, default_range=(-5.0, 5.0)):
        if n_dims < 2:
            raise ValueError("n_dims must be at least 2 for 2D visualization.")

        self.func = func
        self.n_dims = n_dims
        self.param_names = param_names or [f'x{i}' for i in range(n_dims)]
        self.default_range = default_range
        self._is_batched = None  # Will be determined on the first run

        # To store the last computed data for quick replotting
        self._last_computed_data = {}

        self._create_widgets()
        self._link_widgets()

        # Initial call to set up dynamic widgets
        self._on_axis_change(None)

    def _create_widgets(self):
        """Creates all the widgets for the interactive display."""
        # --- Dimension Selectors ---
        self.x_axis_selector = widgets.Dropdown(
            options=self.param_names, value=self.param_names[0], description='X-Axis:'
        )
        self.y_axis_selector = widgets.Dropdown(
            options=self.param_names, value=self.param_names[1], description='Y-Axis:'
        )

        # --- Range Sliders for X and Y axes ---
        self.x_range_slider = widgets.FloatRangeSlider(
            value=self.default_range, min=self.default_range[0]-10, max=self.default_range[1]+10,
            step=0.1, description='X Range:', readout_format='.1f', continuous_update=False
        )
        self.y_range_slider = widgets.FloatRangeSlider(
            value=self.default_range, min=self.default_range[0]-10, max=self.default_range[1]+10,
            step=0.1, description='Y Range:', readout_format='.1f', continuous_update=False
        )
        self.range_sliders_box = widgets.VBox([self.x_range_slider, self.y_range_slider])

        # --- Sliders for constant dimensions (dynamically created) ---
        self.constant_sliders = {}
        self.constant_sliders_box = widgets.VBox([])

        # --- Visualization Controls ---
        self.resolution_slider = widgets.IntSlider(
            value=25, min=10, max=200, step=5, description='Resolution:', continuous_update=False
        )
        self.contours_slider = widgets.IntSlider(
            value=15, min=0, max=50, step=1, description='Contours:'
        )
        self.log_scale_check = widgets.Checkbox(value=False, description='Log Scale')

        # --- Output Area ---
        self.output = widgets.Output()

    def _link_widgets(self):
        """Links widget events to handler functions."""
        # Changes that require re-calculating the grid
        self.x_axis_selector.observe(self._on_axis_change, names='value')
        self.y_axis_selector.observe(self._on_axis_change, names='value')
        self.resolution_slider.observe(self._update_plot, names='value')
        self.x_range_slider.observe(self._update_plot, names='value')
        self.y_range_slider.observe(self._update_plot, names='value')

        # Changes that only require replotting
        self.log_scale_check.observe(self._replot, names='value')
        self.contours_slider.observe(self._replot, names='value')

    def _on_axis_change(self, change):
        """Handles changes in X or Y axis selection."""
        # Prevent selecting the same axis for X and Y
        if self.x_axis_selector.value == self.y_axis_selector.value:
            # Find a new unique value for the y-axis
            x_idx = self.param_names.index(self.x_axis_selector.value)
            new_y_idx = (x_idx + 1) % self.n_dims
            self.y_axis_selector.value = self.param_names[new_y_idx]
            # The y_axis_selector's own observer will handle the update from here
            return

        self._update_constant_sliders()
        self._update_plot(None) # Trigger a full re-computation

    def _update_constant_sliders(self):
        """Creates or updates sliders for dimensions not on the X or Y axes."""
        x_dim = self.x_axis_selector.value
        y_dim = self.y_axis_selector.value

        new_sliders = {}
        sliders_to_display = []

        for i, name in enumerate(self.param_names):
            if name != x_dim and name != y_dim:
                # Reuse existing slider if possible to preserve value
                if name in self.constant_sliders:
                    slider = self.constant_sliders[name]
                else:
                    slider = widgets.FloatSlider(
                        value=sum(self.default_range)/2.0, min=self.default_range[0], max=self.default_range[1],
                        step=0.1, description=name, continuous_update=False, readout_format='.1f'
                    )

                # Link the slider to the update function
                slider.observe(self._update_plot, names='value')
                new_sliders[name] = slider
                sliders_to_display.append(slider)

        self.constant_sliders = new_sliders
        self.constant_sliders_box.children = tuple(sliders_to_display[:32])

    def _evaluate_on_grid(self, grid_tensor):
        """Evaluates the function on a grid, handling batched/unbatched cases."""
        with torch.no_grad():
            # First time running, detect if function is batched
            if self._is_batched is None:
                try:
                    # Try calling with a batch of 2
                    test_batch = grid_tensor[:2]
                    output = self.func(test_batch)
                    if output.shape[0] == 2:
                        self._is_batched = True
                    else: # Output shape is wrong, assume not batched
                        self._is_batched = False
                except Exception:
                    self._is_batched = False

            # Evaluate based on detected type
            if self._is_batched:
                results = self.func(grid_tensor)
            else:
                # Apply function row-by-row if not batched
                results = torch.stack([self.func(row) for row in grid_tensor])

        return results.squeeze().cpu().numpy()

    def _update_plot(self, change):
        """
        Performs the full computation of the function on the grid.
        This is slow and should only be called when necessary.
        """
        x_idx = self.param_names.index(self.x_axis_selector.value)
        y_idx = self.param_names.index(self.y_axis_selector.value)

        resolution = self.resolution_slider.value
        x_range = self.x_range_slider.value
        y_range = self.y_range_slider.value

        # Create grid
        x_vals = torch.linspace(x_range[0], x_range[1], resolution)
        y_vals = torch.linspace(y_range[0], y_range[1], resolution)
        grid_x, grid_y = torch.meshgrid(x_vals, y_vals, indexing='ij')

        # Prepare input tensor for the function
        # Shape: (resolution*resolution, n_dims)
        input_tensor = torch.full((resolution * resolution, self.n_dims), 0.0)

        # Set constant values from sliders
        for i, name in enumerate(self.param_names):
            if i != x_idx and i != y_idx:
                input_tensor[:, i] = self.constant_sliders[name].value

        # Set grid values
        input_tensor[:, x_idx] = grid_x.flatten()
        input_tensor[:, y_idx] = grid_y.flatten()

        # Evaluate function and reshape
        z_vals = self._evaluate_on_grid(input_tensor)
        z_grid = z_vals.reshape(resolution, resolution)

        # Cache results for fast replotting
        self._last_computed_data = {
            'x_grid': grid_x.cpu().numpy(),
            'y_grid': grid_y.cpu().numpy(),
            'z_grid': z_grid,
        }

        self._replot(None)

    def _replot(self, change):
        """
        Redraws the plot using the last computed data. This is fast.
        """
        if not self._last_computed_data:
            return

        x_grid = self._last_computed_data['x_grid']
        y_grid = self._last_computed_data['y_grid']
        z_grid = self._last_computed_data['z_grid']

        use_log = self.log_scale_check.value
        num_contours = self.contours_slider.value

        with self.output:
            self.output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 5))

            # --- Plotting logic ---
            norm = LogNorm() if use_log else None

            # Contour plot (filled)
            try:
                contourf = ax.pcolormesh(x_grid, y_grid, z_grid, cmap='viridis', norm=norm)
                # Contour lines
                if num_contours > 0:
                    ax.contour(x_grid, y_grid, z_grid, levels=num_contours, colors='white', linewidths=0.5, alpha=0.7)
                fig.colorbar(contourf, ax=ax, label='Function Value')
            except Exception as e:
                # Handle cases where plotting fails (e.g., all same values, NaNs)
                ax.text(0.5, 0.5, f"Plotting Error:\n{e}", ha='center', va='center', transform=ax.transAxes)

            ax.set_xlabel(self.x_axis_selector.value)
            ax.set_ylabel(self.y_axis_selector.value)
            ax.set_title(f'f({", ".join(self.param_names)})'[:100])
            fig.tight_layout()
            plt.show()

    def display(self):
        """Displays the assembled widget."""
        dim_selectors = widgets.HBox([self.x_axis_selector, self.y_axis_selector])
        viz_controls = widgets.VBox([
            self.resolution_slider, self.contours_slider, self.log_scale_check
        ])

        controls = widgets.VBox([
            widgets.HTML("<b>Axis Selection</b>"), dim_selectors,
            widgets.HTML("<b>Axis Ranges</b>"), self.range_sliders_box,
            widgets.HTML("<b>Constant Parameters</b>"), self.constant_sliders_box,
            widgets.HTML("<b>Visualization</b>"), viz_controls
        ], layout=widgets.Layout(width='400px'))

        display(widgets.HBox([controls, self.output]))

@torch.no_grad
def benchmark_display(bench, range=(-5,5), max_params = 32):
    n = sum(p.numel() for p in bench.parameters())
    params = torch.nn.utils.parameters_to_vector(bench.parameters())

    def objective(x):
        if n > max_params:
            p = params.clone()
            p[:x.numel()] = x
            x = p
        torch.nn.utils.vector_to_parameters(x, bench.parameters())
        return bench.get_loss()

    plotter = FunctionExplorer(objective, min(n,max_params), default_range=range)
    plotter.display()

if __name__ == "__main__":
    def high_dim_func_batched(x): # 10d
        part1 = torch.sin(x[:, 0] * x[:, 1])
        part2 = torch.cos(x[:, 2] - x[:, 3])
        part3 = torch.sum(x[:, 4:]**2, dim=1)
        return part1 + part2 + part3

    plotter_10d = FunctionExplorer(
        func=high_dim_func_batched,
        n_dims=10,
        default_range=(-3.14, 3.14)
    )
    plotter_10d.display()



# experimental
def latin_hypercube_sampler(objective, x0: torch.Tensor, n_points: int, domain: torch.Tensor):
    # domain is (2, ndim) or (2, )

    from scipy.stats.qmc import LatinHypercube
    sampler = LatinHypercube(d=x0.numel())
    points_norm = sampler.random(n=n_points) # samples in [0, 1]

    min, max = domain.cpu()
    X = torch.from_numpy(points_norm).float() * (max - min) + min
    return X, *objective(X.to(x0))

@torch.no_grad
def pca_projector(k: int, X: list[torch.Tensor], Y: list[torch.Tensor], G: list[torch.Tensor], whiten: bool = False):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k, whiten=whiten)
    pca.fit(np.asarray(G))

    def project(x):
        return torch.from_numpy(pca.transform(x.detach().cpu().numpy())).to(x)

    def unproject(x_proj):
        return torch.from_numpy(pca.inverse_transform(x_proj.detach().cpu().numpy())).to(x_proj)

    return project, unproject



class Visualizer(nn.Module):
    def __init__(self, objective: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]], x0: torch.Tensor):
        super().__init__()
        self.objective = objective

        self.x0 = x0
        self.device = x0.device
        self.dtype = x0.dtype
        self.ndim = x0.numel()

        self.project = lambda x: x
        self.unproject = lambda x: x

        y, g = objective(x0)
        self.X = [x0.detach().cpu()] # params
        self.Y = [y.detach().cpu()] # values
        self.G = [g.detach().cpu()] # gradients

        self.x_best = x0
        self.y_best = y
        self._iter_x_best = x0

        self.shift = True


    @torch.no_grad
    def get_loss_grad(self, x: torch.Tensor):
        with torch.enable_grad():

            losses = []
            grads = []

            for xi in x:
                l,g = self.objective(xi)
                losses.append(l.detach().cpu())
                grads.append(g.detach().cpu())

                if l < self.y_best:
                    self._iter_x_best = xi.detach().cpu()
                    self.y_best = losses[-1]

        self.X.extend(x.detach().cpu())
        self.Y.extend(losses)
        self.G.extend(grads)

        return torch.stack(losses), torch.stack(grads)

    @torch.no_grad
    def get_projected_loss_grad(self, x_proj: torch.Tensor):
        x = self.unproject(x_proj)
        if self.shift: x += self.x_best.to(x)

        losses, grads = self.get_loss_grad(x)
        return losses, self.project(grads)

    def search_iteration(
        self,
        n_points: int = 100,
        domain: tuple[float, float] = (-10, 10),
        sampler=latin_hypercube_sampler,
        projected: bool = True,
        shift: bool = True,
    ):
        self.shift = shift

        objective = self.get_projected_loss_grad if projected else self.get_loss_grad
        x0 = self.project(self.x0.unsqueeze(0))[0] if projected else self.x0

        sampler(objective, x0=x0, n_points=n_points, domain=torch.tensor(domain, device=self.device, dtype=self.dtype))
        self.x_best = self._iter_x_best


    def update_subspace(self, projector, k: int):
        self.project, self.unproject = projector(k, self.X, self.Y, self.G)


    def fit(self, levels: Sequence[int], n_points: int = 100, n_random = 10, domain: tuple[float, float] = (-10, 10), sampler=latin_hypercube_sampler, shift: bool = True, projector=partial(pca_projector, whiten=False), verbose: bool=False):
        for k in levels:
            self.search_iteration(n_points=n_points, domain=domain, sampler=sampler, projected=True, shift=shift)
            self.search_iteration(n_points=n_random, domain=domain, sampler=sampler, projected=False, shift=shift)

            self.update_subspace(projector=projector, k = k)
            if verbose: print(f'{k = }, {self.y_best = }')

    @torch.no_grad
    def get_grid(self, grid_range:float=2, grid_points=25):
        a1 = torch.linspace(-grid_range, grid_range, grid_points)
        a2 = torch.linspace(-grid_range, grid_range, grid_points)
        X, Y = torch.meshgrid(a1, a2, indexing='xy')
        Z = torch.zeros(grid_points, grid_points)

        for i in range(grid_points):
            for j in range(grid_points):

                x_proj = torch.stack((X[i, j], Y[i, j]))
                x = self.unproject(x_proj.unsqueeze(0))[0] + self.x_best

                y, _ = self.objective(x)
                Z[i, j] = y.detach().cpu()

        return X, Y, Z

    @torch.no_grad
    def plot(self, grid_range:float | None, grid_points=25, points: torch.Tensor | None=None, values: torch.Tensor | None = None, norm=None):
        trajectory_proj = None

        if points is not None: # (n, ndim)
            if values is not None: values = values.detach().cpu()
            trajectory_proj = self.project(points.detach().cpu() + self.x_best)
            print(f"{trajectory_proj = }")

        if grid_range is None:
            assert trajectory_proj is not None
            grid_range = float(trajectory_proj.abs().amax().item()) * 1.1

        X, Y, Z = self.get_grid(grid_range=grid_range, grid_points=grid_points)
        plt.figure(figsize=(8, 6))
        colormesh = plt.pcolormesh(X.numpy(), Y.numpy(), Z.numpy(), cmap='coolwarm', norm=norm)
        plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, colors='white', alpha=0.5, linewidths=0.7)
        plt.colorbar(colormesh, label="value")

        plt.plot(0, 0, 'r+', markersize=15, label=f'x_min (y={self.y_best.item():.2f})')
        if trajectory_proj is not None:
            plt.plot(*zip(*trajectory_proj), color='r', lw=0.75, alpha=0.5)
            plt.scatter(*zip(*trajectory_proj), c=values)

        plt.xlabel("α₁")
        plt.ylabel("α₂")
        plt.title(f"{self.ndim}-D function")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def bench_objective(bench):

    @torch.no_grad
    def objective(x: torch.Tensor, backward: bool=True):
        torch.nn.utils.vector_to_parameters(x, bench.parameters())

        if backward:
            with torch.enable_grad():
                loss = bench.get_loss()
                loss.backward()

            grad = torch.cat([p.grad.ravel() if p.grad is not None else torch.zeros_like(p) for p in bench.parameters()])
            return loss, grad

        return bench.get_loss(), None

    return objective
