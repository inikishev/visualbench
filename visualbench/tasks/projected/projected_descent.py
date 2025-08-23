from typing import Literal
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils.format import totensor
from ..synthetic import rosenbrock, chebushev_rosenbrock, ackley, rotated_quadratic, rastrigin


@torch.no_grad
def _make_basis(p1:torch.Tensor, p2:torch.Tensor, p3:torch.Tensor):
    """move p3 to form an orthogonal basis from p1-p2 and a perpendicular vector."""
    u = p2 - p1
    uu = u.dot(u)
    if uu <= 1e-15:
        p2 = p2 + torch.randn_like(p2) * 1e-6
        return _make_basis(p1, p2, p3)

    u_hat = u / uu.sqrt()

    v = p3 - p1
    v_onto_u_hat = v.dot(u_hat)
    w = v - v_onto_u_hat * u_hat

    w_norm = torch.linalg.vector_norm(w) # pylint:disable=not-callable
    if w_norm <= 1e-12:
        # p3 is collinear with p1 and p2
        p3 = p3 + torch.randn_like(p3) * 1e-6
        return _make_basis(p1, p2, p3)

    w_hat = w / w_norm

    return torch.stack([u_hat, w_hat], -1)

@torch.no_grad
def _draw_trajectory(Z:torch.Tensor, history_proj: torch.Tensor, xmin, xmax, ymin, ymax, resolution):
    z_min, z_max = Z.amin(), Z.amax()
    if (z_max - z_min) > 1e-9:
        z_p01 = torch.quantile(Z.flatten(), 0.01)
        z_p99 = torch.quantile(Z.flatten(), 0.99)
        if z_p99 - z_p01 < 1e-9:
            z_p01, z_p99 = z_min, z_max

        image = (255 * (Z - z_p01) / (z_p99 - z_p01)).clip(min=0, max=255).to(torch.uint8)
    else:
        image = torch.zeros_like(Z, dtype=torch.uint8)

    plot_xrange = xmax - xmin
    plot_yrange = ymax - ymin

    px = torch.round((history_proj[:, 0] - xmin) / plot_xrange * (resolution - 1)).long()
    py = torch.round((history_proj[:, 1] - ymin) / plot_yrange * (resolution - 1)).long()

    image = image.unsqueeze(-1).repeat_interleave(3, -1).clone()

    if len(history_proj) > 1:
        odd_color = torch.tensor((255, 64, 64),dtype=image.dtype, device=image.device)
        even_color = torch.tensor((64, 64, 255),dtype=image.dtype, device=image.device)

        px_path, py_path = px[:-1], py[:-1]

        valid_mask = (px_path >= 0) & (px_path < resolution) & \
                     (py_path >= 0) & (py_path < resolution)

        indices = torch.arange(len(px_path), device=px_path.device)

        odd_points_mask = valid_mask & (indices % 2 == 1)
        even_points_mask = valid_mask & (indices % 2 == 0)

        image[py_path[even_points_mask], px_path[even_points_mask]] = even_color
        image[py_path[odd_points_mask], px_path[odd_points_mask]] = odd_color


    current_point_color = torch.tensor((64, 255, 64),dtype=image.dtype, device=image.device)
    dot_radius = 1 # 3x3
    if len(history_proj) > 0:
        px_curr, py_curr = px[-1], py[-1]
        if (0 <= px_curr < resolution) and (0 <= py_curr < resolution):

            y_start = (py_curr - dot_radius).clamp(0, resolution - 1)
            y_end = (py_curr + dot_radius + 1).clamp(0, resolution)
            x_start = (px_curr - dot_radius).clamp(0, resolution - 1)
            x_end = (px_curr + dot_radius + 1).clamp(0, resolution)
            image[y_start:y_end, x_start:x_end] = current_point_color

    return image

_PointType = int | float | Literal["best"]
class ProjectedFunctionDescent(Benchmark):
    """_summary_

    Args:
        x0 (Any): initial point
        bounds (tuple[float, float] | None, optional): bounds (only for info). Defaults to None.
        make_images (bool, optional): whether to make images. Defaults to True.
        seed (int, optional): seed. Defaults to 0.
        resolution (int, optional): resolution. Defaults to 128.
        smoothing (float, optional): basis smoothing. Defaults to 0.95.
        n_visible (int | None, optional):
            number of last points to consider when determining rendering range. Defaults to 200.
        points (tuple, optional):
            tuple of three things that determine what points are used as the basis. Defaults to ('best', 0.9, 0.95).
        log_scale (bool, optional):
            whether to visualize on log scale
    """
    def __init__(
        self,
        x0,
        bounds: tuple[float, float] | None = None,
        make_images: bool = True,
        seed=0,

        # vis settings
        resolution: int = 128,
        smoothing: float = 0.95,
        n_visible:int | None = 200,

        points: tuple[_PointType,_PointType,_PointType] = ('best', 0.9, 0.95),
        log_scale:bool = False,
    ):
        super().__init__(bounds=bounds, make_images=make_images, seed=seed, log_params=False)

        self._resolution = resolution
        self._x = nn.Parameter(totensor(x0))
        if self._x.ndim != 1: raise RuntimeError(self._x.shape)
        self._smoothing = smoothing

        self._best_params = None
        self._param_history = []
        self._lowest_loss = None

        self._basis = None
        self._shift = None
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None
        self._points = points
        self._n_visible = n_visible
        self._log_scale = log_scale

    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """x is a vector, it can have leading batch dimensions, then loss should have them too"""

    @torch.no_grad
    def _get_point(self, point_type):
        if point_type == 0.: point_type = 0

        if isinstance(point_type, int):
            if 0 <= point_type < len(self._param_history) or 0 < -point_type <= len(self._param_history):
                return self._param_history[point_type]

            return torch.randn_like(self._x)

        if isinstance(point_type, float):
            if point_type > 1 or point_type < -1:
                raise ValueError(point_type)

            if len(self._param_history) == 0: return torch.randn_like(self._x)
            idx = round((len(self._param_history)-1) * point_type)
            return self._param_history[idx]

        if point_type == 'best':
            if self._best_params is None: return torch.randn_like(self._x)
            return self._best_params

        raise ValueError(point_type)

    @torch.no_grad
    def _get_basis(self):
        points = [self._get_point(p) for p in self._points]
        return _make_basis(*points)

    @torch.no_grad
    def _make_frame(self):
        p1, p2, p3 = [self._get_point(p) for p in self._points]
        center = p1
        basis = _make_basis(p1, p2, p3) # (ndim, 2)

        if self._basis is None: self._basis = basis
        else: self._basis.lerp_(basis, 1-self._smoothing)
        basis = self._basis

        history = torch.stack(self._param_history)
        history_proj = (history - center) @ basis # (n, 2)

        visible = history_proj[-self._n_visible:] if self._n_visible is not None else history_proj
        if visible.shape[0] == 0: return torch.zeros(self._resolution, self._resolution) # Handle empty history

        xmin, xmax = visible[:,0].amin(), visible[:,0].amax()
        ymin, ymax = visible[:,1].amin(), visible[:,1].amax()

        xrange = xmax - xmin + 1e-9
        yrange = ymax - ymin + 1e-9
        xmin -= xrange / 2
        xmax += xrange / 2
        ymin -= yrange / 2
        ymax += yrange / 2

        X, Y = torch.meshgrid(
            torch.linspace(xmin, xmax, self._resolution, device=basis.device,),
            torch.linspace(ymin, ymax, self._resolution, device=basis.device,),
            indexing='xy',
        )
        XY_proj = torch.stack([X, Y], -1) # (resolution, resolution, 2)

        grid = XY_proj @ basis.T + center
        Z = self.evaluate(grid) # (resolution, resolution)
        if self._log_scale:
            Z = torch.log10(Z + 1e-12)

        return _draw_trajectory(Z, history_proj, xmin, xmax, ymin, ymax, self._resolution)

    def get_loss(self):
        loss = self.evaluate(self._x)

        if self._make_images:
            x_clone = self._x.detach().clone() # pylint:disable=not-callable
            self._param_history.append(x_clone)


            if self._lowest_loss is None or loss < self._lowest_loss:
                self._lowest_loss = loss.detach()
                self._best_params = x_clone

            frame = self._make_frame()
            self.log_image("landscape", frame, to_uint8=True)

        return loss



class Rosenbrock(ProjectedFunctionDescent):
    """Ill-conditioned banana-shaped function"""
    def __init__(
        self,
        dim=512,
        a=1.0,
        b=100.0,
        pd_fn=torch.square,
        bias: float = 1e-1,

        resolution: int = 128,
        smoothing: float = 0.95,
        points: tuple[_PointType,_PointType,_PointType] = ('best', 0.9, 0.95),
        n_visible:int | None = 200,
        log_scale:bool=True,
    ):
        x0 = torch.tensor([-1.2, 1.]).repeat(dim//2)
        super().__init__(x0, resolution=resolution, smoothing=smoothing, points=points, n_visible=n_visible, log_scale=log_scale)
        self.shift = torch.nn.Buffer(torch.randn(x0.size(), generator=self.rng.torch()) * bias)
        self.pd_fn = pd_fn
        self.a = a
        self.b = b

    def evaluate(self, x):
        x = x + self.shift
        return rosenbrock(x, self.a, self.b, self.pd_fn)



class ChebushevRosenbrock(ProjectedFunctionDescent):
    """Nesterov’s Chebyshev-Rosenbrock Functions."""
    def __init__(self, dim=128, p=10, a=1/4, pd_fn=torch.square, bias:float=1, log_scale:bool=True, **kwargs):
        x0 = torch.tensor([-1.2, 1.]).repeat(dim//2)
        super().__init__(x0, log_scale=log_scale, **kwargs)

        self.bias = torch.nn.Buffer(torch.randn(x0.size(), generator=self.rng.torch()) * bias)
        self.pd_fn = pd_fn
        self.a = a
        self.p = p

    def evaluate(self, x):
        x = x + self.bias
        return chebushev_rosenbrock(x, self.a, self.p, self.pd_fn)


class RotatedQuadratic(ProjectedFunctionDescent):
    """The diabolical quadratic with a hessian that looks like this

    ```python
    tensor([[2, c, c, c],
            [c, 2, c, c],
            [c, c, 2, c],
            [c, c, c, 2]])
    ```

    condition number is (2 + c(dim-1)) / (2 - c).

    This is as rotated as you can get. When `c` is closer to 2, it becomes more ill-conditioned.
    """
    def __init__(self, dim=512, c=1.9999, log_scale:bool=True, **kwargs):
        generator = torch.Generator().manual_seed(0)
        x = torch.nn.Parameter(torch.randn(dim, generator=generator), **kwargs)
        super().__init__(x, log_scale=log_scale)

        self.c = c
        self.b = torch.nn.Buffer(torch.randn(dim, generator=generator, requires_grad=False))

    def evaluate(self, x):
        x = x + self.b
        return rotated_quadratic(x, self.c)


class Rastrigin(ProjectedFunctionDescent):
    """Classic non-convex function with many local minima."""
    def __init__(
        self,
        dim=512,
        A=10.0,
        x0_val: float = 3.0,
        **kwargs
    ):
        x0 = torch.full((dim,), fill_value=x0_val)
        super().__init__(x0, **kwargs)

        self.A = A
        self.dim = dim
        self.bias = nn.Buffer(torch.randn((dim,), generator=self.rng.torch()))

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias
        return rastrigin(x, self.A)


class Ackley(ProjectedFunctionDescent):
    """Another classic non-convex function with many local minima."""
    def __init__(
        self,
        dim=512,
        a=20.0,
        b=0.2,
        c=2 * math.pi,
        x0_val: float = 15.0,
        **kwargs,
    ):
        x0 = torch.full((dim,), fill_value=x0_val)
        super().__init__(x0, **kwargs)
        self.dim = dim
        self.a = a
        self.b = b
        self.c = c
        self.bias = nn.Buffer(torch.randn((dim,), generator=self.rng.torch()))

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias
        return ackley(x, self.a, self.b, self.c)


class BumpyBowl(ProjectedFunctionDescent):
    """Quadratic + rastrigin"""
    def __init__(self, dim=128, A=1.0, k=5.0, bowl_strength=0.01, **kwargs):
        x0 = torch.full((dim,), fill_value=3.0)
        super().__init__(x0, log_scale=True, **kwargs)
        self.A = A
        self.k = k
        self.bowl_strength = bowl_strength
        self.bias = nn.Buffer(torch.randn((dim,), generator=self.rng.torch()))

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias
        bowl_loss = self.bowl_strength * (x**2).sum(-1)
        rastrigin_loss = rastrigin(self.k * x, self.A)
        return bowl_loss + rastrigin_loss


class NNLoss(ProjectedFunctionDescent):
    """Neural network"""
    def __init__(
        self,
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        n_samples=50,
        act = F.relu,
        log_scale:bool=True,
        **kwargs
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.w1_s = (hidden_dim, input_dim)
        self.b1_s = (hidden_dim,)
        self.w2_s = (output_dim, hidden_dim)
        self.b2_s = (output_dim,)

        self.w1_n = hidden_dim * input_dim
        self.b1_n = hidden_dim
        self.w2_n = output_dim * hidden_dim
        self.b2_n = output_dim

        n_params = self.w1_n + self.b1_n + self.w2_n + self.b2_n

        x0 = torch.empty(n_params)

        super().__init__(x0, log_scale=log_scale, **kwargs)

        nn.init.kaiming_uniform_(self._x.view(1,-1), a=math.sqrt(5), generator=self.rng.torch())
        self.X = nn.Buffer(torch.randn(n_samples, input_dim, generator=self.rng.torch()))
        self.y = nn.Buffer(torch.sinc(self.X.norm(dim=-1)).unsqueeze(-1))
        self.act = act

    def _unflatten_params(self, x: torch.Tensor):
        batch_shape = x.shape[:-1] # (*batch_dims, D)
        x_p = x.movedim(-1, 0) # (D, *batch_dims)
        permute_order = list(range(1, x_p.ndim)) + [0]

        w1_flat, x_p = x_p[:self.w1_n], x_p[self.w1_n:]
        b1_flat, x_p = x_p[:self.b1_n], x_p[self.b1_n:]
        w2_flat, x_p = x_p[:self.w2_n], x_p[self.w2_n:]
        b2_flat = x_p[:self.b2_n]

        w1 = w1_flat.permute(permute_order).reshape(*batch_shape, *self.w1_s)
        b1 = b1_flat.permute(permute_order).reshape(*batch_shape, *self.b1_s)
        w2 = w2_flat.permute(permute_order).reshape(*batch_shape, *self.w2_s)
        b2 = b2_flat.permute(permute_order).reshape(*batch_shape, *self.b2_s)

        return w1, b1, w2, b2



    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2 = self._unflatten_params(x)
        h = self.act(torch.einsum('...oi,...ni->...no', w1, self.X) + b1.unsqueeze(-2))
        y_pred = torch.einsum('...oi,...ni->...no', w2, h) + b2.unsqueeze(-2)
        y = self.y.expand_as(y_pred)
        return F.mse_loss(y_pred, y, reduction='none').mean(dim=(-2, -1))

class NNLossSin(NNLoss):
    """NN loss with sin act and smaller init"""
    def __init__(self, **kwargs):
        super().__init__(act=torch.sin, **kwargs)
        with torch.no_grad():
            self._x *= 0.1