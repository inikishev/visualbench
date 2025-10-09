import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence, Callable, Iterable
from typing import Literal, cast, Any

import torch
from torch import nn
from torch.nn import functional as F

from ...benchmark import Benchmark
from ...utils import algebras, format
from ..synthetic import (
    ackley,
    chebushev_rosenbrock,
    rastrigin,
    rosenbrock,
    rotated_quadratic,
)


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
        expand (float, optional):
            how much to expand visualization bounds. 0 means no expanding;
            If size of current visualizaion bounds is ``(x, y)``, the size
            of expanded bounds is ``(x + expand*x, y + expand*y)``.

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
        expand: float = 1,

        points: tuple[_PointType,_PointType,_PointType] = ('best', 0.9, 0.95),
        log_scale:bool = False,
    ):
        super().__init__(bounds=bounds, make_images=make_images, seed=seed, log_params=False)

        self._resolution = resolution
        self._x = nn.Parameter(format.totensor(x0))
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
        self._expand = expand

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

        if self._expand != 0:
            xterm = xrange * (self._expand / 2)
            xmin -= xterm
            xmax += xterm

            yterm = yrange * (self._expand / 2)
            ymin -= yterm
            ymax += yterm

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


class NeuralNet(ProjectedFunctionDescent):
    def __init__(
        self,
        X: Any,
        y: Any,
        widths: Iterable[int] = (2, 16, 1),
        act: Callable = F.relu,
        algebra: Any = None,
        log_scale: bool = True,
        **kwargs
    ):
        widths = list(widths)
        if len(widths) < 2:
            raise ValueError("`widths` must have at least two elements (input and output layers).")

        self.widths = widths
        self.n_layers = len(widths) - 1
        self.input_dim = widths[0]
        self.act = act

        # Calculate shapes and parameter counts for each layer
        self.layer_shapes = []
        self.layer_n_params = []
        n_params = 0
        for i in range(self.n_layers):
            in_dim, out_dim = self.widths[i], self.widths[i+1]
            w_shape, b_shape = (out_dim, in_dim), (out_dim,)
            w_n, b_n = out_dim * in_dim, out_dim
            n_params += w_n + b_n
            self.layer_shapes.append({'w': w_shape, 'b': b_shape})
            self.layer_n_params.append({'w': w_n, 'b': b_n})

        # Initialize parameters and call parent constructor
        x0 = torch.randn(n_params, generator=torch.Generator().manual_seed(0))
        super().__init__(x0, log_scale=log_scale, **kwargs)

        # Use Kaiming initialization for the parameter vector
        nn.init.kaiming_uniform_(self._x.view(1, -1), a=math.sqrt(5), generator=self.rng.torch())

        self.X = nn.Buffer(format.to_HW(X, generator=self.rng.torch(), dtype=torch.float32))
        self.y = nn.Buffer(format.to_HW(y, generator=self.rng.torch(), dtype=torch.float32))
        self.algebra = algebras.get_algebra(algebra)


    def _unflatten_params(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Unflattens the parameter vector `x` into lists of weight matrices and bias vectors.
        """
        batch_shape = x.shape[:-1]
        x_p = x.movedim(-1, 0)
        permute_order = list(range(1, x_p.ndim)) + [0]
        weights, biases = [], []
        current_pos = 0

        for i in range(self.n_layers):
            w_n, b_n = self.layer_n_params[i]['w'], self.layer_n_params[i]['b']
            w_shape, b_shape = self.layer_shapes[i]['w'], self.layer_shapes[i]['b']

            w_flat = x_p[current_pos : current_pos + w_n]
            weights.append(w_flat.permute(permute_order).reshape(*batch_shape, *w_shape))
            current_pos += w_n

            b_flat = x_p[current_pos : current_pos + b_n]
            biases.append(b_flat.permute(permute_order).reshape(*batch_shape, *b_shape))
            current_pos += b_n

        return weights, biases

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass and computes the MSE loss for a given parameter vector `x`.
        """
        weights, biases = self._unflatten_params(x)
        h = self.X

        for i in range(self.n_layers):
            w, b = weights[i], biases[i]
            h = algebras.matmul(h, w.transpose(-1, -2), algebra=self.algebra)
            h = h + b.unsqueeze(-2)

            # Apply activation function for all but the last layer
            if i < self.n_layers - 1:
                h = self.act(h)

        y_pred = h
        y_true = self.y.expand_as(y_pred)
        return F.mse_loss(y_pred, y_true, reduction='none').mean(dim=(-2, -1))

def _symmetrize_tensor(T: torch.Tensor) -> torch.Tensor:
    if T.ndim == 1: return T
    permutations = list(itertools.permutations(range(T.ndim), r=T.ndim))

    T_symm = torch.zeros_like(T)
    for perm in permutations:
        T_symm.add_(T.permute(perm), alpha = 1/len(permutations))

    return T_symm

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
def _poly_eval(x: torch.Tensor, coeffs: list[torch.Tensor], penalty: float):
    val = cast(torch.Tensor, 0)
    for i,T in enumerate(coeffs, 1):
        s1 = ''.join(_LETTERS[:i]) # abcd
        s2 = ',...'.join(_LETTERS[:i]) # a,b,c,d
        # this would make einsum('abcd,a,b,c,d', T, x, x, x, x)
        val += torch.einsum(f"...{s1},...{s2}", T, *(x for _ in range(i))) / math.factorial(i)

    if penalty > 0:
        val += penalty * torch.linalg.vector_norm(x, dim=-1) ** (len(coeffs) + 1) # pylint:disable=not-callable

    return val

class Polynomial(ProjectedFunctionDescent):
    def __init__(
        self,
        dim=10,
        ord=3,
        symmetric: bool = True,
        penalty: float = 1,
        **kwargs
    ):

        x0 = torch.randn(dim, generator=torch.Generator().manual_seed(0))

        super().__init__(x0, log_scale=False, **kwargs)

        self.ord = ord
        self.penalty = penalty

        generator = self.rng.torch()
        coeffs = []
        for i in range(1, ord+1):
            shape = [dim for _ in range(i)]
            T = torch.randn(shape, generator=generator)
            if symmetric: T = _symmetrize_tensor(T)
            coeffs.append(T)

        for i, c in enumerate(coeffs):
            self.register_buffer(f"coeff_{i}", c)


    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = [getattr(self, f"coeff_{i}") for i in range(self.ord)]

        return _poly_eval(x, coeffs, penalty=self.penalty)