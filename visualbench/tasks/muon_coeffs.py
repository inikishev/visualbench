import math

import numpy as np
import sympy as sp
import torch
from PIL import Image, ImageDraw
from torch import nn

from ..benchmark import Benchmark
from ..utils import tonumpy, totensor

# this code is https://leloykun.github.io/ponder/muon-opt-coeffs/ ported to pytorch
# """
# Tool for optimizing the coefficients of the Newton-Schulz iterators in Muon.

# Usage notes:
# - Use a high `epsilon` value to prevent the singular values from either blowing up or switching signs.
# - Set --enable_flatness_aux_loss to get flatter composite curves.
# """

def glr_iterator_torch(x: torch.Tensor, gamma: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    fp0 = -(1 + r)
    fp1 = -(1 - l)
    fp2 = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    fp3 = 1 - l
    fp4 = 1 + r

    fp = torch.stack([fp0, fp1, fp2, fp3, fp4])

    if x.ndim > 0: product_term = torch.prod(x.unsqueeze(-1) - fp, dim=-1)
    else: product_term = torch.prod(x - fp)

    iterator = x + gamma * product_term
    return iterator

def a_torch(gamma: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return 1.0 + gamma * (1.0 - l)**2 * (1.0 + r)**2

def b_torch(gamma: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return -gamma * ((1.0 - l)**2 + (1.0 + r)**2)

def c_torch(gamma: torch.Tensor, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return gamma

def abc_iterator(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * x + b * x**3 + c * x**5

def glr_to_abc_reparam(gamma: torch.Tensor, l: torch.Tensor, r: torch.Tensor, decimals: int = 4):
    abc = torch.stack([a_torch(gamma, l, r), b_torch(gamma, l, r), c_torch(gamma, l, r)])
    return abc + (torch.round(abc, decimals=decimals) - abc).detach()

x_ = sp.Symbol("x", real=True)
def abc_to_glr_reparam(a: float, b: float, c: float, verbose: bool = False):
    iterator_fn = a*x_ + b*x_**3 + c*x_**5 # type:ignore
    iterator_roots = sp.nroots(iterator_fn - x_)
    if verbose:
        print(iterator_roots)
    iterator_roots_real = [root.evalf() for root in iterator_roots if root.is_real]
    iterator_roots = sorted(iterator_roots_real)
    return float(c), float(1 - iterator_roots[-2]), float(iterator_roots[-1] - 1)


# https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
def scan(f, init, xs, length=None):
    if xs is None:
        assert length is not None
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)

def loss(
    x: torch.Tensor,
    params: torch.Tensor,
    eps: float,
    precision: int,
):
    def scan_body_fn(y: torch.Tensor, glr: torch.Tensor):
        gamma, l, r = glr

        # The peak of the previous iteration should be at most 1 + r - eps
        # to prevent singular values from blowing up
        intermediate_loss = torch.clip(y.max() - (1 + r - eps), min=0)

        a, b, c = glr_to_abc_reparam(gamma, l, r, precision)
        new_y = abc_iterator(y, a, b, c)

        # The iterator must not cross the x-axis
        # to prevent singular values from switching signs
        intermediate_loss += torch.clip(eps - torch.amin(torch.where(y > 0.5, new_y, torch.inf)), min=0)

        return new_y, intermediate_loss
    y, intermediate_losses = scan(scan_body_fn, x, params)

    # This auxiliary loss term encourages the contraction of the
    # attractor basins of the iterators
    aesthetic_aux_loss = (
        torch.clip(params[1:,2] - params[:-1,2], min=0).sum()
        + torch.clip(params[1:,1] - params[:-1,1], min=0).sum()
        + torch.clip(params[1:,0] - params[:-1,0], min=0).sum()
    )

    # This auxiliary loss term encourages the flatness of the composite curve
    # Taken from @YouJiacheng's code here: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b
    y_max = torch.amax(y)
    y_min = torch.amin(torch.where(x > 0.05, y, torch.inf))
    diff_ratio = (y_max - y_min) / torch.clip(y_max, min=1e-3)


    loss1 = torch.sqrt(torch.mean((y - 1) ** 2))
    # loss2 = (
    #     intermediate_losses.mean()
    #     + enable_contraction_aux_loss * aesthetic_aux_loss
    #     + enable_flatness_aux_loss * diff_ratio
    # )
    return loss1, intermediate_losses, aesthetic_aux_loss, diff_ratio

# VISUALIZATION STUFF
def _map_coords(x_data, y_data, viewport, data_range):
    """Maps data coordinates to pixel coordinates within a viewport."""
    x_vp, y_vp, w_vp, h_vp = viewport
    x_min, x_max, y_min, y_max = data_range

    mask = (x_data >= x_min) & (x_data <= x_max)
    x_filt, y_filt = x_data[mask], y_data[mask]

    if len(x_filt) < 2:
        return []

    if np.any(~np.isfinite(y_filt)):
        return [[1.,2.],[3.,4.]]

    denom = (x_max - x_min)
    if abs(denom) < 1e-16: denom = 1
    px = x_vp + (x_filt - x_min) / denom * w_vp

    denom = (y_max - y_min)
    if abs(denom) < 1e-16: denom = 1
    py = y_vp + h_vp - ((y_filt - y_min) / denom * h_vp)

    py = np.clip(py, y_vp, y_vp + h_vp)
    return list(zip(px, py))

def _make_frame(
    x_coords: np.ndarray,
    intermediate_ys: list[np.ndarray],
    final_y_kj: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    pad = (width//40) + (height//40)
    w_plot, h_plot = (width - 3*pad) // 2, (height - 3*pad) // 2
    vp_tl = (pad, pad, w_plot, h_plot)
    vp_tr = (width - pad - w_plot, pad, w_plot, h_plot)
    vp_bl = (pad, height - pad - h_plot, w_plot, h_plot)
    vp_br = (width - pad - w_plot, height - pad - h_plot, w_plot, h_plot)

    range_tl = [x_coords.min(), x_coords.max(), -0.1, 1.3]
    range_tr = [0.0, 1.0, 0.6, 1.25]
    range_bl = [0.0, 0.01, 0.6, 1.25]
    range_br = [0.0, 0.001, 0.0, 0.5]

    final_y_optimized = intermediate_ys[-1]

    x_line = np.array(range_tl[:2])
    points = _map_coords(x_line, x_line, vp_tl, range_tl)
    if points: draw.line(points, fill=(220, 220, 220), width=1) # y=x line # type:ignore

    num_iters = len(intermediate_ys) - 1
    for i in range(1, len(intermediate_ys)):
        fraction = (i - 1) / max(1, num_iters - 1) # later iters are brighter
        r = int(173 * (1 - fraction) + 0 * fraction)
        g = int(216 * (1 - fraction) + 84 * fraction)
        b = int(230 * (1 - fraction) + 180 * fraction)
        color = (r, g, b)

        points = _map_coords(x_coords, intermediate_ys[i], vp_tl, range_tl)
        if points: draw.line(points, fill=color, width=1, joint='curve') # type:ignore

    for vp, data_range in [(vp_tr, range_tr), (vp_bl, range_bl), (vp_br, range_br)]:
        # Keller-Jordan curve (black)
        points_kj = _map_coords(x_coords, final_y_kj, vp, data_range)
        if points_kj: draw.line(points_kj, fill='black', width=1, joint='curve') # type:ignore

        # MY CURVE (blue)
        points_opt = _map_coords(x_coords, final_y_optimized, vp, data_range)
        if points_opt: draw.line(points_opt, fill=(0, 84, 180), width=1, joint='curve') # type:ignore

    return np.array(img, dtype=np.uint8)


class MuonCoeffs(Benchmark):
    """https://leloykun.github.io/ponder/muon-opt-coeffs/ ported to pytorch

    ``evaluate_self`` to evaluate best coeffs found so far.
    """
    def __init__(self, num_ns_iters=5, precision: int = 4, eps: float = 1/16, w_steepness: float=1, w_stability:float=1, w_contraction:float=1, w_flatness:float = 0, resolution = (384, 384)):
        super().__init__()

        # all code is https://leloykun.github.io/ponder/muon-opt-coeffs/
        # Reparametrize Keller Jordan's a-b-c coefficients to gamma-l-r
        kj_a, kj_b, kj_c = 3.4445, -4.7750, 2.0315
        kj_gamma, kj_inner_radius, kj_outer_radius = abc_to_glr_reparam(kj_a, kj_b, kj_c)
        # Check if the reparametrization is correct
        kj_abc = glr_to_abc_reparam(*torch.tensor([kj_gamma, kj_inner_radius, kj_outer_radius]), decimals=precision)
        assert torch.allclose(kj_abc, torch.tensor([kj_a, kj_b, kj_c]), atol=1e-4)

        x = torch.cat([
            # The extra 0.1 is there to account for numerical instability
            torch.linspace(0, 1.1, 2**10),
            # Gradients typically have low stable rank (i.e. most of the singular values are close to 0).
            # To simulate that, we add a couple more points near 0.
            torch.linspace(0, 0.1, 2**9),
        ])
        self.x = nn.Buffer(x)

        init_params = torch.tensor([[kj_gamma, kj_inner_radius, kj_outer_radius]]*num_ns_iters, requires_grad=True)
        self.params = nn.Parameter(init_params)

        self.precision = precision
        self.eps = eps

        self.w_steepness = w_steepness
        self.w_stability = w_stability
        self.w_contraction = w_contraction
        self.w_flatness = w_flatness

        self.resolution = resolution

        # pre-compute the Keller-Jordan 5-steps curve for visualization
        with torch.no_grad():
            kj_params_tensor = torch.tensor([[kj_gamma, kj_inner_radius, kj_outer_radius]]*num_ns_iters)
            y_kj = self.x.clone()
            for p in kj_params_tensor:
                gamma, l, r = p
                a, b, c = glr_to_abc_reparam(gamma, l, r, self.precision)
                y_kj = abc_iterator(y_kj, a, b, c)
            self.y_kj = nn.Buffer(y_kj)

        self._show_titles_on_video = False
        self.set_multiobjective_func(torch.sum)

    def get_loss(self):
        steepness_loss, intermediate_losses, aesthetic_aux_loss, diff_ratio = loss(x=self.x, params=self.params, eps=self.eps, precision=self.precision)
        self.log("loss steepness", steepness_loss)

        stability_loss = intermediate_losses.mean()
        self.log("loss flatness", stability_loss)

        self.log("loss aux contraction", aesthetic_aux_loss)
        self.log("loss aux flatness", diff_ratio)

        with torch.no_grad():
            steepness = 1
            for gamma, l, r in self.params:
                a, b, c = glr_to_abc_reparam(gamma, l, r, self.precision)
                steepness *= a

            self.log('steepness', steepness)

        if self._make_images:
            with torch.no_grad():
                intermediate_ys_list: list[torch.Tensor] = [self.x]
                y = self.x
                for glr in self.params:
                    gamma, l, r = glr
                    a, b, c = glr_to_abc_reparam(gamma, l, r, self.precision)
                    y = abc_iterator(y, a, b, c)
                    intermediate_ys_list.append(y)

                x_np = self.x.cpu().numpy()
                intermediate_ys_np = [t.cpu().numpy() for t in intermediate_ys_list]
                y_kj_np = self.y_kj.cpu().numpy()

                frame = _make_frame(
                    x_coords=x_np,
                    intermediate_ys=intermediate_ys_np,
                    final_y_kj=y_kj_np,
                    width=self.resolution[0], height=self.resolution[1],
                )
                self.log_image("image", frame, to_uint8 = False, show_best=True)

        return torch.stack([
            self.w_steepness*steepness_loss,
            self.w_stability*stability_loss,
            self.w_contraction*aesthetic_aux_loss,
            self.w_flatness*diff_ratio
        ])

    @torch.no_grad
    def best_coeffs(self):
        best_params = self.best_params()[0]
        return torch.stack([totensor(glr_to_abc_reparam(*p, decimals=self.precision)) for p in best_params])

    @torch.no_grad
    def evaluate_params(self, params):
        """returns (steepness_loss, intermediate_losses, stability_loss, contraction_loss, flatness_loss, steepness)"""
        params = totensor(params).to(self.x)
        if params.ndim > 2: params = params.squeeze()
        if params.shape != self.params.shape: raise RuntimeError(f"{params.shape = }, {self.params.shape = }")

        steepness_loss, intermediate_losses, aesthetic_aux_loss, diff_ratio = loss(x=self.x, params=params.to(self.x), eps=self.eps, precision=self.precision)

        steepness = 1
        for gamma, l, r in params:
            a, b, c = glr_to_abc_reparam(gamma, l, r, self.precision)
            steepness *= a

        return steepness_loss, intermediate_losses, intermediate_losses.mean(), aesthetic_aux_loss, diff_ratio, steepness

    @torch.no_grad
    def evaluate_coeffs(self, coeffs):
        """returns (steepness_loss, intermediate_losses, stability_loss, contraction_loss, flatness_loss, steepness)"""
        coeffs = tonumpy(coeffs)
        if coeffs.shape != self.params.shape: raise RuntimeError(f"{coeffs.shape = }, {self.params.shape = }")

        params = torch.stack([torch.tensor(abc_to_glr_reparam(*c)) for c in coeffs])

        # Check if the reparametrization is correct
        kj_abc = torch.stack([glr_to_abc_reparam(*p, decimals=self.precision) for p in params])
        assert torch.allclose(totensor(coeffs).to(kj_abc), kj_abc, atol=1e-4)

        return self.evaluate_params(params)

    @torch.no_grad
    def evaluate_self(self):
        """returns (steepness_loss, intermediate_losses, stability_loss, contraction_loss, flatness_loss, steepness)"""
        best = self.best_params()
        return self.evaluate_params(best)