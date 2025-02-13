import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from myai.transforms import normalize, znormalize

from ..benchmark import Benchmark
from ..utils import CUDA_IF_AVAILABLE, to_float_hwc_tensor


def create_polynomial_image(coefficients: torch.Tensor, x_range, y_range, step):
    """
    Creates an image representing the polynomial values over a grid of x, y coordinates.

    Args:
        coefficients (torch.Tensor): n-dimensional tensor of polynomial coefficients.
                                     For nth order, it's a tensor of shape (2, 2, ..., 2) n times.
        x_range (tuple): Tuple of (x_min, x_max) for the x-axis range.
        y_range (tuple): Tuple of (y_min, y_max) for the y-axis range.
        step (float): Step size for both x and y axes.

    Returns:
        torch.Tensor: 2D tensor representing the image, where each element is the polynomial value.
    """
    x = torch.arange(x_range[0], x_range[1] + step, step, device=coefficients.device, dtype=coefficients.dtype)
    y = torch.arange(y_range[0], y_range[1] + step, step, device=coefficients.device, dtype=coefficients.dtype)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    image = torch.zeros_like(X, dtype=torch.float32)
    order = coefficients.ndim

    index_options = [0, 1]
    for indices_tuple in itertools.product(index_options, repeat=order):
        indices = list(indices_tuple)
        coef_val = coefficients[tuple(indices)]

        term = torch.ones_like(X, dtype=torch.float32)
        for index_val in indices:
            if index_val == 0:
                term = term * X
            else: # index_val == 1
                term = term * Y
        image = image + coef_val * term

    return image


@torch.no_grad
def normalize_to_uint8(x:torch.Tensor | np.ndarray):
    if isinstance(x, np.ndarray): return normalize(x, 0, 255).astype(np.uint8)
    return normalize(x.detach(), 0, 255).cpu().numpy().astype(np.uint8)

def l1(x,y):
    return (x-y).abs().mean()
def l2(x,y):
    return (x-y).pow(2).mean()



class PolynomialReconstructor(Benchmark):
    """fits a polynomial to an image.

    Args:
        mat (torch.Tensor):
            image bw or colored.
        loss (Callable, optional): final loss is `loss(A@B, B@A) + loss(A@B, I) + loss(B@A, I) + loss(diag(B@A), 1) + loss(diag(A@B), 1)`. Defaults to l1.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cuda'.
    """
    target: torch.nn.Buffer
    def __init__(self, target: Any, order: int = 4, loss: Callable = l2, dtype: torch.dtype=torch.float32):
        self.loss_fn = loss
        target = znormalize(to_float_hwc_tensor(target).to(dtype = dtype, memory_format = torch.contiguous_format).moveaxis(-1, 0))
        mat_reference = normalize_to_uint8(target.moveaxis(0, -1))

        super().__init__(reference_images = mat_reference, seed=0)
        self.register_buffer('target', target)

        shape = [target.shape[0]] + [2] * order
        self.coeffs = torch.nn.Parameter(torch.zeros(shape, dtype=dtype))

        self.shape = target.shape[1:]
        self.val = max(self.shape)

    def get_loss(self):
        preds = torch.stack([create_polynomial_image(c, (0, self.shape[1]/self.val), (0, self.shape[0]/self.val), 1/self.val)[:self.shape[0], :self.shape[1]] for c in self.coeffs])
        loss = self.loss_fn(preds, self.target)

        return loss, {"image": (normalize(preds, 0, 255)).clamp(0,255).detach().cpu().numpy().astype(np.uint8)}
