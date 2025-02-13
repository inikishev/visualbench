from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from myai.transforms import normalize

from .. import Benchmark
from ..utils import to_float_hwc_tensor, CUDA_IF_AVAILABLE


@torch.no_grad
def normalize_to_uint8(x:torch.Tensor | np.ndarray):
    if isinstance(x, np.ndarray): return normalize(x, 0, 255).astype(np.uint8)
    return normalize(x.detach(), 0, 255).cpu().numpy().astype(np.uint8)

def l1(x,y):
    return (x-y).abs().mean()
def l2(x,y):
    return (x-y).pow(2).mean()


class Sphere(Benchmark):
    """Basic vector restoration benchmark (n-dimensional tensor with MSE loss)

    Args:
        target (torch.Tensor): any target but to plot pass an image.
        loss (Callable, optional): final loss is `loss(A@B, B@A) + loss(A@B, I) + loss(B@A, I) + loss(diag(B@A), 1) + loss(diag(A@B), 1)`. Defaults to l1.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cuda'.
    """
    target: torch.nn.Buffer
    def __init__(self, target: Any, loss: Callable = l2, dtype: torch.dtype=torch.float32):
        self.loss_fn = loss
        target = to_float_hwc_tensor(target).to(dtype = dtype, memory_format = torch.contiguous_format)
        mat_reference = normalize_to_uint8(target)

        super().__init__(reference_images = mat_reference, seed=0)
        self.register_buffer('target', target)
        self.preds = torch.nn.Parameter(self.target.clone().zero_().requires_grad_(True))

        self.target_min = self.target.min().item()
        self.target_max = self.target.max().item()

    def get_loss(self):
        loss = self.loss_fn(self.preds, self.target)

        #print(list(self.parameters()))
        return loss, {"image": (normalize(self.preds, 0, 255)).clamp(0,255).detach().cpu().numpy().astype(np.uint8)}