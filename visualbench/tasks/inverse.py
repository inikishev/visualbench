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


class MatrixInverse(Benchmark):
    """Finding inverse of a matrix. This supports video rendering.

    Args:
        mat (torch.Tensor):
            square matrix, can have additional first channels dimension which is treated as batch dimension.
        loss (Callable, optional): final loss is `loss(A@B, B@A) + loss(A@B, I) + loss(B@A, I) + loss(diag(B@A), 1) + loss(diag(A@B), 1)`. Defaults to l1.
        dtype (dtype, optional): dtype. Defaults to torch.float32.
        device (Device, optional): device. Defaults to 'cuda'.
    """
    mat: torch.nn.Buffer
    def __init__(self, mat: Any, loss: Callable = l1, dtype: torch.dtype=torch.float32):
        mat = to_float_hwc_tensor(mat).moveaxis(-1, 0)
        if mat.shape[-1] != mat.shape[-2]: raise ValueError(f'{mat.shape = } - not a matrix!')
        mat = mat.to(dtype = dtype, memory_format = torch.contiguous_format)
        self.loss_fn = loss

        mat_reference = normalize_to_uint8(mat)
        labels = ['input']

        try:
            true_inv = torch.linalg.inv(mat).cpu().numpy().astype(np.uint8) # pylint:disable=not-callable
            labels.append('true inverse')
        except torch.linalg.LinAlgError as e:
            true_inv = torch.linalg.pinv(mat) # pylint:disable=not-callable
            labels.append('pseudoinverse')

        true_inv_reference = normalize_to_uint8(true_inv)

        super().__init__(reference_images = [mat_reference, true_inv_reference], reference_labels = labels, save_edge_params = True, seed=0)
        self.register_buffer('mat', mat)
        self.inverse = torch.nn.Parameter(self.mat.clone().requires_grad_(True))

    def get_loss(self):
        AB = self.mat @ self.inverse
        BA = self.inverse @ self.mat
        I = torch.eye(self.mat.shape[-1], device = AB.device, dtype=AB.dtype)
        I_diag = torch.ones(BA.shape[-1], device = AB.device, dtype=AB.dtype)
        loss = self.loss_fn(AB, BA)  +\
            self.loss_fn(AB, I) +\
            self.loss_fn(BA, I) +\
            self.loss_fn(BA.diagonal(0,-2,-1), I_diag) +\
            self.loss_fn(AB.diagonal(0,-2,-1), I_diag)

        #print(list(self.parameters()))
        return loss, {"image_output": normalize_to_uint8(self.inverse), "image_AB": normalize_to_uint8(AB), "image_BA": normalize_to_uint8(BA),}