import copy
from collections.abc import Iterable, Callable
from typing import Any

import numpy as np
import torch

CUDA_IF_AVAILABLE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def copy_state_dict(state: torch.nn.Module | dict[str, Any], device=None):
    """clones tensors and ndarrays, recursively copies dicts, deepcopies everything else, also moves to device if it is not None"""
    if isinstance(state, torch.nn.Module): state = state.state_dict()
    c = state.copy()
    for k,v in state.items():
        if isinstance(v, torch.Tensor):
            if device is not None: v = v.to(device)
            c[k] = v.clone()
        if isinstance(v, np.ndarray): c[k] = v.copy()
        elif isinstance(v, dict): c[k] = copy_state_dict(v)
        else:
            if isinstance(v, torch.nn.Module) and device is not None: v = v.to(device)
            c[k] = copy.deepcopy(v)
    return c


def normalize(x: torch.Tensor, min=0, max=1) -> torch.Tensor:
    x = x.float()
    x = x - x.min()
    xmax = x.max()
    if xmax != 0: x /= xmax
    else: return x
    return x * (max - min) + min


def znormalize(x:torch.Tensor, mean=0., std=1.) -> torch.Tensor:
    xstd = x.std()
    if xstd != 0: return ((x - x.mean()).div_(xstd / std)).add_(mean)
    return x - x.mean()


def count_learnable_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@torch.no_grad
def apply_jet_cmap(HW: torch.Tensor) -> torch.Tensor:
    """fast jet cmap maker HW must be between 0 and 1 THIS RETURNS UINT8 TENSOR"""
    if HW.ndim > 2: HW = HW.squeeze()
    if HW.ndim > 2: raise ValueError(f"{HW.shape = }")
    val = torch.clamp(HW, 0.0, 1.0)
    r = torch.clamp(1.5 - torch.abs(val - 0.75) * 4.0, 0.0, 1.0)
    g = torch.clamp(1.5 - torch.abs(val - 0.5) * 4.0, 0.0, 1.0)
    b = torch.clamp(1.5 - torch.abs(val - 0.25) * 4.0, 0.0, 1.0)
    return torch.stack([r, g, b], dim=-1).to(torch.uint8).detach().cpu()

@torch.no_grad
def apply_overflow_cmap(HW: torch.Tensor) -> torch.Tensor:
    """HW is between 0 and 1 but it can overflow. overflow below is blue and above is red. otherwise it goes black to white THIS RETURNS UINT8 TENSOR"""
    # ANOTHER IDEA
    # -4 PINK
    # -3 RED
    # -2 ORANGE
    # -1 YELLOW
    # 0 BLACK
    # 1 WHITE
    # 2 LIGHT BLUE?
    # 3 BLUE
    # 4 GREEN
    if HW.ndim > 2: HW = HW.squeeze()
    if HW.ndim > 2: raise ValueError(f"{HW.shape = }")
    frame = HW[:,:,None].repeat_interleave(3, 2)
    red_overflow = (frame - 1).clip(min=0)
    red_overflow[:,:,1:] *= 2
    blue_overflow = - frame.clip(max=0)
    blue_overflow[:,:,2] *= 2
    frame = ((frame - red_overflow + blue_overflow) * 255).clip(0,255).to(torch.uint8).detach().cpu()
    return frame

def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].view_as(r))
        cur += numel
    return tensors


def maybe_per_sample_loss(loss: Callable, args: tuple, per_sample=False):
    if per_sample: return loss(*args, reduction='none').ravel()
    return loss(*args)