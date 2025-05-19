import copy
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