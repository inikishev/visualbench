import warnings
from os import PathLike
from typing import Any

import numpy as np
import torch

from .image import _imread


def _imread_normalize(x) -> torch.Tensor:
    x = _imread(x).float()
    x -= x.mean()
    std = x.std()
    if std != 0: x /= std
    return x.contiguous()

def generic_numel(x: np.ndarray | torch.Tensor) -> int:
    if isinstance(x, torch.Tensor): return x.numel()
    return x.ndim

def is_scalar(x: Any) -> bool:
    if isinstance(x, (np.ndarray, torch.Tensor)): return generic_numel(x) == 1
    return isinstance(x, (int,float,bool))

def totensor(x, device=None, dtype=None, clone=None) -> torch.Tensor:
    if isinstance(x, (str, PathLike)): x = _imread_normalize(x)

    if isinstance(x, torch.Tensor): x = x.to(dtype=dtype, device=device, copy=False)

    elif isinstance(x, np.ndarray):
        if clone is None: x = x.copy()
        x = torch.from_numpy(x).to(dtype=dtype, device=device, copy=False)

    elif isinstance(x, (int,float,bool)):
        x = torch.tensor(x, device=device, dtype=dtype)

    else:
        x = np.asarray(x)
        if clone is None: x = x.copy()
        x = torch.from_numpy(x).to(dtype=dtype, device=device, copy=False)

    if clone: x = x.clone()
    return x

def tonumpy(x) -> np.ndarray:
    if isinstance(x, (str, PathLike)): x = _imread_normalize(x)
    if isinstance(x, np.ndarray): return x
    if isinstance(x, torch.Tensor): return x.numpy(force=True)
    return np.asarray(x)

def tofloat(x) -> float:
    if isinstance(x, torch.Tensor): return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

def maybe_tofloat(x):
    if isinstance(x, (np.ndarray, torch.Tensor)) and generic_numel(x) == 1: return tofloat(x)
    return x


def to_CHW(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    if isinstance(x, int): return torch.randn(1,x,x, device=device,dtype=dtype, generator=generator)
    if isinstance(x, (list,tuple)):
        if len(x) == 2 and all(isinstance(i,int) for i in x): return torch.randn((1,*x), device=device,dtype=dtype, generator=generator)
        if len(x) == 3 and all(isinstance(i,int) for i in x): return torch.randn(x, device=device,dtype=dtype, generator=generator)

    x = totensor(x, device=device, dtype=dtype, clone=clone)

    if x.ndim > 3:
        x = x.squeeze()
        if x.ndim > 3:
            raise RuntimeError(f"Too many dimensions {x.shape}")

    if x.ndim == 2: return x.unsqueeze(0)

    if x.size(0) > x.size(-1): return x.moveaxis(-1, 0)
    return x

def to_HWC(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    return to_CHW(x, device=device, dtype=dtype, clone=clone, generator=generator).moveaxis(0, -1)

def to_3HW(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    x = to_CHW(x, device=device, dtype=dtype, clone=clone, generator=generator)

    if x.size(0) == 3:
        return x

    if x.size(0) > 3:
        warnings.warn(f"clipping {x.shape} to 3HW format {x[:3].shape}")
        return x[:3]

    if x.size(0) == 1:
        return x.repeat_interleave(3, 0)

    if x.size(0) == 2:
        return torch.cat([x, x.float().mean(0).type_as(x).unsqueeze(0)])

    raise RuntimeError(f"wtf {x.shape}")

def to_HW3(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    return to_3HW(x, device=device, dtype=dtype, clone=clone, generator=generator).moveaxis(0, -1)

def to_HW(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    return to_CHW(x, device=device, dtype=dtype, clone=clone, generator=generator).mean(0)

def to_square(x, device=None, dtype=None, clone=None, generator=None) -> torch.Tensor:
    """makes x square among last two dimensions"""
    if isinstance(x, int): return torch.randn(x,x, device=device,dtype=dtype, generator=generator)

    x = totensor(x, device=device, dtype=dtype, clone=clone)
    if x.ndim < 2: raise RuntimeError(f"Not enough dims in {x.shape} to make it square")
    *_, m, n = x.shape

    if m == n: return x
    if m > n: return x[..., :n, :]
    if n > m: return x[..., :, :m]

    raise RuntimeError(f"wtf {x.shape}")

def normalize_to_uint8(x, min=None, max=None) -> torch.Tensor:
    x = totensor(x).detach().float()
    if min is None: min = x.min()
    if max is None: max = x.max()

    x = x - min # 1st not in-place
    max = max - min
    if max != 0:
        x /= max/255 # we need 0-255 range

    x.clip_(0, 255)
    return x.to(torch.uint8)

