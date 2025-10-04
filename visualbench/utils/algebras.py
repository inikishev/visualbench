import importlib.util
from typing import TYPE_CHECKING, overload

import torch

from . import torchalgebras as ta
from .torchalgebras.base import MaybeAlgebraicTensor


def get_algebra(algebra: "str | ta.Algebra | None"):
    if algebra is None: return None
    return ta.get_algebra(algebra)

def _to_tensor(x):
    if isinstance(x, torch.Tensor): return x
    if not hasattr(x, "algebra"): raise TypeError(type(x))
    return x.data

@overload
def from_algebra(tensor1: "MaybeAlgebraicTensor") -> torch.Tensor: ...
@overload
def from_algebra(tensor1: "MaybeAlgebraicTensor"  , tensor2: "MaybeAlgebraicTensor", *tensors: "MaybeAlgebraicTensor") -> list[torch.Tensor]: ...
def from_algebra(tensor1: "MaybeAlgebraicTensor", tensor2: "MaybeAlgebraicTensor | None" = None, *tensors: "MaybeAlgebraicTensor") -> torch.Tensor | list[torch.Tensor]:
    torch_tensors = [_to_tensor(t) for t in (tensor1, tensor2, *tensors) if t is not None]
    if len(torch_tensors) == 1: return torch_tensors[0]
    return torch_tensors

def mul(x:torch.Tensor, y:torch.Tensor, algebra):
    if algebra is None: return x * y
    return algebra.mul(x, y)

def matmul(x:torch.Tensor, y:torch.Tensor, algebra: "ta.Algebra | None"):
    if algebra is None: return x @ y
    return algebra.matmul(x, y)

def dot(x:torch.Tensor, y:torch.Tensor, algebra: "ta.Algebra | None"):
    if algebra is None: return x.dot(y)
    return algebra.dot(x, y)

def outer(x:torch.Tensor, y:torch.Tensor, algebra: "ta.Algebra | None"):
    if algebra is None: return x.outer(y)
    return algebra.outer(x, y)

def kron(x:torch.Tensor, y:torch.Tensor, algebra: "ta.Algebra | None"):
    if algebra is None: return x.kron(y)
    return algebra.kron(x, y)

def sum(x:torch.Tensor, algebra: "ta.Algebra | None", dim=None, keepdim=False):
    if algebra is None: return torch.sum(x, dim=dim, keepdim=keepdim)
    return algebra.sum(x, dim=dim, keepdim=keepdim)