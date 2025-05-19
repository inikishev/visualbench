import importlib.util
from typing import TYPE_CHECKING, overload
import torch
if TYPE_CHECKING:
    import torchalgebras as ta
    from torchalgebras.base import MaybeAlgebraicTensor

def get_algebra(algebra: "str | ta.Algebra | None"):
    if algebra is None: return None

    try: import torchalgebras as ta
    except ModuleNotFoundError: raise ModuleNotFoundError("Specifying an algebra requires `torchalgebras` installed") from None

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