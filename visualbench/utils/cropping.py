from collections.abc import Sequence
from typing import Literal,overload
import torch, numpy as np

@overload
def crop(input:torch.Tensor, reduction:Sequence[int], where: Literal["center", "start", "end"] = "center",) -> torch.Tensor: ...
@overload
def crop(input:np.ndarray, reduction:Sequence[int], where: Literal["center", "start", "end"] = "center",) -> np.ndarray: ...
def crop(input:torch.Tensor|np.ndarray, reduction:Sequence[int], where: Literal["center", "start", "end"] = "center",):
    """Crop `input` using `crop`:

    `output.shape[i]` = `input.shape[i] - crop[i]`.

    """
    if where == 'center':
        slices = [(int(i / 2), -int(i / 2)) if i % 2 == 0 else (int(i / 2), -int(i / 2) - 1) for i in reduction]
    elif where == 'start':
        slices = [(None, -i) for i in reduction]
    elif where == 'end':
        slices = [(i, None) for i in reduction]

    slices = [slice(i if i!=0 else None, j if j != 0 else None) for i, j in slices]

    return input[(..., *slices)]

@overload
def crop_to_shape(input:torch.Tensor, shape:Sequence[int], where: Literal["center", "start", "end"] = "center",) -> torch.Tensor: ...
@overload
def crop_to_shape(input:np.ndarray, shape:Sequence[int], where: Literal["center", "start", "end"] = "center",) -> np.ndarray: ...
def crop_to_shape(input:torch.Tensor | np.ndarray, shape:Sequence[int], where: Literal["center", "start", "end"] = "center",):
    """Crop `input` to `shape`."""
    return crop(input, [i - j for i, j in zip(input.shape, shape)], where=where)

def crop_like(input:torch.Tensor, target:torch.Tensor, where: Literal["center", "start", "end"] = "center",):
    """Crop `input` to `target.shape`."""
    return crop_to_shape(input, target.shape, where=where)

# def spatial_crop(x:torch.Tensor, amount:int = 1):
#     """Crops spatial dim sizes in a BC* tensor by `amount`.
#     For example, if `amount = 1`, (16, 3, 129, 129) -> (16, 3, 128, 128).
#     This crops at the end. Useful to crop padding which can only add even size."""
#     slices = [slice(None, -amount) for _ in range(x.ndim - 2)]
#     return x[:,:,*slices]
