from collections.abc import Sequence
from typing import Literal, overload

import torch
import numpy as np
from .python_tools import reduce_dim
from .cropping import crop as _crop

@overload
def _pad(input: torch.Tensor,padding: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center") -> torch.Tensor: ...
@overload
def _pad(input: np.ndarray,padding: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center") -> np.ndarray: ...
@torch.no_grad
def _pad(
    input: torch.Tensor | np.ndarray,
    padding: Sequence[int],
    mode: str = "constant",
    value=None,
    where: Literal["center", "start", "end"] = "center",
) -> np.ndarray | torch.Tensor:
    # create padding sequence for torch.nn.functional.pad
    if where == 'center':
        dims_padding = [(int(i / 2), int(i / 2)) if i % 2 == 0 else (int(i / 2), int(i / 2) + 1) for i in padding]
    elif where == 'start':
        dims_padding = [(i, 0) for i in padding]
    elif where == 'end':
        dims_padding = [(0, i) for i in padding]
    else: raise ValueError(f'Invalid where: {where}')

    # broadcasting (e.g. if padding 3×128×128 by [4, 4], it will pad by [0, 4, 4])
    if len(dims_padding) < input.ndim:
        dims_padding = [(0, 0)] * (input.ndim - len(dims_padding)) + dims_padding

    if mode == 'zeros': mode = 'constant'; value = 0
    elif mode == 'min': mode = 'constant'; value = float(input.min())
    elif mode == 'max': mode = 'constant'; value = float(input.max())
    elif mode == 'mean': mode = 'constant'; value = float(input.mean())

    if isinstance(input, np.ndarray):
        if mode == 'constant': kwargs = {"constant_values": value}
        else: kwargs = {}
        return np.pad(input, pad_width = dims_padding, mode = mode, **kwargs) # type:ignore
    return torch.nn.functional.pad(input, reduce_dim(reversed(dims_padding)), mode=mode, value=value)

@overload
def pad(input: torch.Tensor,padding: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center",crop = True) -> torch.Tensor: ...
@overload
def pad(input: np.ndarray,padding: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center",crop = True) -> np.ndarray: ...
def pad(
    input: torch.Tensor | np.ndarray,
    padding: Sequence[int],
    mode: str = "constant",
    value=None,
    where: Literal["center", "start", "end"] = "center",
    crop = True,
) -> np.ndarray | torch.Tensor:
    """
    Padding function that is easier to read:

    `output.shape[i]` = `input.shape[i] + padding[i]`.

    Args:
        input (torch.Tensor): input to pad.
        padding (str): how much padding to add per each dimension of `input`.
        mode (str, optional): padding mode (https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html). Defaults to 'constant'.
        value (_type_, optional): padding constant value. Defaults to None.
        where (str, optional): where to pad.
            if `center`, will pad start and end of each dimension evenly,
            if `start`, will pad at the start of each dimension,
            if `end`, will pad at the end. Defaults to 'center'.
        crop (bool, optional): allow cropping if padding is negative. Defaults to False.

    Returns:
        torch.Tensor: Padded `input`.
    """
    pad_values = [i if i > 0 else 0 for i in padding]
    if sum(pad_values) > 0:
        input = _pad(input, pad_values, mode, value, where)
    if crop:
        crop_values = [-i if i < 0 else 0 for i in padding]
        if sum(crop_values) > 0:
            input =  _crop(input, crop_values, where=where)
    return input

@overload
def pad_to_shape(input: torch.Tensor,shape: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center",crop = False,) -> torch.Tensor: ...
@overload
def pad_to_shape(input: np.ndarray,shape: Sequence[int],mode: str = "constant",value=None,where: Literal["center", "start", "end"] = "center",crop = False,) -> np.ndarray: ...
def pad_to_shape(
    input:torch.Tensor | np.ndarray,
    shape:Sequence[int],
    mode:str = "constant",
    value=None,
    where:Literal["center", "start", "end"] = "center",
    crop = False,
) -> np.ndarray | torch.Tensor:
    # broadcasting
    if len(shape) < input.ndim:
        shape = list(input.shape[:input.ndim - len(shape)]) + list(shape)

    return pad(
        input=input,
        padding=[shape[i] - input.shape[i] for i in range(input.ndim)],
        mode=mode,
        value=value,
        where=where,
        crop = crop,
    )

def pad_like(input:torch.Tensor, target:torch.Tensor, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None, crop = False,):
    return pad_to_shape(input, target.size(), where=where, mode=mode, value=value, crop=crop)

def pad_dim(
    input: torch.Tensor,
    dim: int,
    padding: int,
    where: Literal["center", "start", "end"] = "center",
    mode="constant",
    value: float | None=None,
    crop=True,
):
    """pad one dimension.

    Args:
        input (torch.Tensor): input tensor.
        dim (int): dimension to pad.
        padding (int): how much to increase that dimension by.
        where (str, optional): where to pad.
            if `center`, will pad start and end of each dimension evenly,
            if `start`, will pad at the start of each dimension,
            if `end`, will pad at the end. Defaults to 'center'.
        mode (str, optional): padding mode from pytorch plus "min" and "zeroes". Defaults to "constant".
        value (float | None, optional): value for constant padding. Defaults to None.
        crop (bool, optional): allow cropping if padding is negative. Defaults to False.

    Returns:
        padded tensor.
    """
    shape = list(input.shape)
    shape[dim] += padding
    return pad_to_shape(input, shape, where=where, mode=mode, value=value, crop = crop)

def pad_dim_to_size(input:torch.Tensor, dim: int, size:int, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None, crop=False):
    shape = list(input.shape)
    shape[dim] = size
    return pad_to_shape(input, shape, where=where, mode=mode, value=value, crop = crop)

def pad_dim_like(input:torch.Tensor, target:torch.Tensor, dim: int, where:Literal['center', 'start', 'end'] = 'center', mode='constant', value=None, crop=False):
    shape = list(input.shape)
    shape[dim] = target.shape[dim]
    return pad_to_shape(input, shape, where=where, mode=mode, value=value, crop = crop)