from contextlib import nullcontext
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from .format import tonumpy


@torch.no_grad
def _make_2d_function_mesh(
    f,
    xrange,
    yrange,
    num: int | None,
    step: int | None,
    lib: "Literal['numpy', 'torch']",
    batched: bool,
    enable_grad,
    dtype,
    device,
):
    xrange = tonumpy(xrange)
    yrange = tonumpy(yrange)
    if num is None: num = (xrange[1] - xrange[0]) / step

    np_or_torch = np if lib == 'numpy' else torch
    x = np_or_torch.linspace(xrange[0], xrange[1], num) # type:ignore
    y = np_or_torch.linspace(yrange[0], yrange[1], num) # type:ignore
    X,Y = np_or_torch.meshgrid(x, y, indexing='ij') # grid of point # type:ignore

    if dtype is not None or device is not None:
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            try:
                X = X.astype(dtype)
                Y = Y.astype(dtype)
            except Exception as e:
                print(f'cant move to dtype and device: {e}')

        if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
            try:
                X = X.to(device = device, dtype=dtype)
                Y = Y.to(device = device, dtype=dtype)
            except Exception as e:
                print(f'cant move to dtype and device: {e}')

    with torch.enable_grad() if enable_grad else nullcontext():
        if batched: Z = f(X, Y)
        else: Z = np.vectorize(f)(X, Y)
        return X, Y, Z



@torch.no_grad
def funcplot2d(
    f,
    xrange,
    yrange,
    num: int | None = 1000,
    step  = None,
    cmap = 'gray',
    norm = None,
    surface_alpha = 1.,
    levels = 12,
    contour_cmap = 'binary',
    contour_lw = 0.5,
    contour_alpha = 0.3,
    log_contour = False,
    grid_alpha = 0.,
    grid_color = 'gray',
    grid_lw=0.5,
    lib: "Literal['numpy', 'torch']" = 'numpy',
    batched: bool = True,
    dtype=None,
    device=None,
    ax=None,
):
    if ax is None: ax = plt.gca()
    X, Y, Z = _make_2d_function_mesh(f, xrange, yrange, num=num, step=step, lib=lib, batched=batched, dtype=dtype,device=device, enable_grad=False)
    X, Y, Z = [tonumpy(i) for i in (X, Y, Z)]

    contour_norm = None
    if log_contour:
        Z_min = np.percentile(Z[Z>0], 1).clip(min=1e-2)
        log_levels = np.logspace(np.log10(Z_min), np.log10(Z.max()), levels)
        levels = log_levels
        # contour_norm = 'symlog'

    ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = surface_alpha, norm = norm)
    if levels is not None: ax.contour(X, Y, Z, levels=levels, cmap=contour_cmap, linewidths=contour_lw, alpha=contour_alpha, norm=contour_norm)
    if grid_alpha > 0: ax.grid(alpha=grid_alpha, lw=grid_lw, color=grid_color)
    return ax