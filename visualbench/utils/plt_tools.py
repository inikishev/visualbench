import math
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any, Literal

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .format import tonumpy


def _auto_loss_yrange(*losses, yscale=None):
    losses = [tonumpy(l) for l in losses if l is not None]
    ymin = min(np.nanmin(l) for l in losses)
    finite_first = [l[0] for l in losses if math.isfinite(l[0])]
    if len(finite_first) == 0: return None
    ymax = max(finite_first)
    if ymin >= ymax: return None

    # expand range a little
    d = ymax - ymin
    ymin -= d*0.05; ymax += d*0.05

    if isinstance(yscale, dict): yscale = yscale['value']
    if isinstance(yscale, str):
        if yscale == 'symlog': ymin = max(ymin, 0)
        if yscale == 'log': ymin = min(np.nanmin(l) for l in losses)


    return (ymin, ymax)


def plot_loss(losses: dict[str, dict[int,float] | None], ylim: Literal['auto'] | tuple[float,float] | None = 'auto', yscale=None, smoothing: float | tuple[float,float, float] = 0, ax=None):
    if ylim == 'auto':
        ylim = _auto_loss_yrange(*(list(l.values()) for l in losses.values() if l is not None), yscale=yscale)

    if ax is None: ax = plt.gca()
    if yscale is not None: ax.set_yscale(yscale)
    if ylim is not None: ax.set_ylim(*ylim)
    if isinstance(smoothing, (int,float)): smoothing = (smoothing, smoothing, smoothing)

    for i, (label, loss_dict) in enumerate(losses.items()):
        if loss_dict is None: continue

        steps = list(loss_dict.keys())
        loss = list(loss_dict.values())

        sm = smoothing[i]
        if sm != 0: loss = gaussian_filter1d(loss, sm, mode='nearest')
        args:dict = {"lw":0.5} if label.endswith(' (perturbed)') else {}
        ax.plot(steps, loss, label=label, **args)

    ax.set_title("loss")
    ax.set_xlabel('loss')
    ax.set_ylabel('num forward/backward passes')
    return ax


def make_axes(
    n: int,
    nrows: int | float | None = None,
    ncols: int | float | None = None,
    fig=None,
    figsize: float | tuple[float, float] | None = None,
    axsize: float | tuple[float, float] | None = None,
    dpi: float | None = None,
    facecolor: Any | None = None,
    edgecolor: Any | None = None,
    frameon: bool = True,
    layout: Literal["constrained", "compressed", "tight", "none"] | None = "compressed",
) -> list[matplotlib.axes.Axes]:
    fix_ncols = ncols is not None
    fix_nrows = nrows is not None
    if fix_ncols and fix_nrows: fix_ncols = fix_nrows = False

    # distribute rows and cols
    if ncols is None:
        if nrows is None:
            nrows = n ** 0.45
        ncols = n / nrows # type:ignore
    else:
        nrows = n / ncols # type:ignore

    # ensure rows and cols are correct
    if nrows is None or ncols is None: raise ValueError('shut up pylance')
    nrows = round(nrows)
    ncols = round(ncols)
    nrows = max(nrows, 1)
    ncols = max(ncols, 1)

    c = True
    while nrows * ncols < n:
        if fix_ncols: nrows += 1
        elif fix_nrows: ncols += 1
        else:
            if c: ncols += 1
            else: nrows += 1
            c = not c

    nrows = min(nrows, n)
    ncols = min(ncols, n)

    # create the figure if it is None
    if isinstance(figsize, (int,float)): figsize = (figsize, figsize)

    if axsize is not None:
        if isinstance(axsize, (int,float)): axsize = (axsize, axsize)
        figsize = (ncols*axsize[0], nrows*axsize[1])

    if fig is None:
        fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor=edgecolor,
            frameon=frameon,
            layout=layout,
        )

    # create axes
    a: matplotlib.axes.Axes | np.ndarray = fig.subplots(nrows = nrows, ncols = ncols)
    if isinstance(a, np.ndarray): axes: list[matplotlib.axes.Axes] = a.ravel().tolist() # pyright: ignore[reportAssignmentType]
    else: axes = [a]

    # remove axis on overflow
    for i, ax in enumerate(axes):
        if i >= n: ax.set_axis_off()

    return axes


def legend_(
    ax,
    loc: Literal[
        "upper left",
        "upper right",
        "lower left",
        "lower right",
        "upper center",
        "lower center",
        "center left",
        "center right",
        "center",
        "best",
    ]
    | tuple[float, float] = "best",
    size: float | None = 6,
    edgecolor=None,
    linewidth: float | None = 3.0,
    frame_alpha=0.3,
    prop=None,
):
    if prop is None: prop = {}
    if size is not None and 'size' not in prop: prop['size'] = size

    leg = ax.legend(loc=loc, prop=prop, edgecolor=edgecolor,)
    leg.get_frame().set_alpha(frame_alpha)

    if linewidth is not None:
        for line in leg.get_lines():
            line.set_linewidth(linewidth)

    return ax
