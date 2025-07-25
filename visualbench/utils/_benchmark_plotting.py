from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from .plt_tools import legend_, make_axes, plot_loss
from .format import to_HW3

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _key_exists(logger, k:str, plot_perturbed: bool):
    if k.endswith(' (perturbed)') and not plot_perturbed: return False
    if k.endswith(' (difference)'): return False
    if k in logger: return True
    if f'train {k}' in logger: return True
    if f'test {k}' in logger: return True
    if f'{k} (perturbed)' in logger: return True
    if f'train {k} (perturbed)' in logger: return True
    return False


def plot_trajectory(self: "Benchmark", cmap = 'coolwarm', loss_scale:Any = 'symlog', projector: Literal['pca'] | Any = 'pca', use_diff: bool = False, ax=None):
    if 'params' in self.logger: trajectory = self.logger.numpy('params') # n_points, n_dims
    elif 'projected' in self.logger: trajectory = self.logger.numpy('projected')
    else: raise RuntimeError("Either params or projections must be logged to plot trajectory")
    assert trajectory.ndim == 2
    ndim = trajectory.shape[1]

    loss = np.asarray(list(self.logger['train loss'].values()))
    mask: np.ndarray = np.isfinite(trajectory).any(axis=1) # pyright:ignore[reportAssignmentType]
    if len(loss) > len(mask): loss = loss[:len(mask)] # this rarely happens on KeyboardInterrupt
    if len(mask) > len(loss): mask = mask[:len(loss)] # this rarely happens on KeyboardInterrupt
    mask = np.logical_and(mask, np.isfinite(loss))

    trajectory = trajectory[mask] # remove nans
    loss = loss[mask]

    if ndim > 2:

        # use PCA or something else to reduce dimensionality
        if projector == 'pca':
            if len(trajectory) == 1:
                from sklearn.random_projection import GaussianRandomProjection
                projector = GaussianRandomProjection(n_components=2)
            else:
                from sklearn.decomposition import PCA
                projector = PCA(n_components=2)

        if use_diff: projector.fit(np.gradient(trajectory, axis=0), loss)
        else: projector.fit(trajectory, loss)

        trajectory = projector.transform(trajectory)

    if trajectory.shape[1] == 1: trajectory = np.concat([trajectory, trajectory], 1)
    assert trajectory.shape[1] == 2
    if ax is None: ax = plt.gca()
    ax.scatter(x=trajectory[:,0], y=trajectory[:,1], alpha=0.4, s=4, c=loss, cmap=cmap, norm=loss_scale)

    tgt = "params" if "params" in self.logger else f"{ndim} random projections"
    if use_diff: tgt = f"{tgt} difference"

    if ndim > 2: title = f"trajectory ({projector.__class__.__name__} on {tgt})"
    else: title = f"trajectory ({tgt})"
    ax.set_title(title)

    return ax

def plot_summary(
    self: "Benchmark",
    ylim: tuple[float, float] | Literal["auto"] | None,
    yscale,
    smoothing: float | tuple[float, float,float],
    axsize: float | tuple[float, float] | None,
    dpi: float | None,
    fig,
):
    # ------------------------------ create a figure ----------------------------- #
    image_keys = [k for k in self._image_keys.union(self._image_lowest_keys) if _key_exists(self.logger, k, self._plot_perturbed)]
    plot_keys = [k for k in self._plot_keys if _key_exists(self.logger, k, self._plot_perturbed)]

    n = len(image_keys) + len(plot_keys) + len(self._reference_images)
    if 'params' in self.logger or 'projected' in self.logger: n += 1

    axes = make_axes(n, axsize=axsize, dpi=dpi, fig=fig)
    axes_iter = iter(axes)

    # -------------------------------- linecharts -------------------------------- #
    for k in plot_keys:
        ax = next(axes_iter)
        if k == 'loss':
            self.plot_loss(ylim=ylim, yscale=yscale, smoothing=smoothing, ax=ax)

        else:
            train_k = f'train {k}'
            train_k_perturbed = f'train {k} (perturbed)'
            test_k = f'test {k}'
            ks = [k,train_k,test_k,train_k_perturbed] if self._plot_perturbed else [k,train_k,test_k]
            ks = [k for k in ks if k in self.logger]

            for kk in ks:
                x,y = self.logger[kk].keys(), self.logger[kk].values()
                args:dict = {"lw":0.5} if kk.endswith(' (perturbed)') else {}
                ax.plot(list(x), list(y), label=kk, **args)

            if len(ks) > 1: legend_(ax)
            ax.grid(which = 'major', axis='both', alpha=0.3)
            ax.grid(which = 'minor', axis='both', alpha=0.1)
            ax.set_title(k)
            ax.set_xlabel('num forward/backward passes')
            ax.set_ylabel(k)

    # -------------------------------- trajectory -------------------------------- #
    if 'params' in self.logger or 'projected' in self.logger:
        ax = next(axes_iter)
        plot_trajectory(self, ax=ax, loss_scale=yscale)

    # ---------------------------------- images ---------------------------------- #
    step = self.logger.stepmin('test loss') if 'test loss' in self.logger else self.logger.stepmin('train loss')

    def imshow_(v, title):
        ax = next(axes_iter)
        ax.imshow(to_HW3(v))
        ax.set_axis_off()
        ax.set_title(title)

    for k,v in self._reference_images.items():
        imshow_(v, k)

    for k in image_keys:
        if k.endswith(' (perturbed)') and not self._plot_perturbed: continue
        imshow_(self.logger.closest(k, step), k)

    return fig