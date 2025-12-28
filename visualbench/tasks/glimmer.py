from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark
from ..utils import totensor, tonumpy
from .tsne import _make_colors, _pca, _make_frame

class Glimmer(Benchmark):
    """Glimmer - multidimensional scaling method for visualizing datasets.

    The loss is sum of squares of ``distance(x_i, x_j) - ||(p_i - p_j)||^2``
    with ``i`` for 1 to ``n`` and `j` from ``i + 1`` to ``n``,

    where:
    - `x_`i` is ith sample,
    - ``p_i`` is ith projected sample,
    - ``distance`` is some distance function like L2 norm.

    Renders:
        if ``n_components`` is 2, renders the reduced dataset.

    Args:
        inputs (torch.Tensor | np.ndarray | Any): The high-dimensional data, shape (n_samples, n_features).
        targets (torch.Tensor | np.ndarray | Any | None):
            Labels for visualization. Can be integer class labels or float regression targets.
        n_components (int): Dimensionality of the embedded space (usually 2).
        distance (Callable): distance function.
    """
    def __init__(
        self,
        inputs: torch.Tensor | np.ndarray | Any,
        targets: torch.Tensor | np.ndarray | Any | None = None,
        n_components: int = 2,
        p=2,
        pca_init: bool=True,
        resolution = 512,
    ):
        super().__init__()
        inputs = totensor(inputs)
        if targets is not None:
            targets = totensor(targets).squeeze()
            if targets.ndim != 1: raise ValueError(targets.shape)

        self.X_dists = nn.Buffer(torch.pdist(inputs, p=p))
        self.n_samples = inputs.shape[0]
        if pca_init:
            self.Y = nn.Parameter(_pca(inputs))
        else:
            self.Y = nn.Parameter(torch.randn(self.n_samples, n_components, generator=self.rng.torch()) * 0.0001)

        self.p = p
        self.resolution = resolution
        self._colors = _make_colors(targets, self.n_samples)

        self.set_multiobjective_func(lambda x: x.pow(2).mean())


    def get_loss(self) -> torch.Tensor:
        Y_dists = torch.pdist(self.Y, self.p) ** self.p

        loss = Y_dists - self.X_dists

        if self._make_images:
            with torch.no_grad():
                frame = _make_frame(self._colors, self.Y.numpy(force=True), self.resolution)
                self.log_image('data', frame, to_uint8=False)

        return loss


