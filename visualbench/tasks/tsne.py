from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark
from ..utils import totensor, tonumpy


def _calculate_P(X: torch.Tensor, perplexity: float) -> torch.Tensor:
    """Calculates the high-dimensional joint probability distribution P."""
    n_samples = X.shape[0]

    # Calculate pairwise squared Euclidean distances
    sum_X_sq = torch.sum(X**2, 1)
    D_sq = -2 * X @ X.T + sum_X_sq.unsqueeze(1) + sum_X_sq.unsqueeze(0)

    # Binary search for sigma (via beta = 1/(2*sigma^2)) for each point
    P_conditional = torch.zeros((n_samples, n_samples), device=X.device)
    betas = torch.ones(n_samples, device=X.device)
    log_U = torch.log(torch.tensor(perplexity, device=X.device))

    for i in range(n_samples):
        # Binary search for a beta that results in the desired perplexity
        beta_min, beta_max = -np.inf, np.inf
        Di = D_sq[i, torch.cat((torch.arange(i), torch.arange(i + 1, n_samples)))]

        for _ in range(50):
            # Calculate conditional probabilities P_j|i
            P_i = torch.exp(-Di * betas[i])
            sum_Pi = torch.sum(P_i)
            if sum_Pi == 0: sum_Pi = torch.tensor(1e-12, device=X.device)

            H = torch.log(sum_Pi) + betas[i] * torch.sum(Di * P_i) / sum_Pi

            if torch.abs(H - log_U) < 1e-5:
                break # Converged

            if H > log_U:
                beta_min = betas[i].clone()
                if beta_max == np.inf:
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + beta_max) / 2.0
            else:
                beta_max = betas[i].clone()
                if beta_min == -np.inf:
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + beta_min) / 2.0

        # Set the final P_j|i for this point i
        P_i = torch.exp(-D_sq[i, :] * betas[i])
        P_i[i] = 0 # Set diagonal to 0
        P_i /= torch.sum(P_i) + 1e-12
        P_conditional[i, :] = P_i

    # Symmetrize to get joint probabilities P_ij
    P = (P_conditional + P_conditional.T) / (2 * n_samples)
    return torch.max(P, torch.tensor(1e-12, device=X.device))

def _make_colors(targets: Any, n_samples:int):
    if targets is None:
        return np.array([[0, 0, 0]]) # Black for all points

    targets = tonumpy(targets)

    # -------------------------------- categorical ------------------------------- #
    if targets.dtype in (int, np.int64, np.int32):
        unique_classes = np.unique(targets)
        n_classes = len(unique_classes)

        generator = np.random.default_rng(0)
        class_colors = (generator.uniform(0, 255, size=(n_classes, 3))).astype(np.uint8)
        colors = np.zeros((n_samples, 3), dtype=np.uint8)
        for i, cls in enumerate(unique_classes):
            colors[targets == cls] = class_colors[i]
        return colors

    # -------------------------------- regression -------------------------------- #
    targets_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-12)
    colormap_start = np.array([0, 0, 255])  # Blue
    colormap_end = np.array([255, 255, 0]) # Yellow
    colors = (colormap_start[None, :] * (1 - targets_norm)[:, None] + \
            colormap_end[None, :] * targets_norm[:, None]).astype(np.uint8)

    return colors

def _pca(inputs: torch.Tensor):
    from sklearn.decomposition import PCA
    inputs_pca = PCA(2).fit_transform(inputs.numpy(force=True))
    return torch.as_tensor(inputs_pca.copy()).clone()


def _make_frame(colors, Y: np.ndarray, resolution: int = 500, point_size: int = 5):
    # create a blank white canvas
    image = np.full((resolution, resolution, 3), 255, dtype=np.uint8)

    # normalize coordinates to fit within the canvas
    y_min, y_max = Y.min(), Y.max()
    scale = y_max - y_min
    if scale == 0: scale = 1.0

    y_norm = (Y - y_min) / scale

    # add a margin
    margin = point_size * 2
    coords = (y_norm * (resolution - 1 - 2 * margin) + margin).astype(int)

    # draw points
    r = point_size // 2
    for (x, y), color in zip(coords, colors):
        top = max(0, y - r)
        bottom = min(resolution, y + r + 1)
        left = max(0, x - r)
        right = min(resolution, x + r + 1)
        image[top:bottom, left:right] = color

    return image

class TSNE(Benchmark):
    """t-distributed Stochastic Neighbor Embedding dimensionality reduction.

    Renders:
        if ``n_components`` is 2, renders the reduced dataset.

    Args:
        inputs (torch.Tensor | np.ndarray | Any): The high-dimensional data, shape (n_samples, n_features).
        targets (torch.Tensor | np.ndarray | Any | None):
            Labels for visualization. Can be integer class labels or float regression targets.
        n_components (int): Dimensionality of the embedded space (usually 2).
        perplexity (float): The perplexity is related to the number of nearest neighbors
                            that is taken into account for each point.
        exaggeration_factor (float): Factor to multiply P by during early optimization.
        exaggeration_iters (int): The iteration number to stop early exaggeration.
    """
    def __init__(
        self,
        inputs: torch.Tensor | np.ndarray | Any,
        targets: torch.Tensor | np.ndarray | Any | None = None,
        n_components: int = 2,
        perplexity: float = 30.0,
        exaggeration_factor: float = 1.0,
        exaggeration_iters: int = 250,
        pca_init: bool=True,
        resolution = 512,
    ):
        super().__init__()
        inputs = totensor(inputs)
        if targets is not None:
            targets = totensor(targets).squeeze()
            if targets.ndim != 1: raise ValueError(targets.shape)

        self.n_samples = inputs.shape[0]
        self.perplexity = perplexity
        self.exaggeration_factor = exaggeration_factor
        self.exaggeration_iters = exaggeration_iters

        if pca_init:
            self.Y = nn.Parameter(_pca(inputs))
        else:
            self.Y = nn.Parameter(torch.randn(self.n_samples, n_components, generator=self.rng.torch()) * 0.0001)

        self.resolution = resolution
        with torch.no_grad():
            self.P = nn.Buffer(_calculate_P(inputs, perplexity))

        self.iteration = 0
        self._colors = _make_colors(targets, self.n_samples)

        self.set_multiobjective_func(torch.sum)
        self._show_titles_on_video = False


    def get_loss(self) -> torch.Tensor:
        sum_Y_sq = torch.sum(self.Y**2, 1)
        dists_Y_sq = -2 * self.Y @ self.Y.T + sum_Y_sq.unsqueeze(1) + sum_Y_sq.unsqueeze(0)

        num = 1.0 / (1.0 + dists_Y_sq)
        num.fill_diagonal_(0.0)
        Q = num / (torch.sum(num) + 1e-12)
        Q = torch.max(Q, torch.tensor(1e-12, device=self.device))

        if self.iteration < self.exaggeration_iters:
            P_eff = self.P * self.exaggeration_factor
        else:
            P_eff = self.P

        loss = P_eff * torch.log(P_eff / Q)

        if self._make_images:
            with torch.no_grad():
                frame = _make_frame(self._colors, self.Y.numpy(force=True), self.resolution)
                self.log_image('data', frame, to_uint8=False)

        self.iteration += 1
        return loss


