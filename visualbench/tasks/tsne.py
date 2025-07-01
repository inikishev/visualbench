from typing import Any, Literal

import imageio
import numpy as np
import torch
import torch.nn as nn

from ..benchmark import Benchmark
from ..utils import totensor


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


class TSNE(Benchmark):
    """
    Args:
        inputs (torch.Tensor): The high-dimensional data, shape (n_samples, n_features).
        targets (Optional[Union[torch.Tensor, np.ndarray]]):
            Labels for visualization. Can be integer class labels or float regression targets.
        n_components (int): Dimensionality of the embedded space (usually 2).
        perplexity (float): The perplexity is related to the number of nearest neighbors
                            that is taken into account for each point.
        exaggeration_factor (float): Factor to multiply P by during early optimization.
        exaggeration_iters (int): The iteration number to stop early exaggeration.
        device (Optional[str]): The device to run computations on ('cpu' or 'cuda').
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
            from sklearn.decomposition import PCA
            inputs_pca = PCA(2).fit_transform(inputs.numpy(force=True))
            self.Y = nn.Parameter(torch.as_tensor(inputs_pca.copy()).clone())
        else:
            self.Y = nn.Parameter(torch.randn(self.n_samples, n_components, generator=self.rng.torch()) * 0.0001)
        self.resolution = resolution
        with torch.no_grad():
            self.P = nn.Buffer(_calculate_P(inputs, perplexity))

        self.iteration = 0

        # make colors
        if targets is None:
            self._colors = np.array([[0, 0, 0]] * self.n_samples) # Black for all points

        else:
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()

            if targets.dtype == int or targets.dtype == np.int64 or targets.dtype == np.int32: # Categorical
                unique_classes = np.unique(targets)
                n_classes = len(unique_classes)
                # Generate a consistent color for each class
                np.random.seed(42) # for reproducibility
                class_colors = (np.random.rand(n_classes, 3) * 255).astype(np.uint8)
                colors = np.zeros((self.n_samples, 3), dtype=np.uint8)
                for i, cls in enumerate(unique_classes):
                    colors[targets == cls] = class_colors[i]
                self._colors = colors

            else:
                targets_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-12)
                colormap_start = np.array([0, 0, 255])  # Blue
                colormap_end = np.array([255, 255, 0]) # Yellow
                colors = (colormap_start[None, :] * (1 - targets_norm)[:, None] + \
                        colormap_end[None, :] * targets_norm[:, None]).astype(np.uint8)

                self._colors = colors


    def _make_frame(self, Y_np: np.ndarray, canvas_size: int = 500, point_size: int = 5):
        # Create a blank white canvas
        image = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

        # Normalize coordinates to fit within the canvas
        y_min, y_max = Y_np.min(), Y_np.max()
        scale = y_max - y_min
        if scale == 0: scale = 1.0

        y_norm = (Y_np - y_min) / scale

        # Add a margin
        margin = point_size * 2
        coords = (y_norm * (canvas_size - 1 - 2 * margin) + margin).astype(int)

        # Draw points (as small squares)
        r = point_size // 2
        for (x, y), color in zip(coords, self._colors):
            # Clamp coordinates to be within canvas bounds
            top = max(0, y - r)
            bottom = min(canvas_size, y + r + 1)
            left = max(0, x - r)
            right = min(canvas_size, x + r + 1)
            image[top:bottom, left:right] = color

        return image

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

        loss = torch.sum(P_eff * torch.log(P_eff / Q))

        if self._make_images:
            with torch.no_grad():
                frame = self._make_frame(self.Y.detach().cpu().numpy(), self.resolution)
                self.log_image('data', frame, to_uint8=False)

        self.iteration += 1
        return loss


