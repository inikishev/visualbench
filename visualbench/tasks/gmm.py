import math
from typing import Any

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

from visualbench.benchmark import Benchmark
from visualbench.utils import totensor


def _default_X(n_samples: int, n_components: int, n_features: int, seed: int | None = 0):
    from sklearn.datasets import make_blobs
    X, y = make_blobs( # pylint:disable=unbalanced-tuple-unpacking # pyright:ignore[reportAssignmentType]
        n_samples=n_samples,
        centers=n_components,
        n_features=n_features,
        cluster_std=1.5,
        random_state=seed
    )

    torch.manual_seed(seed)
    X_torch = torch.from_numpy(X).float()

    # scale them more
    for i in range(n_components):
        mask = (y == i)
        transform = torch.randn(n_features, n_features) * 0.5 + torch.eye(n_features)
        X_torch[mask] = X_torch[mask] @ transform

    return X_torch


class GaussianMixtureNLL(Benchmark):
    """Fitting a gaussian mixture.

    Renders:
        ellipsoids corresponding to 95% confidence interval. If X has more than 2 features, renders in a pre-computed PCA projection.
    """
    blobs = staticmethod(_default_X)
    def __init__(self, X: Any, n_components: int, eps: float = 1e-6, resolution:int|tuple[int,int]=(384,384), padding = 40):
        super().__init__()
        if X is None: X = _default_X(5000, n_components=n_components, n_features=2, seed=self.rng.seed)
        else: X = totensor(X)

        self.X = nn.Buffer(X)
        self.n_features = X.size(1)
        self.n_components = n_components
        self.eps = eps
        self._unconstrained_pis = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, self.n_features, generator=self.rng.torch()))
        initial_cholesky = torch.eye(self.n_features).unsqueeze(0).repeat(n_components, 1, 1)
        self._cholesky_factors = nn.Parameter(initial_cholesky)

        # PCA pre compute
        self.PCA = None
        self.PCA_mean = None
        if self.n_features > 2:
            X_mean = self.X.mean(dim=0)
            X_centered = self.X - X_mean
            _, _, V = torch.linalg.svd(X_centered) # pylint:disable=not-callable
            self.PCA = V[:2, :].T
            self.PCA_mean = X_mean.unsqueeze(0)

        if self.PCA is not None:
            assert self.PCA_mean is not None
            projector = self.PCA.cpu()
            proj_mean = self.PCA_mean.cpu()
            X_2d = (X - proj_mean) @ projector
        else:
            X_2d = X

        # draw points
        if isinstance(resolution, tuple): width, height = resolution
        else: width = height = resolution

        min_vals = X_2d.min(dim=0).values
        max_vals = X_2d.max(dim=0).values
        data_range = max_vals - min_vals
        data_range[data_range == 0] = 1.0

        scale_x = (width - 2 * padding) / data_range[0]
        scale_y = (height - 2 * padding) / data_range[1]
        scale = min(scale_x, scale_y)

        width_px = data_range[0] * scale
        height_px = data_range[1] * scale
        offset_x = (width - width_px) / 2
        offset_y = (height - height_px) / 2

        def to_pixels(coords):
            transformed = (coords - min_vals) * scale
            px = transformed * torch.tensor([1.0, -1.0]) + torch.tensor([offset_x, height - offset_y])
            return px.int()

        self.to_pixels = to_pixels

        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        pixel_coords = to_pixels(X_2d)

        valid_x = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width)
        valid_y = (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
        px = pixel_coords[valid_x & valid_y]

        frame[px[:, 1], px[:, 0]] = [0, 0, 0]
        self.frame = frame

    def _get_params(self):
        """constraints are parametrized"""
        pis = torch.softmax(self._unconstrained_pis, dim=0)
        L = torch.tril(self._cholesky_factors)
        diag_indices = torch.arange(self.n_features)
        diag_vals = torch.exp(L[:, diag_indices, diag_indices]) + self.eps
        L[:, diag_indices, diag_indices] = diag_vals
        return pis, self.means.unsqueeze(0), L

    def _generate_frame(self):
        with torch.no_grad():
            pis, means, L = self._get_params()
            covs = L @ L.transpose(-1, -2)

            pis = pis.cpu()
            means = means.squeeze(0).cpu()
            covs = covs.cpu()

            if self.PCA is not None:
                assert self.PCA_mean is not None
                projector = self.PCA.cpu()
                proj_mean = self.PCA_mean.cpu()
                means_2d = (means - proj_mean) @ projector
                covs_2d = projector.T @ covs @ projector
            else:
                means_2d = means
                covs_2d = covs

        img = Image.fromarray(self.frame)
        draw = ImageDraw.Draw(img)

        for k in range(self.n_components):
            if pis[k] > 0.01:
                mean_k, cov_k = means_2d[k], covs_2d[k]

                try:
                    eigvals, eigvecs = torch.linalg.eigh(cov_k) # pylint:disable=not-callable
                except torch.linalg.LinAlgError:
                    continue

                eigvals = torch.clamp(eigvals, min=1e-9)

                # 95% confidence interval -> sqrt of chi-squared value for 2 DoF
                s = 2.4477
                radii = s * torch.sqrt(eigvals)

                t = torch.linspace(0, 2 * math.pi, 100)
                circle_pts = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
                ellipse_world_pts = (circle_pts @ torch.diag(radii) @ eigvecs.T) + mean_k
                ellipse_pixel_pts = self.to_pixels(ellipse_world_pts)

                draw.line(list(map(tuple, ellipse_pixel_pts.tolist())), fill='red', width=2)

        return np.array(img, dtype=np.uint8)


    def get_loss(self):
        n_samples, _ = self.X.shape
        X = self.X.unsqueeze(1) # (N, 1, D)
        pis, means, cholesky_factors = self._get_params()

        # log-determinant of covariance matrices
        log_det_chol = torch.sum(torch.log(cholesky_factors.diagonal(dim1=-2, dim2=-1)), dim=1)
        log_det_cov = 2 * log_det_chol

        # mahalanobis distance
        diff = X - means
        mahalanobis_sq = torch.empty(n_samples, self.n_components, device=X.device)
        for k in range(self.n_components):
            y = torch.linalg.solve_triangular( # pylint:disable=not-callable
                cholesky_factors[k], diff[:, k, :].T, upper=False
            ).T
            mahalanobis_sq[:, k] = torch.sum(y**2, dim=1)

        # log-probability for each sample under each component
        log_probs = -0.5 * (
            self.n_features * math.log(2 * math.pi) +
            log_det_cov.unsqueeze(0) +
            mahalanobis_sq
        )

        # mixing coefficients
        log_pis = torch.log(pis).unsqueeze(0)
        weighted_log_probs = log_probs + log_pis
        log_likelihood_per_sample = torch.logsumexp(weighted_log_probs, dim=1)

        if self._make_images:
            frame = self._generate_frame()
            self.log_image('frame', frame, to_uint8=False)

        return -torch.mean(log_likelihood_per_sample)

if __name__ == "__main__":
    bench = GaussianMixtureNLL(GaussianMixtureNLL.blobs(5000, 8, 8), 8).cuda()
    opt = torch.optim.Adam(bench.parameters(), 1e-1)
    bench.run(opt, max_passes=1000)
    bench.plot()