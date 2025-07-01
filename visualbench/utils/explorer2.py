from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube


def rand_zoom(evaluate, center: torch.Tensor, f_max, iters: int):
    scales = torch.ones_like(center)
    points = torch.from_numpy(LatinHypercube(scales.numel()).random(iters)).to(scales) * 2 - 1

    for p in points:
        candidate = center + (p * scales)
        f, _ = evaluate(candidate.unsqueeze(0), False)
        f = f[0]

        if f < f_max: scales += p
        else: scales *= (1.1-p.abs())

    return scales

def scaled_random_search(evaluate, center: torch.Tensor, f_max, search_iters: int, zoom_iters: int):
    scales = rand_zoom(evaluate=evaluate, center=center, f_max=f_max, iters=zoom_iters)

    points = center + torch.from_numpy(LatinHypercube(scales.numel()).random(search_iters)).to(scales) * scales
    f, _ = evaluate(points, False)


def pls_project(k, X: list[torch.Tensor], f: list[torch.Tensor]):
    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(k)

    pls.fit(np.asarray(torch.stack(X)), np.asarray(torch.stack(f).unsqueeze(1)))

    def project(x):
        x_proj = pls.transform(np.asarray(x.detach().cpu()))
        return torch.from_numpy(x_proj).to(x)

    def unproject(x_proj):
        x = pls.inverse_transform(np.asarray(x_proj.detach().cpu()))
        return torch.from_numpy(x).to(x_proj)

    return project, unproject


class Visualizer:
    def __init__(self, objective, points, grads, values):
        self.objective = objective

        self.X: list[torch.Tensor] = [p.detach().cpu() for p in points]
        self.f: list[torch.Tensor] = list(values)
        self.G: list[torch.Tensor | None] = list(grads) if grads is not None else [None for _ in self.X]

        idx = np.argmin(self.f)
        self.f_best = self.f[idx]
        self.x_best = self.X[idx]
        self.x_0 = self.X[0]
        self.f_0 = self.f[0]

        self.project = lambda x: x
        self.unproject = lambda x: x

        self.device = points[0].device
        self.dtype = points[0].dtype

    @torch.no_grad
    def evaluate(self, x: torch.Tensor | Sequence[torch.Tensor], backward: bool):
        n = len(x)

        self.X.extend(x.detach().cpu() if isinstance(x, torch.Tensor) else (xi.detach().cpu() for xi in x))
        for xi in x:
            fi, gi = self.objective(xi.to(device=self.device, dtype=self.dtype), backward)

            self.f.append(fi.detach().cpu())
            self.G.append(gi.detach().cpu() if gi is not None else None)

        return self.f[-n:], (self.G[-n:] if backward else None)

    def projected_evaluate(self, x_proj: torch.Tensor | Sequence[torch.Tensor], backward: bool):

        x = self.unproject(x_proj)
        f, G = self.evaluate(x, backward)

        if G is not None:
            G = self.project(G)

        return f, G

    def search_iteration(self, k, search_iters = 50, zoom_iters = 50):
        scaled_random_search(self.projected_evaluate, self.project(self.x_best.unsqueeze(0))[0], self.f_best, search_iters=search_iters, zoom_iters=zoom_iters)
        scaled_random_search(self.evaluate, self.x_best, self.f_best, search_iters=search_iters//10, zoom_iters=zoom_iters//10)

        self.project, self.unproject = pls_project(k, self.X, self.f)

    def fit(self, ks: Sequence[int], search_iters = 50, zoom_iters = 50):
        for k in ks:
            self.search_iteration(k, search_iters, zoom_iters)

    @torch.no_grad
    def get_grid(self, grid_points=25, n_samples=16, tol=1.5):
        from scipy.stats.qmc import PoissonDisk

        sampler = PoissonDisk(2, radius=1/n_samples, hypersphere='surface', ncandidates=100)
        samples = sampler.random(n_samples) * 2 - 1
        sigma = 1

        increased = False
        stop = False
        while True:
            print(f'{sigma = }')
            for s_proj in samples:
                x = self.unproject(torch.from_numpy(s_proj).to(self.x_best).unsqueeze(0))[0] + self.x_best
                f, _ = self.objective(x)
                print(f'{x = }, {f = }, {self.f_best = }, {self.x_best = }')

                if f > self.f_0 * tol:
                    if increased: stop = True
                    sigma *= 0.75
                    break

            else:
                increased = True
                sigma *= 1.5

            if stop:
                break


        a1 = torch.linspace(-sigma, sigma, grid_points)
        a2 = torch.linspace(-sigma, sigma, grid_points)
        X, Y = torch.meshgrid(a1, a2, indexing='xy')
        Z = torch.zeros(grid_points, grid_points)

        for i in range(grid_points):
            for j in range(grid_points):

                x_proj = torch.stack((X[i, j], Y[i, j]))
                x = self.unproject(x_proj.unsqueeze(0))[0] + self.x_best

                y, _ = self.objective(x)
                Z[i, j] = y.detach().cpu()

        return X, Y, Z


    @torch.no_grad
    def plot(self, grid_points=25, points: torch.Tensor | None=None, values: torch.Tensor | None = None, norm=None):
        trajectory_proj = None

        if points is not None: # (n, ndim)
            if values is not None: values = values.detach().cpu()
            trajectory_proj = self.project(points.detach().cpu() - self.x_best)

        X, Y, Z = self.get_grid(grid_points=grid_points)
        plt.figure(figsize=(8, 6))
        colormesh = plt.pcolormesh(X.numpy(), Y.numpy(), Z.numpy(), cmap='coolwarm', norm=norm)
        plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, colors='white', alpha=0.5, linewidths=0.7)
        plt.colorbar(colormesh, label="value")

        plt.plot(0, 0, 'r+', markersize=15, label=f'x_min (y={self.f_best.item():.2f})')
        if trajectory_proj is not None:
            plt.plot(*zip(*trajectory_proj), color='r', lw=0.75, alpha=0.5)
            plt.scatter(*zip(*trajectory_proj), c=values)

        plt.xlabel("α₁")
        plt.ylabel("α₂")
        plt.title("function")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


