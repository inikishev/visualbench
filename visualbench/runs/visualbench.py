"""not for laptop"""
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from kornia.losses import ssim_loss
from sklearn.datasets import make_swiss_roll
from torch import nn
from torch.nn import functional as F

from .. import data, models, tasks
from ..utils import CUDA_IF_AVAILABLE
from .benchpack import OptimizerBenchPack

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")

def _unbatched_ssim(x,y):
    return ssim_loss(x[None,:], y[None,:],5)

class Visualbench(OptimizerBenchPack):
    def __init__(
        self,
        opt_fn: Callable,
        sweep_name: str,

        # MBS parameters
        hyperparam: str | None = "lr",
        log_scale: bool = True,
        grid: Iterable[float] = (2, 1, 0, -1, -2, -3, -4, -5),
        step: float = 1,
        num_candidates: int = 2,
        num_binary: int = 12,
        num_expansions: int = 12,
        rounding = 1,
        fixed_hyperparams: dict | None = None,
        max_dim: int | None = None,
        tune: bool = True,
        skip:str | Sequence[str] | None = None,

        # storage
        root: str = "Visualbench",
        print_records: bool = True,
        print_progress: bool = True,
        print_time: bool = False,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = True,

        # pass stuff
        num_extra_passes: float | Callable[[int], float] = 0,
        step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,

        init_fn = lambda opt_fn, bench, value: opt_fn([p for p in bench.parameters() if p.requires_grad], value)
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def run(self):
        torch.manual_seed(0)
        self.run_2d()
        self.run_projected()
        self.run_visual()
        self.run_linalg()


    def run_2d(self):
        torch.manual_seed(0)

        bench = tasks.FunctionDescent('booth')
        self.run_bench(bench, '2D - booth', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('booth').set_noise(10, 10)
        self.run_bench(bench, '2D - booth (noisy)', passes=1000, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('ill2')
        self.run_bench(bench, '2D - ill2', passes=1000, sec=10, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1000, sec=30, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosen').set_noise(0.5, 0.5)
        self.run_bench(bench, '2D - rosenbrock (noisy)', passes=1000, sec=30, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2000, sec=60, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('mycs1')
        self.run_bench(bench, '2D - mycs1', passes=500, sec=60, metrics='train loss', vid_scale=1)

        bench = tasks.SimultaneousFunctionDescent('rosen').to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - rosenbrock', passes=1000, sec=60, metrics='train loss', vid_scale=3)

        bench = tasks.DecisionSpaceDescent.with_x0("ill2", models.MLP([512, 256, 128, 64, 32, 16, 8, 4, 2]), 512, maxiter=2000, progress=False).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'decision space descent - ill2', passes=1000, sec=60, metrics='train loss', vid_scale=3)

        bench = tasks.DecisionSpaceDescent.with_x0("ill2", models.MLP([512, 256, 128, 64, 32, 16, 8, 4, 2]), 512, maxiter=2000, noise=100, progress=False).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'decision space descent - ill2 (noisy)', passes=1000, sec=60, metrics='train loss', vid_scale=3)

    def run_projected(self):
        torch.manual_seed(0)

        # ------------------------------ Rosenbrock-384 ------------------------------ #
        bench = tasks.projected.Rosenbrock(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Projected - Rosenbrock 384', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

    def run_visual(self):
        torch.manual_seed(0)

        # ------------------------------- NeuralDrawer ------------------------------- #
        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ReLU+bn', passes=2_000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), batch_size=16, expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer BS-16 - ReLU+bn', passes=2_000, sec=60, metrics='train loss', vid_scale=2)

        # ------------------------------- Colorization ------------------------------- #
        # ndim  = 1024
        # 3.2s. ~ 1m. 4s.
        bench = tasks.Colorization.small().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        # ------------------------------- Graph layout ------------------------------- #
        # ndim = 128
        # 3.8s. ~ 1m. 16s.
        bench = tasks.GraphLayout(tasks.GraphLayout.GRID()).to(CUDA_IF_AVAILABLE)
        bench_name = 'Visual - Graph layout optimization'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.

        # ----------------------- Sine Approximator - Tanh 7-4 ---------------------- #
        # ndim = 29
        # 4.2s ~ 1m. 24s.
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=7, resolution=(384,768),
        ) # NO CUDA
        bench_name = 'Visual - Sine Approximator - Tanh 7-4'
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=7, resolution=(384,768), batch_size=16,
        ) # NO CUDA
        bench_name = 'Visual - Sine Approximator BS-16 - Tanh 7-4'
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='test loss', vid_scale=1)

        # ----------------------- Particle minmax ---------------------- #
        # ndim = 64
        # 2s ~ 40s
        bench = tasks.ClosestFurthestParticles(32, spread=0.75) # NO CUDA
        self.run_bench(bench, 'Visual - Particle min-max', passes=2_000, sec=60, metrics='train loss', vid_scale=1)

        # -------------------------------- Muon coeffs ------------------------------- #
        # ndim = 15
        # 9.1s. ~ 3m. 3s.
        bench = tasks.MuonCoeffs(resolution=(512, 512)) # NO CUDA
        bench_name = 'Visual - Muon coefficients'
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', vid_scale=1, binary_mul=0.75)

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = 600
        # 4.4s. ~ 1m. 30s.
        bench = tasks.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Alpha Evolve B1', passes=4_000, sec=90, metrics='train loss', vid_scale=1)

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = 49,152
        # 14s. ~ 4m. 40s.
        # 9+4=13 ~ 3m.
        bench = tasks.StyleTransfer(data.FROG96, data.GEOM96).to(CUDA_IF_AVAILABLE)
        bench_name = "Visual - Style Transfer"
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', binary_mul=0.4, vid_scale=2)

        # ----------------------------------- moons ---------------------------------- #
        bench = tasks.Moons(models.MLP([2,2,2,2,2,2,2,2,1], act_cls=nn.ELU, ortho_init=True)).to(CUDA_IF_AVAILABLE)
        bench_name = 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU'
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP([2,2,2,2,2,2,2,2,1], act_cls=nn.ELU, ortho_init=True), batch_size=1, n_samples=2048, train_split=1024).to(CUDA_IF_AVAILABLE)
        bench_name= "Visual - Moons BS-1 - MLP(2-2-2-2-2-2-2-2-1)-ELU"
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics='test loss', vid_scale=2, test_every=10)

        # ------------------------------- lines drawer ------------------------------- #
        bench = tasks.LinesDrawer(data.WEEVIL96, 100, loss=_unbatched_ssim).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - LinesDrawer SSIM', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # -------------------------- deformable registration ------------------------- #
        bench = tasks.DeformableRegistration(data.FROG96, grid_size=(5,5)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - DeformableRegistration', passes=2_000, sec=60, metrics='train loss', vid_scale=4)

    def run_linalg(self):
        torch.manual_seed(0)

        # ---------------------------------- inverse --------------------------------- #
        bench = tasks.Inverse(data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - Inverse MSE', passes=2_000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.StochasticInverse(data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticInverse', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        bench = tasks.LeastSquares(data.FROG96, data.SANIC96, ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - LeastSquares', passes=2_000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.StochasticMatrixRecovery(data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticMatrixRecovery - MSE', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        bench = tasks.StochasticMatrixRecovery(data.SANIC96, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticMatrixRecovery - L1', passes=2_000, sec=60, metrics='test loss', vid_scale=2)
