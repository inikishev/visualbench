import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import gpytorch
import gpytorch.kernels as gk
import torch
from accelerate import Accelerator
from kornia.losses import ssim_loss
from sklearn.datasets import load_breast_cancer, make_swiss_roll
from torch import nn
from torch.nn import functional as F

from .. import data, models, tasks
from .. import losses as losses_
from ..models.ode import NeuralODE
from ..utils import CUDA_IF_AVAILABLE
from ..utils.clean_mem import clean_mem
from ..utils.python_tools import format_number, to_valid_fname
from .run import Run, Sweep, Task, _target_metrics_to_dict, mbs_search, single_run

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")

def _unbatched_ssim(x,y):
    return ssim_loss(x[None,:], y[None,:],5)

class MBSRun:
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
        rounding=1,
        fixed_hyperparams: dict | None = None,
        max_dim: int | None = None,
        tune: bool = True,

        # storage
        root: str = "optimizers",
        print_records: bool = True,
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = True,
    ):
        self.root = root
        self.sweep_name = sweep_name
        self.summaries_root = f"{self.root} - summaries"
        self.summary_dir = os.path.join(self.summaries_root, f"{to_valid_fname(self.sweep_name)}")
        self.hyperparam = hyperparam

        def run_bench(bench: "Benchmark", task_name: str, passes: int, sec: float, metrics:str | Sequence[str] | dict[str, bool], vid_scale:int|None, fps=60, binary_mul: float = 1, test_every: int | None = None):
            clean_mem()

            dim = sum(p.numel() for p in bench.parameters() if p.requires_grad)
            if max_dim is not None and dim > max_dim: return

            if accelerate and next(bench.parameters()).is_cuda: # skip CPU because accelerator state can't change.
                accelerator = Accelerator()
                bench = accelerator.prepare(bench)

            def logger_fn(value: float):
                if dim > 10_000: clean_mem()
                bench.reset().set_benchmark_mode().set_print_inverval(None)
                opt = opt_fn(bench.parameters(), value)
                bench.run(opt, passes, max_seconds=sec, test_every_forwards=test_every)
                if print_progress and bench.seconds_passed is not None and bench.seconds_passed > sec:
                    print(f"{sweep_name}: '{task_name}' timeout, {bench.seconds_passed} > {sec}!")
                return bench.logger

            if hyperparam is None or (not tune):
                sweep = single_run(logger_fn, metrics=metrics, fixed_hyperparams=fixed_hyperparams, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, print_progress=print_progress, save=save, load_existing=load_existing)

            else:
                sweep = mbs_search(logger_fn, metrics=metrics, search_hyperparam=hyperparam, fixed_hyperparams=fixed_hyperparams, log_scale=log_scale, grid=grid, step=step, num_candidates=num_candidates, num_binary=max(1, int(num_binary*binary_mul)), num_expansions=num_expansions, rounding=rounding, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, save=save, load_existing=load_existing, print_progress=print_progress)

            # render video
            if render_vids and vid_scale is not None:
                for metric, maximize in _target_metrics_to_dict(metrics).items():
                    video_path = os.path.join(self.summary_dir, f'{task_name} - {metric}')
                    if os.path.exists(f'{video_path}.mp4'): continue

                    best_run = sweep.best_runs(metric, maximize, 1)[0]
                    value = 0
                    if tune and hyperparam is not None: value = best_run.hyperparams[hyperparam]
                    bench.reset().set_benchmark_mode(False).set_print_inverval(None)
                    opt = opt_fn(bench.parameters(), value)
                    bench.run(opt, passes, max_seconds=sec, test_every_forwards=test_every)
                    if not os.path.exists(self.summaries_root): os.mkdir(self.summaries_root)
                    if not os.path.exists(self.summary_dir): os.mkdir(self.summary_dir)
                    bench.render(f'{video_path} __TEMP__', scale=vid_scale, fps=fps, progress=False)
                    os.rename(f'{video_path} __TEMP__.mp4', f'{video_path}.mp4')


        self.run_bench = run_bench

    def run(self, ML=True, synthetic=True, stochastic=True, losses=True, visual=True, twod=True):
        if twod:
            self.run_2d()

        if visual:
            self.run_visual()

        if synthetic:
            self.run_synthetic()
            if stochastic: self.run_synthetic_stochastic()

        if ML:
            self.run_real()
            self.run_ML()
            if stochastic: self.run_ML_stochastic()

        if losses:
            self.run_losses()

    def run_visual(self):
        # ------------------------------- neural drawer ------------------------------ #
        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=nn.ReLU, bn=True), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ReLU+bn', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=nn.ELU), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ELU', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=models.act.Sine), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Sine', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

        # ------------------------------- lines drawer ------------------------------- #
        bench = tasks.LinesDrawer(data.WEEVIL96, 100, loss=_unbatched_ssim).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - LinesDrawer SSIM', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ----------------------------- partition drawer ----------------------------- #
        bench = tasks.PartitionDrawer(data.WEEVIL96, 100).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - PartitionDrawer', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ----------------------------------- moons ---------------------------------- #
        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2]),).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2], act_cls=nn.ReLU, bn=True)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ReLU+bn', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2]), batch_size=16, n_samples=2048, test_split=1024).to(CUDA_IF_AVAILABLE)
        bench_name= "Visual - Moons BS-16 - MLP(2-2-2-2-2-2-2-2-1)-ELU"
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics='test loss', vid_scale=2, test_every=1)

        # ------------------------------- Colorization ------------------------------- #
        # ndim  = 24,576
        # 2.7s. ~ 54s.
        bench = tasks.Colorization.snake().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=2_000, sec=90, metrics='train loss', vid_scale=3)

        # ------------------------- Colorization (2nd order) ------------------------- #
        # ndim  = 1024
        # 3.2s. ~ 1m. 4s.
        bench = tasks.Colorization.small(order=2).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (2nd order)', passes=2_000, sec=60, metrics='train loss', vid_scale=8)

        # ------------------------- Colorization (1.3th power) ------------------------- #
        # ndim  = 1024
        # 3.2s. ~ 1m. 4s.
        bench = tasks.Colorization.small(power=1.3).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (1.3th power)', passes=2_000, sec=60, metrics='train loss', vid_scale=8)

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = 600
        # 4.4s. ~ 1m. 30s.
        bench = tasks.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Alpha Evolve B1', passes=4_000, sec=90, metrics='train loss', vid_scale=1)

        # ----------------------------------- t-SNE ---------------------------------- #
        # ndim = 1,138
        # 3.7s. ~ 1m. 12s.
        X, y = make_swiss_roll(1000, noise=0.1, hole=True, random_state=0)
        bench = tasks.TSNE(X, y).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - t-SNE', passes=2_000, sec=90, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.

        # ------------------------------- Graph layout ------------------------------- #
        # ndim = 128
        # 3.8s. ~ 1m. 16s.
        bench = tasks.GraphLayout(tasks.GraphLayout.GRID()).to(CUDA_IF_AVAILABLE)
        bench_name = 'Visual - Graph layout optimization'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = 49,152
        # 14s. ~ 4m. 40s.
        # 9+4=13 ~ 3m.
        bench = tasks.StyleTransfer(data.FROG96, data.GEOM96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Style Transfer', passes=2_000, sec=120, metrics='train loss', binary_mul=0.4, vid_scale=2)

        # -------------------------------- Muon coeffs ------------------------------- #
        # ndim = 15
        # 9.1s. ~ 3m. 3s.
        bench = tasks.MuonCoeffs(resolution=(512, 512)) # NO CUDA
        self.run_bench(bench, 'Visual - Muon coefficients', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # ----------------------- Sine Approximator - Tanh 7-4 ---------------------- #
        # ndim = 15
        # 4.2s ~ 1m. 24s.
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=7, resolution=(384,768),
        ) # NO CUDA

        self.run_bench(bench, 'Visual - Sine Approximator - Tanh 7-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # ----------------------- Sine Approximator - LeakyReLU 10-4 ---------------------- #
        # ndim = 15
        # 6.4s ~ 2m. 8s.
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=10, act=F.leaky_relu, resolution=(384,768),
        ) # NO CUDA
        self.run_bench(bench, 'Visual - Sine Approximator - LeakyReLU 10-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # ----------------------- Particle minmax ---------------------- #
        # ndim = 64
        # 2s ~ 40s
        bench = tasks.ClosestFurthestParticles(32, spread=0.75) # NO CUDA
        self.run_bench(bench, 'Visual - Particle min-max', passes=2_000, sec=60, metrics='train loss', vid_scale=1)


    def run_real(self):
        # ---------------------------- Human heart dipole ---------------------------- #
        # ndim = 8
        # 3.3s. ~ 1m. 6s.
        bench = tasks.HumanHeartDipole() # NO CUDA
        self.run_bench(bench, "Real - Human heart dipole", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

        # ---------------------------- Propane combustion ---------------------------- #
        # ndim = 11
        # 3.3s. ~ 1m. 6s.
        bench = tasks.PropaneCombustion() # NO CUDA
        self.run_bench(bench, "Real - Propane combustion", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

    def run_ML(self):
        # ---------------------- Small MLP (full-batch MNIST-1D) --------------------- #
        # ndim = 850
        # 3s. ~ 1m.
        bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[10, 10, 10, 10], act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - MLP(40-10-10-10-10-10)-ELU", passes=2_000, sec=60, metrics = LOSSES, vid_scale=None)

        # ---------------------- Small ReLU-Net (full-batch MNIST-1D) --------------------- #
        # ndim = 850
        # ?
        bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[10, 10, 10, 10], act_cls=nn.ReLU, bn=True)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - MLP(40-10-10-10-10-10)-ReLU+bn", passes=2_000, sec=60, metrics = LOSSES, vid_scale=None)

        # -------------------- Recurrent MLP (full-batch MNIST-1D) ------------------- #
        # ndim = 2,410
        # 3.6s. ~ 1m. 16s.
        bench = tasks.datasets.Mnist1d(models.RecurrentMLP(40, 10, width=40, n_passes=5, act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - RecurrentMLP", passes=2_000, sec=60, metrics = LOSSES, vid_scale=None)

        # ---------------------- NeuralODE (full-batch MNIST-1D) --------------------- #
        # ndim = 2,050
        # 3.5s ~ 1m. 10s.
        bench = tasks.datasets.Mnist1d(NeuralODE(40, 10, width=40, act_cls=nn.Softplus)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - NeuralODE", passes=2_000, sec=60, metrics = LOSSES, vid_scale=None)

        # --------------------- Thin ConvNet (full-batch MNIST-1D) -------------------- #
        # ndim = 1,338
        # 9.5s. ~ 3m.
        bench = tasks.datasets.Mnist1d(models.mnist1d.TinyLongConvNet()).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - ThinConvNet", passes=2_000, sec=120, metrics = LOSSES, vid_scale=None)

        # ------------------------- GRU (full-batch MNIST-1D) ------------------------ #
        # ndim = 1,510
        # 11s. ~ 3m. 40s.
        bench = tasks.datasets.Mnist1d(models.RNN(1, 10, hidden_size=10, num_layers=2, rnn=nn.GRU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - GRU", passes=2_000, sec=120, metrics = LOSSES, vid_scale=None)

        # ------------------------------ ThinPINN (Wave PDE) ----------------------------- #
        # ndim = 2,499
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=32, n_hidden=4)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - TinyFLS', passes=2_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)

        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - FLS', passes=2_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)


    def run_ML_stochastic(self):
        # ------------------------ logistic regression ------------------------ #
        # ndim = 385
        # 7.5s. ~ 2m. 30s.
        bench = tasks.datasets.Covertype(models.MLP(54, 7, hidden=None), batch_size=8).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Covertype BS-8 - Logistic Regression'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # ------------------------------- MLP (MNIST-1D) ------------------------------ #
        # ndim = 56,874
        # ?
        bench = tasks.datasets.Mnist1d(
            models.MLP(40, 10, hidden=[64,96,128,256], act_cls=nn.ELU),
            batch_size=64
        ).to(CUDA_IF_AVAILABLE)
        bench_name = "MLS - MNIST-1D BS-64 - MLP(40-64-96-128-256-10)"
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=10, metrics = "test loss", vid_scale=None)

        # ----------------------------- ConvNet (MNIST-1D) ---------------------------- #
        # ndim = 134,410
        # 19s. ~ 7m.
        bench = tasks.datasets.Mnist1d(
            models.mnist1d.ConvNet(dropout=0.5),
            batch_size=32,
            test_batch_size=512
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D BS-32 - ConvNet', passes=6_000, sec=360, test_every=20, metrics='test loss', vid_scale=None)

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.datasets.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MNIST-1D BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=20, metrics='test loss', vid_scale=None)


    def run_losses(self):
        # ----------------------------------- LInf ----------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.linf_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - L-Infinity', passes=2000, sec=30, metrics=LOSSES, vid_scale=None)

        # ---------------------------------- Median ---------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.median_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Median', passes=2000, sec=30, metrics=LOSSES, vid_scale=None)

        # ---------------------------------- Quartic --------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.quartic_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Quartic', passes=2000, sec=30, metrics=LOSSES, vid_scale=None)


        # ------------------------------- Quartic rooot ------------------------------ #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.qrmse_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Quartic root', passes=2000, sec=30, metrics=LOSSES, vid_scale=None)

        # ------------------------------------ L4 ------------------------------------ #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100,
            criterion=partial(losses_.norm_loss, ord=4), normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - L4', passes=2000, sec=30, metrics=LOSSES, vid_scale=None)


    def run_synthetic(self):
        # ---------------------------- Diabolical Function --------------------------- #
        # ndim = 512
        # 1.6s. ~ 32s.
        bench = tasks.IllConditioned().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Ill conditioned quadratic', passes=2_000, sec=30, metrics='train loss', vid_scale=None)

        # -------------------------------- Rosenbrock -------------------------------- #
        # ndim = 512
        # ?
        bench = tasks.Rosenbrock().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Rosenbrock', passes=2_000, sec=30, metrics='train loss', vid_scale=None)

        # -------------------------------- Least Squares -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.LeastSquares(data.WEEVIL96, data.FROG96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Least Squares', passes=2_000, sec=30, metrics='train loss', vid_scale=2)

        # -------------------------------- Inverse L1 -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.Inverse(data.WEEVIL96, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse - L1', passes=2_000, sec=30, metrics='train loss', vid_scale=2)

        # -------------------------------- Inverse MSE -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.Inverse(data.WEEVIL96, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse - MSE', passes=2_000, sec=30, metrics='train loss', vid_scale=2)

        # -------------------------------- Tropical QR L1 -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.QR(data.WEEVIL96, criterion=F.l1_loss, algebra='tropical').to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Tropical QR - L1', passes=2_000, sec=30, metrics='train loss', vid_scale=2)

        # -------------------------------- Tropical QR MSE -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.QR(data.WEEVIL96, criterion=F.mse_loss, algebra='tropical').to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Tropical QR - MSE', passes=2_000, sec=30, metrics='train loss', vid_scale=2)

        # ----------------------------- Matrix idempotent ---------------------------- #
        # ndim = 27,648
        # 8.2s ~ 2m. 44s.
        bench = tasks.MatrixIdempotent(data.SANIC96, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Matrix idempotent', passes=2_000, sec=30, metrics='train loss', vid_scale=2)


    def run_synthetic_stochastic(self):
        # --------------------------- Stochastic inverse L1 -------------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticInverse(data.WEEVIL96, vec=True, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic inverse - L1', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # --------------------------- Stochastic inverse MSE -------------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticInverse(data.WEEVIL96, vec=True, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic inverse - MSE', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ------------------------ Stochastic matrix recovery L1 ------------------------ #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixRecovery(data.get_text(), vec=True, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix recovery - L1', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ------------------------ Stochastic matrix recovery MSE ------------------------ #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixRecovery(data.get_text(), vec=True, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix recovery - MSE', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ----------------------- Stochastic matrix root L1 ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixRoot(data.SANIC96, 10, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix root - L1', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ----------------------- Stochastic matrix root MSE ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixRoot(data.SANIC96, 10, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix root - MSE', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ----------------------- Stochastic matrix idempotent ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixIdempotent(data.SANIC96, n=10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix idempotent', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        # ----------------------- Stochastic matrix idempotent (hard) ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixIdempotent(data.SANIC96, n=10, vec=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix idempotent (hard)', passes=4_000, sec=60, metrics='test loss', vid_scale=2)


    def run_2d(self):
        bench = tasks.FunctionDescent('booth')
        self.run_bench(bench, '2D - booth', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('ill')
        self.run_bench(bench, '2D - ill', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('star')
        self.run_bench(bench, '2D - star', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('around')
        self.run_bench(bench, '2D - around', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('dipole')
        self.run_bench(bench, '2D - dipole field', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('rastrigin')
        self.run_bench(bench, '2D - rastrigin', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('rosen10')
        self.run_bench(bench, '2D - rosenbrock-10', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('spiral')
        self.run_bench(bench, '2D - spiral', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2000, sec=60, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('oscillating')
        self.run_bench(bench, '2D - oscillating', passes=2000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        # ------------------------------- simultaneous ------------------------------- #
        bench = tasks.SimultaneousFunctionDescent('rosen', log_scale=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - rosenbrock', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('rosenabs', log_scale=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - rosenbrock abs', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('dipole').to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - dipole', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('rastrigin').to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - rastrigin', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('around', log_scale=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - around', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('spiral', log_scale=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - spiral', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)

        bench = tasks.SimultaneousFunctionDescent('oscillating', log_scale=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, '2D simultaneous - oscillating', passes=2000, sec=30, metrics='train loss', vid_scale=2, fps=60)


    def render(self, axsize=(6,3), dpi=300, extra_references: str | Sequence | None = None, n_best:int=1):
        from .plotting import REFERENCE_OPTS, render_summary

        if extra_references is None: extra_references = []
        if isinstance(extra_references, str): extra_references = [extra_references]
        reference_opts = list(REFERENCE_OPTS) + [r for r in extra_references if r not in REFERENCE_OPTS]

        dir = self.summaries_root
        if not os.path.exists(dir): os.mkdir(dir)

        render_summary(
            self.root,
            dirname=os.path.join(dir, f"{to_valid_fname(self.sweep_name)}"),
            main=self.sweep_name,
            references=reference_opts,
            n_best=n_best,
            axsize=axsize, dpi=dpi,
        )



def _maybe_format(x):
    if isinstance(x, float): return format_number(x, 3)
    return x

def _dict_to_str(d: dict):
    return ' '.join([f"{k}={_maybe_format(v)}" for k,v in d.items()])

def print_task_summary(task_name:str, metric: str = "train loss", maximize=False, root: str = "optimizers",) -> None:
    task = Task.load(os.path.join(root, task_name), load_loggers=False, decoder=None)
    sweeps = task.best_sweeps(metric, maximize, n=1000)
    runs = [s.best_runs(metric, maximize, n=1)[0] for s in sweeps]

    for i, r in enumerate(runs):
        key = 'max' if maximize else 'min'
        if len(r.hyperparams) == 0: n = f"{i}: {r.run_name}"
        else: n = f"{i}: {r.run_name} ({_dict_to_str(r.hyperparams)})"
        print(n.ljust(100)[:100], f"{format_number(r.stats[metric][key], 5)}")

