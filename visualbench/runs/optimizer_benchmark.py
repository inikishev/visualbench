import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import gpytorch
import gpytorch.kernels as gk
import torch
from accelerate import Accelerator
from sklearn.datasets import load_breast_cancer
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
        no_tuning: bool = False,

        # storage
        root: str = "optimizers",
        print_records: bool = True,
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
    ):
        self.root = root
        self.sweep_name = sweep_name
        self.hyperparam = hyperparam

        def run_bench(bench: "Benchmark", task_name: str, passes: int, sec: float, metrics:str | Sequence[str] | dict[str, bool], binary_mul: float = 1, test_every: int | None = None):
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

            if hyperparam is None or no_tuning:
                single_run(logger_fn, metrics=metrics, fixed_hyperparams=fixed_hyperparams, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, print_progress=print_progress, save=save, load_existing=load_existing)

            else:
                mbs_search(logger_fn, metrics=metrics, search_hyperparam=hyperparam, fixed_hyperparams=fixed_hyperparams, log_scale=log_scale, grid=grid, step=step, num_candidates=num_candidates, num_binary=max(1, int(num_binary*binary_mul)), num_expansions=num_expansions, rounding=rounding, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, save=save, load_existing=load_existing, print_progress=print_progress)


        self.run_bench = run_bench

    def _run_test(self):
        bench = tasks.Inverse(data.ATTNGRAD96, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Inverse L1', passes=2_000, sec=30, metrics='train loss')

        bench = tasks.Inverse(data.ATTNGRAD96, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Inverse L2', passes=2_000, sec=30, metrics='train loss')

    def _run_test2(self):
        bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[10, 10, 10, 10], act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLP(10-10-10-10-10) - MNIST-1D full-batch", passes=2_000, sec=60, metrics = LOSSES)

    def run(self, ML=True, synthetic=True, stochastic=True, losses=True):
        if ML:
            self.run_ML()
            if stochastic: self.run_ML_stochastic()

        if synthetic:
            self.run_synthetic()
            #if stochastic: self.run_synthetic_stochastic()

        if losses:
            self.run_losses()


    def run_ML(self):
        # ---------------------------- Gaussian processes ---------------------------- #
        # ndim = 7
        # 1.7s. ~ 34s.
        bench = tasks.GaussianProcesses(
            'chaotic_potential',
            200,
            grid=False,
            mean = gpytorch.means.LinearMean(2),
            covar=lambda: gk.ScaleKernel(gk.RBFKernel(2)),
            noise=0.05,

        ).to(CUDA_IF_AVAILABLE) # 7
        self.run_bench(bench, 'ML - RBF Gaussian Processes', passes=400, sec=30, metrics='train loss')

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = 600
        # 4.4s. ~ 1m. 30s.
        bench = tasks.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Alpha Evolve B1', passes=4_000, sec=60, metrics='train loss')

        # ----------------------------------- t-SNE ---------------------------------- #
        # ndim = 1,138
        # 3.7s. ~ 1m. 12s.
        bench = tasks.TSNE(load_breast_cancer().data).to(CUDA_IF_AVAILABLE) # type:ignore #pylint:disable=no-member
        self.run_bench(bench, 'ML - t-SNE', passes=2_000, sec=60, metrics='train loss') # 4.4s. ~ 1m. 30s.

        # ------------------------------- Graph layout ------------------------------- #
        # ndim = 128
        # 3.8s. ~ 1m. 16s.
        bench = tasks.GraphLayout(tasks.GraphLayout.GRID()).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Graph layout optimization', passes=2_000, sec=60, metrics='train loss') # 4.4s. ~ 1m. 30s.

        # ---------------------------- Logistic regression --------------------------- #
        # ndim = 163,880
        # 0.5s ~ 10s.
        bench = tasks.datasets.OlivettiFaces(models.MLP(4096, 40, hidden=None)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Olivetti Faces - Logistic Regression', passes=400, sec=30, metrics=LOSSES)

        # # ------------------------------ MAE regression ------------------------------ #
        # # ndim = 11
        # # ?
        # bench = tasks.datasets.Friedman1(
        #     models.MLP(10, 1, hidden=None), criterion=F.l1_loss, normalize_x=False, normalize_y=False
        # ).to(CUDA_IF_AVAILABLE)
        # self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - L1', passes=2000, sec=30, metrics=LOSSES)
        # not interpretable

        # ---------------------- Small MLP (full-batch MNIST-1D) --------------------- #
        # ndim = 850
        # 3s. ~ 1m.
        bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[10, 10, 10, 10], act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - MLP(10-10-10-10-10)", passes=2_000, sec=60, metrics = LOSSES)

        # ------------------------- MLP (full-batch MNIST-1D) ------------------------- #
        # ndim = 6,970
        # 3.6s. ~ 1m. 16s.
        bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[40,40,40,40], act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - MLP(40-40-40-40-10)", passes=2_000, sec=60, metrics = LOSSES)

        # -------------------- Recurrent MLP (full-batch MNIST-1D) ------------------- #
        # ndim = 2,410
        # 3.6s. ~ 1m. 16s.
        bench = tasks.datasets.Mnist1d(models.RecurrentMLP(40, 10, width=40, n_passes=5, act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - RecurrentMLP", passes=2_000, sec=60, metrics = LOSSES)

        # ---------------------- NeuralODE (full-batch MNIST-1D) --------------------- #
        # ndim = 2,050
        # 3.5s ~ 1m. 10s.
        bench = tasks.datasets.Mnist1d(NeuralODE(40, 10, width=40, act_cls=nn.Softplus)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - NeuralODE", passes=2_000, sec=60, metrics = LOSSES)

        # --------------------- Thin ConvNet (full-batch MNIST-1D) -------------------- #
        # ndim = 1,338
        # 9.5s. ~ 3m.
        bench = tasks.datasets.Mnist1d(models.mnist1d.TinyLongConvNet()).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - ThinConvNet", passes=2_000, sec=120, metrics = LOSSES)

        # ------------------------- GRU (full-batch MNIST-1D) ------------------------ #
        # ndim = 1,510
        # 11s. ~ 3m. 40s.
        bench = tasks.datasets.Mnist1d(models.RNN(1, 10, hidden_size=10, num_layers=2, rnn=nn.GRU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - GRU", passes=2_000, sec=120, metrics = LOSSES)

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = 49,152
        # 14s. ~ 4m. 40s.
        # 9+4=13 ~ 3m.
        bench = tasks.StyleTransfer(data.SANIC96, data.get_qrcode()).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Style Transfer', passes=2_000, sec=120, metrics='train loss', binary_mul=0.4)

        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - PINN (Wave PDE)', passes=2_000, sec=240, metrics='train loss', binary_mul=0.3)


    def run_losses(self):
        # ----------------------------------- LInf ----------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.linf_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - L-Infinity', passes=2000, sec=30, metrics=LOSSES)

        # ---------------------------------- Median ---------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.median_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Median', passes=2000, sec=30, metrics=LOSSES)

        # ---------------------------------- Quartic --------------------------------- #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.quartic_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Quartic', passes=2000, sec=30, metrics=LOSSES)


        # ------------------------------- Quartic rooot ------------------------------ #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100, criterion=losses_.qrmse_loss, normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - Quartic root', passes=2000, sec=30, metrics=LOSSES)

        # ------------------------------------ L4 ------------------------------------ #
        # ndim = 101
        # 3.4s. ~ 1m. 8s.
        bench = tasks.datasets.Friedman1(
            models.MLP(100, 1, hidden=None), n_features=100,
            criterion=partial(losses_.norm_loss, ord=4), normalize_x=False, normalize_y=False
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Friedman 1 - Linear Regression - L4', passes=2000, sec=30, metrics=LOSSES)



    def run_ML_stochastic(self):
        # ------------------------ Online logistic regression ------------------------ #
        # ndim = 385
        # 7.5s. ~ 2m. 30s.
        bench = tasks.datasets.Covertype(models.MLP(54, 7, hidden=None), batch_size=1).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - Covertype - Online Logistic Regression', passes=4_000, sec=60, test_every=10, metrics='test loss')

        # ------------------------------- MLP (MNIST-1D) ------------------------------ #
        # ndim = 56,874
        # ?
        bench = tasks.datasets.Mnist1d(
            models.MLP(40, 10, hidden=[64,96,128,256], act_cls=nn.ELU),
            batch_size=64
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLS - MNIST-1D BS-64 - MLP(64-96-128-256-10)", passes=4_000, sec=120, test_every=10, metrics = "test loss")

        # ----------------------------- ConvNet (MNIST-1D) ---------------------------- #
        # ndim = 134,410
        # 19s. ~ 7m.
        bench = tasks.datasets.Mnist1d(
            models.mnist1d.ConvNet(dropout=0.5),
            batch_size=32,
            test_batch_size=512
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D BS-32 - ConvNet', passes=6_000, sec=360, test_every=20, metrics='test loss')

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.datasets.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D BS-128 - RNN(2x40)', passes=4_000, sec=120, test_every=20, metrics='test loss')


    def run_synthetic(self):
        # ---------------------------- Diabolical Function --------------------------- #
        # ndim = 512
        # 1.6s. ~ 32s.
        bench = tasks.IllConditioned(c=1.99999).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Ill conditioned quadratic', passes=2_000, sec=30, metrics='train loss')

        # ------------------------------- Colorization ------------------------------- #
        # ndim  = 24,576
        # 2.7s. ~ 54s.
        bench = tasks.Colorization.snake().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Colorization', passes=2_000, sec=30, metrics='train loss')

        # ------------------------- Colorization (2nd order) ------------------------- #
        # ndim  = 1024
        # 3.2s. ~ 1m. 4s.
        bench = tasks.Colorization.small(order=2).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Colorization (2nd order)', passes=2_000, sec=30, metrics='train loss')

        # -------------------------------- Rosenbrock -------------------------------- #
        # ndim = 512
        # ?
        bench = tasks.Rosenbrock().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Rosenbrock', passes=2_000, sec=30, metrics='train loss')

        # # -------------------------------- LogSumAExp -------------------------------- #
        # # ndim = 512
        # # 2.3s. ~ 46s.
        # bench = tasks.LogSumExp().to(CUDA_IF_AVAILABLE)
        # self.run_bench(bench, 'S - LogSumAExp', passes=2_000, sec=30, metrics='train loss')

        # -------------------------------- Inverse L1 -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.Inverse(data.ATTNGRAD96, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse - L1', passes=2_000, sec=30, metrics='train loss')

        # -------------------------------- Inverse L2 -------------------------------- #
        # ndim = ?
        # ?
        bench = tasks.Inverse(data.ATTNGRAD96, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse - L2', passes=2_000, sec=30, metrics='train loss')

        # ----------------------------- Matrix idempotent ---------------------------- #
        # ndim = 27,648
        # 8.2s ~ 2m. 44s.
        bench = tasks.MatrixIdempotent(data.SANIC96, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Matrix idempotent', passes=2_000, sec=30, metrics='train loss')

        # -------------------------- Normal scalar curvature ------------------------- #
        # ndim = 16,384
        # 4.4s ~ 1m. 28s.
        bench = tasks.NormalScalarCurvature().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Normal scalar curvature', passes=2_000, sec=30, metrics='train loss')

        # ------------------------------- Kato problem ------------------------------- #
        # ndim = 9,216
        # 1.7s ~ 34s.
        bench = tasks.Kato(data.get_maze()).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Kato problem', passes=2_000, sec=30, metrics='train loss')


    def run_synthetic_stochastic(self):
        # --------------------------- Stochastic inverse L1 -------------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticInverse(data.ATTNGRAD96, vec=True, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic inverse - L1', passes=2_000, sec=60, metrics='train loss')

        # --------------------------- Stochastic inverse L2 -------------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticInverse(data.ATTNGRAD96, vec=True, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic inverse - L2', passes=2_000, sec=60, metrics='train loss')

        # ------------------------ Stochastic matrix recovery ------------------------ #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixRecovery(data.SANIC96, vec=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix recovery', passes=2_000, sec=60, metrics='train loss')

        # ----------------------- Stochastic matrix idempotent ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixIdempotent(data.SANIC96, n=10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix idempotent', passes=2_000, sec=60, metrics='train loss')

        # ----------------------- Stochastic matrix idempotent (hard) ----------------------- #
        # ndim = ?
        # ?
        bench = tasks.StochasticMatrixIdempotent(data.SANIC96, n=10, vec=True).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - Stochastic matrix idempotent (hard)', passes=4_000, sec=60, metrics='train loss')

    def render(self, axsize=(6,3), dpi=300, extra_references: str | Sequence | None = None, n_best:int=1):
        from .plotting import REFERENCE_OPTS, render_summary_v2

        if extra_references is None: extra_references = []
        if isinstance(extra_references, str): extra_references = [extra_references]
        reference_opts = list(REFERENCE_OPTS) + [r for r in extra_references if r not in REFERENCE_OPTS]


        dir = f"{self.root} - summaries"
        if not os.path.exists(dir): os.mkdir(dir)

        render_summary_v2(
            self.root,
            dirname=os.path.join(dir, f"{to_valid_fname(self.sweep_name)}"),
            main=self.sweep_name,
            hyperparams=self.hyperparam,
            references=reference_opts,
            n_best=n_best,
            axsize=axsize, dpi=dpi,
        )

    def render_single_image(self, axsize=(6,3), dpi=300, format='png'):
        if format.startswith('.'): format = format[1:]
        from .plotting import render_summary

        render_summary(
            self.root,
            fname=f"{to_valid_fname(self.sweep_name)}.{format}",
            main=self.sweep_name,
            hyperparams=self.hyperparam,
            axsize=axsize, dpi=dpi,
        )
