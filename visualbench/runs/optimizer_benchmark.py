import os
import time
import warnings
from collections import UserDict, UserList
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import gpytorch
import gpytorch.kernels as gk
import msgspec
import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter1d

from .. import data, models, tasks
from ..logger import Logger
from ..utils.format import tonumpy
from ..utils import CUDA_IF_AVAILABLE
from ..utils.python_tools import round_significant
from . import mbs
from .run import mbs_search, single_run

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")
def mbs_optimizer_run(
    opt_fn: Callable,
    run_name: str,

    # MBS parameters
    hyperparam: str | None = "lr",
    log_scale: bool = True,
    grid: Iterable[float] = (3, 2, 1, 0, -1, -2, -3, -4, -5),
    step: float = 1,
    num_candidates: int = 2,
    num_binary: int = 11,
    num_expansions: int = 11,
    rounding=1,

    fixed_hyperparams: dict | None = None,
    max_dims:int | None = None,

    # storage
    root: str = "optimizer benchmarks",
    print_records: bool = True,
    save: bool = True
):
    def run_bench(bench: "Benchmark", task_name: str, passes: int, sec: float, targets:str | Sequence[str] | dict[str, bool], binary_mul: float = 1, test_every: int | None = None):
        if max_dims is not None and sum(p.numel() for p in bench.parameters() if p.requires_grad) > max_dims: return

        def logger_fn(value: float):
            bench.reset().set_benchmark_mode()
            opt = opt_fn(bench.parameters(), value)
            bench.run(opt, passes, max_seconds=sec, test_every_forwards=test_every)
            return bench.logger

        if hyperparam is not None:
            mbs_search(logger_fn, targets=targets, search_hyperparam=hyperparam, fixed_hyperparams=fixed_hyperparams, log_scale=log_scale, grid=grid, step=step, num_candidates=num_candidates, num_binary=max(1, int(num_binary*binary_mul)), num_expansions=num_expansions, rounding=rounding, root=root, task_name=task_name, run_name=run_name, print_records=print_records, save=save)

        else:
            single_run(logger_fn, targets=targets, fixed_hyperparams=fixed_hyperparams, root=root, task_name=task_name, run_name=run_name, print_records=print_records, save=save)

    # ---------------------------- Diabolical Function --------------------------- #
    # ndim = 512
    # 1.6s. ~ 32s.
    bench = tasks.IllConditioned(c=1.99999).to(CUDA_IF_AVAILABLE)
    run_bench(bench, 'Ill conditioned quadratic', passes=2_000, sec=30, targets='train loss')

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
    run_bench(bench, 'RBF Gaussian Processes', passes=400, sec=30, targets='train loss')

    # ------------------------------ Style transfer ------------------------------ #
    # ndim = 49,152
    # 14s. ~ 4m. 40s.
    # 9+4=13 ~ 3m.
    bench = tasks.StyleTransfer(data.SANIC96, data.get_qrcode()).to(CUDA_IF_AVAILABLE)
    run_bench(bench, 'Style Transfer', passes=2_000, sec=120, targets='train loss', binary_mul=0.4)

    # ------------------------------ PINN (Wave PDE) ----------------------------- #
    # ndim = 132,611
    # 22s. ~ 7m. 20s.
    # 9+3=12 ~ 4m. 20s.
    bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
    run_bench(bench, 'PINN (Wave PDE)', passes=2_000, sec=240, targets='train loss', binary_mul=0.3)

    # ------------------------------ Alpha Evolve B1 ----------------------------- #
    # ndim = 600
    # 4.4s. ~ 1m. 30s.
    bench = tasks.AlphaEvolveB1().to(CUDA_IF_AVAILABLE) # 600
    run_bench(bench, 'Alpha Evolve B1', passes=4_000, sec=60, targets='train loss') # 4.4s. ~ 1m. 30s.

    # ---------------------------- Logistic regression --------------------------- #
    # ndim = 163,880
    # 0.5s ~ 10s.
    bench = tasks.datasets.OlivettiFaces(models.MLP(4096, 40, hidden=None)).to(CUDA_IF_AVAILABLE)
    run_bench(bench, 'Logistic Regression - Olivetti Faces', passes=400, sec=30, targets=LOSSES)

    # ------------------------- MLP (full-batch MNIST-1D) ------------------------- #
    # ndim = 6,970
    # ?
    bench = tasks.datasets.Mnist1d(models.MLP(40, 10, hidden=[40,40,40,40], act=nn.ELU)).to(CUDA_IF_AVAILABLE)
    run_bench(bench, "MLP(40-40-40-40-10) - MNIST-1D full-batch", passes=2_000, sec=60, targets = LOSSES)

    # --------------------- Thin ConvNet (full-batch MNIST-1D) -------------------- #
    # ndim = 1,338
    # 9.5s. ~ 3m.
    bench = tasks.datasets.Mnist1d(models.mnist1d.TinyLongConvNet()).to(CUDA_IF_AVAILABLE)
    run_bench(bench, "ThinConvNet - MNIST-1D full-batch", passes=2_000, sec=120, targets = LOSSES)

    # ------------------------ Online logistic regression ------------------------ #
    # ndim = 385
    # 7.5s. ~ 2m. 30s.
    bench = tasks.datasets.Covertype(models.MLP(54, 7, hidden=None), batch_size=1).to(CUDA_IF_AVAILABLE)
    run_bench(bench, 'Online Logistic Regression - Covertype', passes=4_000, sec=60, test_every=10, targets='test loss')

    # ------------------------------- MLP (MNIST-1D) ------------------------------ #
    # ndim = 56,874
    # ?
    bench = tasks.datasets.Mnist1d(
        models.MLP(40, 10, hidden=[64,96,128,256], act=nn.ELU),
        batch_size=64
    ).to(CUDA_IF_AVAILABLE)
    run_bench(bench, "MLP(64-96-128-256-10) - MNIST-1D bs64", passes=4_000, sec=120, test_every=10, targets = "test loss")

    # ----------------------------- ConvNet (MNIST-1D) ---------------------------- #
    # ndim = 134,410
    # 19s. ~ 7m.
    bench = tasks.datasets.Mnist1d(
        models.mnist1d.ConvNet(dropout=0.5),
        batch_size=32,
        test_batch_size=512
    ).cuda()
    run_bench(bench, 'ConvNet - MNIST-1D bs32', passes=6_000, sec=360, test_every=20, targets='test loss')

    # ------------------------------- RNN (MNIST-1D) ------------------------------ #
    # ndim = 20,410
    # 11s. ~ 3m. 30s.
    bench = tasks.datasets.Mnist1d(
        models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
        batch_size=128
    )
    run_bench(bench, 'RNN(2x40) - MNIST-1D bs128', passes=4_000, sec=120, test_every=20, targets='test loss')

