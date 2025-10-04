"""not for laptop"""
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .. import models, tasks
from ..models.ode import NeuralODE
from ..utils import CUDA_IF_AVAILABLE
from .benchpack import OptimizerBenchPack

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")

class MBSOptimizerBenchmark(OptimizerBenchPack):
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
        skip:str | Sequence[str] | None = None,

        # storage
        root: str = "tinybench",
        print_records: bool = True,
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = False,

        # pass stuff
        num_extra_passes: float | Callable[[int], float] = 0,
        step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,

        init_fn = lambda opt_fn, bench, value: opt_fn([p for p in bench.parameters() if p.requires_grad], value)
    ):
        kwargs = locals().copy()
        del kwargs["self"]
        super().__init__(**kwargs)

    def run(self):
        self.run_ml()
        self.run_mls()

    def run_ml(self):
        """non-stochastic ML tasks"""
        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - FLS', passes=10_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)

    def run_mls(self):
        """stochastic ML tasks"""
        # ------------------------ Online Logistic regression ------------------------ #
        # ndim = 385
        # 5s. ~ 1m. 40s.
        bench = tasks.datasets.Covertype(models.MLP([54, 7]), batch_size=1).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Covertype BS-1 - Online Logistic Regression'
        self.run_bench(bench, bench_name, passes=10_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # --------------------------- Matrix factorization --------------------------- #
        # ...
        bench = tasks.MFMovieLens("/var/mnt/hdd/datasets/MovieLens 100K", batch_size=32, device='cuda').cuda()
        bench_name = 'MLS - MovieLens BS-32 - Matrix Factorization'
        self.run_bench(bench, bench_name, passes=10_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # ------------------------------- MLP (MNIST-1D) ------------------------------ #
        # ndim = 56,874
        # 9.4s ~ 2m. 28s.
        bench = tasks.datasets.Mnist1d(
            models.MLP([40, 64, 96, 128, 256, 10], act_cls=nn.ELU),
            batch_size=64
        ).to(CUDA_IF_AVAILABLE)
        bench_name = "MLS - MNIST-1D BS-64 - MLP(40-64-96-128-256-10)"
        self.run_bench(bench, bench_name, passes=10_000, sec=120, test_every=20, metrics = "test loss", vid_scale=None, binary_mul=0.75)

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.datasets.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128,
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MNIST-1D BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=10_000, sec=120, test_every=20, metrics='test loss', vid_scale=None, binary_mul=0.5)

        # ---------------------------- ConvNet (MNIST-1D) ---------------------------- #
        # ndim = 134,410
        bench = tasks.datasets.Mnist1d(
            models.vision.ConvNet(40, 1, 10, act_cls=nn.ELU, dropout=0.7),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLS - MNIST-1D BS-32 - ConvNet", passes=10_000, sec=60, test_every=50, metrics = "test loss", vid_scale=None)

        # ----------------------- Sparse Autoencoder (MNIST-1D) ---------------------- #
        # 8.0s ~ 2m. 30s.
        bench = tasks.datasets.Mnist1dAutoencoding(
            models.vision.ConvNetAutoencoder(1, 1, 1, 40, hidden=(64,96,128,256), sparse_reg=0.1),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet', passes=10_000, sec=120, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.75)

        # ---------------------------- ConvNet (SynthSeg) ---------------------------- #
        # 18.8s ~ 6m. 12s.
        # 9+3=12 ~ 3m. 44s.
        bench = tasks.datasets.SynthSeg1d(
            models.vision.ConvNetAutoencoder(1, 1, 5, 32, hidden=(64,96,128)),
            num_samples=10_000, batch_size=64, test_batch_size=512
        ).cuda()
        self.run_bench(bench, 'MLS - SynthSeg BS-64 - ConvNet', passes=10_000, sec=240, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.3)
