"""not for laptop"""
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import rtdl_revisiting_models
import torch
from monai.losses.dice import DiceFocalLoss
from torch import nn

from .. import data, models, tasks
from ..utils import CUDA_IF_AVAILABLE
from .benchpack import OptimizerBenchPack
from .colab_utils import load_movie_lens

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")

class MLBench(OptimizerBenchPack):
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
        num_binary: int = 5,
        num_expansions: int = 12,
        rounding = 1,
        fixed_hyperparams: dict | None = None,
        max_dim: int | None = None,
        tune: bool = True,
        skip:str | Sequence[str] | None = None,

        # storage
        root: str = "MLBench",
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
        self.run_ml()
        self.run_mls()


    def run_ml(self):
        torch.manual_seed(0)

        # # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # # ndim = 132,611
        # # 22s. ~ 7m. 20s.
        # # 9+3=12 ~ 4m. 20s.
        # bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3), n_pde=512, n_ic=256, n_bc=256).to(CUDA_IF_AVAILABLE)
        # self.run_bench(bench, 'ML - Wave PDE - FLS', passes=10_000, sec=600, metrics='train loss', vid_scale=4)

    def run_mls(self):
        torch.manual_seed(0)

        # ------------------------ Online Logistic regression ------------------------ #
        # ndim = 385
        # 5s. ~ 1m. 40s.
        bench = tasks.Collinear(models.MLP([32, 10]), batch_size=1).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Ill-conditioned logistic regression BS-1'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=50, metrics='test loss', vid_scale=None)

        # --------------------------- Matrix factorization --------------------------- #
        # ...
        path = "/var/mnt/hdd/datasets/MovieLens 100K"
        if not os.path.exists(path):
            path = "MovieLens-100k/ml-100k"
            if not os.path.exists(path):
                path = load_movie_lens()
        bench = tasks.MFMovieLens(path, batch_size=32, device='cuda').to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MovieLens BS-32 - Matrix Factorization'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=50, metrics='test loss', vid_scale=None)

        # ------------------------------ MLP (Colinear) ------------------------------ #
        model = models.MLP([32, 64, 96, 128, 256, 10])
        bench = tasks.Collinear(model, batch_size=64, test_batch_size=4096).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Colinear BS-64 - MLP(32-64-96-128-256-10)'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=100, metrics='test loss', vid_scale=None)

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128,
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Mnist1d-5_000 BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=20, metrics='test loss', vid_scale=None)

        # ------------------------- FTTransformer (MNIST-1D) ------------------------- #
        class NoCat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = rtdl_revisiting_models.FTTransformer(n_cont_features=40, cat_cardinalities=[], d_out=10,
                    **rtdl_revisiting_models.FTTransformer.get_default_kwargs(1))

            def forward(self, x):
                return self.model.forward(x, None)

        bench = tasks.Mnist1d(NoCat(), batch_size=32, test_batch_size=1024, num_samples=20_000).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Mnist1d-20_000 BS-32 - FTTransformer'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=200, metrics='test loss', vid_scale=None)

        # ---------------------------- ConvNet (MNIST-1D) ---------------------------- #
        # ndim = 134,410
        bench = tasks.Mnist1d(
            models.vision.ConvNet(40, 1, 10, widths=(64, 128, 256), dropout=0.7),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        bench_name = "MLS - Mnist1d-5_000 BS-32 - ConvNet"
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=50, metrics = "test loss", vid_scale=None)

