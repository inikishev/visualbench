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

class FastMLBench(OptimizerBenchPack):
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
        root: str = "FastMLBench",
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
        self.run_visual()
        self.run_mlp()
        self.run_rnn()
        self.run_autoencoder()


    def run_visual(self):
        bench = tasks.FunctionDescent('ill2')
        self.run_bench(bench, '2D - ill2', passes=1000, sec=10, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1000, sec=30, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2000, sec=60, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('medianreg1d')
        self.run_bench(bench, '2D - medianreg1d', passes=1000, sec=60, metrics='train loss', vid_scale=1)

        bench = tasks.projected.Rosenbrock(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Projected - Rosenbrock 384', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        bench = tasks.Colorization.small().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

    # ------------------------------ MLP (Colinear) ------------------------------ #
    def run_mlp(self):
        torch.manual_seed(0)

        model = models.MLP([32, 64, 96, 128, 256, 10])
        bench = tasks.Collinear(model, batch_size=64, test_batch_size=4096).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Colinear BS-64 - MLP(32-64-96-128-256-10)'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=100, metrics='test loss', vid_scale=None)

    # ------------------------------- RNN (MNIST-1D) ------------------------------ #
    def run_rnn(self):
        torch.manual_seed(0)

        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128,
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Mnist1d-5_000 BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=20_000, sec=1_000, test_every=20, metrics='test loss', vid_scale=None)

    # --------------------------- Autoencoder (MNIST-1D) --------------------------- #
    def run_autoencoder(self):
        torch.manual_seed(0)

        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.Mnist1dAutoencoding(
            model = models.vision.ConvNetAutoencoder(1, 1, 1, 40, hidden=(64, 64, 64, 64, 2), act_cls=nn.ELU),
            batch_size=128,
            test_batch_size=512,
        ).to(CUDA_IF_AVAILABLE).set_render_every(2)
        bench_name = 'MLS - Mnist1d-5_000 BS-128 - ConvNetAutoencoder(64-64-64-64-2 ELU)'
        self.run_bench(bench, bench_name, passes=10_000, sec=1_000, test_every=100, metrics='test loss', vid_scale=2)
