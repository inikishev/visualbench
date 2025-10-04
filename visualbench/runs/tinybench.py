"""trying to make it work"""
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
        bench = tasks.StochasticMatrixRoot(16, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'StochasticMatrixRoot(16, p=10)', passes=10_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)