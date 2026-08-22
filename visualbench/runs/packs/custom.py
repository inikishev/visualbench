from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from ..benchpack import OptimizerBenchPack

if TYPE_CHECKING:
    from ...benchmark import Benchmark

LOSSES = ("train loss", "test loss")

class CustomBench(OptimizerBenchPack):
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
        skip: str | Sequence[str] | None = None,

        # storage
        root: str = "CustomBench",
        print_records: bool = True,
        print_progress: bool = True,
        print_time: bool = False,
        save: bool = True,
        accelerate: bool = True,
        accelerate_kwargs: dict[str, Any] | None = None,
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

        self.run_fns = []

    def run(self):
        torch.manual_seed(0)
        for run_fn in self.run_fns: run_fn()

    def add_bench(
        self,
        bench: "Benchmark",
        task_name: str,
        passes: int,
        sec: float,
        metrics: str | Sequence[str] | dict[str, bool],
        vid_scale: int | None,
        fps=60,
        binary_mul: float = 1,
        test_every: int | None = None,
    ):
        kwargs = locals().copy()
        del kwargs["self"]

        self.run_fns.append(partial(self.run_bench, **kwargs))
