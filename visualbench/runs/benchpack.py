import os
import random
from collections.abc import Callable, Iterable, Sequence
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..utils.clean_mem import clean_mem
from ..utils.python_tools import to_valid_fname
from .run import _target_metrics_to_dict, mbs_search, single_run

if TYPE_CHECKING:
    from ..benchmark import Benchmark

# non-required imports
if find_spec("accelerate") is not None:
    from accelerate import Accelerator
else:
    Accelerator = None

LOSSES = ("train loss", "test loss")

class OptimizerBenchPack:
    def __init__(
        self,
        opt_fn: Callable,
        sweep_name: str,
        root: str | None,

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
        print_records: bool = True,
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = True,

        # pass stuff
        num_extra_passes: float | Callable[[int], float] = 0,
        step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,

        init_fn = lambda opt_fn, bench, value: opt_fn([p for p in bench.parameters() if p.requires_grad], value)
    ):
        if skip is None: skip = ()
        if isinstance(skip, str): skip = (skip, )
        self.sweep_name = sweep_name

        self.root = root
        if self.root is None:
            self.summaries_root = None
            self.summary_dir = None
        else:
            self.summaries_root = f"{self.root} - summaries"
            self.summary_dir = os.path.join(self.summaries_root, f"{to_valid_fname(self.sweep_name)}")

        self.hyperparam = hyperparam
        self.tune = tune
        self.opt_fn = opt_fn
        self.init_fn = init_fn

        def run_bench(bench: "Benchmark", task_name: str, passes: int, sec: float, metrics:str | Sequence[str] | dict[str, bool], vid_scale:int|None, fps=60, binary_mul: float = 1, test_every: int | None = None):
            if task_name in skip: return
            dim = sum(p.numel() for p in bench.parameters() if p.requires_grad)
            if max_dim is not None and dim > max_dim: return

            clean_mem()

             # skip CPU because accelerator state can't change.
            if (accelerate) and (Accelerator is not None) and (next(bench.parameters()).is_cuda):
                accelerator = Accelerator()
                bench = accelerator.prepare(bench)

            def logger_fn(value: float):
                if dim > 100_000: clean_mem()

                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)

                bench.reset().set_performance_mode().set_print_inverval(None)
                opt = init_fn(opt_fn, bench, value)
                bench.run(opt, max_passes=passes, max_seconds=sec, test_every_forwards=test_every, num_extra_passes=num_extra_passes, step_callbacks=step_callbacks)
                if print_progress and bench.seconds_passed is not None and bench.seconds_passed > sec:
                    print(f"{sweep_name}: '{task_name}' timeout, {bench.seconds_passed} > {sec}!")
                return bench.logger

            if (hyperparam is None) or (not tune):
                sweep = single_run(logger_fn, metrics=metrics, fixed_hyperparams=fixed_hyperparams, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, print_progress=print_progress, save=save, load_existing=load_existing)

            else:
                sweep = mbs_search(logger_fn, metrics=metrics, search_hyperparam=hyperparam, fixed_hyperparams=fixed_hyperparams, log_scale=log_scale, grid=grid, step=step, num_candidates=num_candidates, num_binary=max(1, int(num_binary*binary_mul)), num_expansions=num_expansions, rounding=rounding, root=root, task_name=task_name, run_name=sweep_name, print_records=print_records, save=save, load_existing=load_existing, print_progress=print_progress)

            # render video
            if (render_vids) and (vid_scale is not None) and (self.summaries_root is not None):
                assert self.summary_dir is not None
                for metric, maximize in _target_metrics_to_dict(metrics).items():
                    video_path = os.path.join(self.summary_dir, f'{task_name} - {metric}')
                    if os.path.exists(f'{video_path}.mp4'): continue

                    best_run = sweep.best_runs(metric, maximize, 1)[0]
                    value = 0
                    if tune and hyperparam is not None: value = best_run.hyperparams[hyperparam]
                    bench.reset().set_performance_mode(False).set_print_inverval(None)
                    opt = init_fn(opt_fn, bench, value)
                    bench.run(opt, max_passes=passes, max_seconds=sec, test_every_forwards=test_every, num_extra_passes=num_extra_passes)
                    if not os.path.exists(self.summaries_root): os.mkdir(self.summaries_root)
                    if not os.path.exists(self.summary_dir): os.mkdir(self.summary_dir)
                    bench.render(f'{video_path} __TEMP__', scale=vid_scale, fps=fps, progress=False)
                    os.rename(f'{video_path} __TEMP__.mp4', f'{video_path}.mp4')


        self.run_bench = run_bench


    def render(self, axsize=(6,3), dpi=300, extra_references: str | Sequence | None = None, n_best:int=1):
        from .plotting import REFERENCE_OPTS, render_summary

        if extra_references is None: extra_references = []
        if isinstance(extra_references, str): extra_references = [extra_references]
        reference_opts = list(REFERENCE_OPTS) + [r for r in extra_references if r not in REFERENCE_OPTS]

        if self.summaries_root is None:
            raise RuntimeError("rendering only supported when `root` is speficied")

        assert self.root is not None
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




