# pylint: disable = unnecessary-lambda
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
import torchzero as tz
from accelerate import Accelerator
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


tz.enable_compilation(True)


LOSSES = ("train loss", "test loss")

class MBSBenchmarkBenchmark:
    def __init__(
        self,
        benchmark: "Benchmark",
        task_name: str,

        # task params
        passes: int,
        sec: float,
        metrics:str | Sequence[str] | dict[str, bool],
        vid_scale:int|None,
        fps=60,
        binary_mul: float = 1,
        test_every: int | None = None,
        yscale: Any = None,

        # MBS parameters
        skip:str | Sequence[str] | None = None,

        # storage
        root: str = "benchmarks",
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = False,
    ):
        # sweep_name: str,
        # num_extra_passes: float | Callable[[int], float] = 0,
        # step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,

        dim = sum(p.numel() for p in benchmark.parameters() if p.requires_grad)
        if skip is None: skip = ()
        if isinstance(skip, str): skip = (skip, )

        self.root = root
        self.task_name = task_name
        self.summaries_root = f"{self.root} - summaries"
        self.summary_dir = os.path.join(self.summaries_root, f"{to_valid_fname(self.task_name)}")
        self.metrics = _target_metrics_to_dict(metrics)
        self.yscale = yscale
        self.passes = passes
        self.benchmark = benchmark


        def run_optimizer(
            opt_fn: "Callable",
            sweep_name: str,
            tune:bool,
            max_dim:int|None,
            num_extra_passes:float|Callable[[int],float]=0,
            step_callbacks=None,
            hyperparam: str | None = "lr",
            log_scale: bool = True,
            grid: Iterable[float] = (2, 1, 0, -1, -2, -3, -4, -5),
            step: float = 1,
            num_candidates: int = 2,
            num_binary: int = 12,
            num_expansions: int = 12,
            rounding=1,
            fixed_hyperparams: dict | None = None,
        ):
            bench = benchmark
            if sweep_name in skip: return
            if max_dim is not None and dim > max_dim: return
            clean_mem()

            if accelerate and next(bench.parameters()).is_cuda: # skip CPU because accelerator state can't change.
                accelerator = Accelerator()
                bench = accelerator.prepare(bench)

            def logger_fn(value: float):
                if dim > 10_000: clean_mem()
                bench.reset().set_performance_mode().set_print_interval(None)
                opt = opt_fn([p for p in bench.parameters() if p.requires_grad], value)
                bench.run(opt, max_passes=passes, max_seconds=sec, test_every_forwards=test_every, num_extra_passes=num_extra_passes, step_callbacks=step_callbacks)
                if print_progress and bench.seconds_passed is not None and bench.seconds_passed > sec:
                    print(f"{sweep_name}: '{task_name}' timeout, {bench.seconds_passed} > {sec}!")
                return bench.logger

            if hyperparam is None or (not tune):
                sweep = single_run(logger_fn, metrics=metrics, fixed_hyperparams=fixed_hyperparams, root=root, task_name=task_name, run_name=sweep_name, print_records=False, print_progress=print_progress, save=save, load_existing=load_existing)

            else:
                sweep = mbs_search(logger_fn, metrics=metrics, search_hyperparam=hyperparam, fixed_hyperparams=fixed_hyperparams, log_scale=log_scale, grid=grid, step=step, num_candidates=num_candidates, num_binary=max(1, int(num_binary*binary_mul)), num_expansions=num_expansions, rounding=rounding, root=root, task_name=task_name, run_name=sweep_name, print_records=False, save=save, load_existing=load_existing, print_progress=print_progress)

            # render video
            if render_vids and vid_scale is not None:
                for metric, maximize in _target_metrics_to_dict(metrics).items():
                    video_path = os.path.join(self.summary_dir, f'{task_name} - {metric}')
                    if os.path.exists(f'{video_path}.mp4'): continue

                    best_run = sweep.best_runs(metric, maximize, 1)[0]
                    value = 0
                    if tune and hyperparam is not None: value = best_run.hyperparams[hyperparam]
                    bench.reset().set_performance_mode(False).set_print_interval(None)
                    opt = opt_fn(bench.parameters(), value)
                    bench.run(opt, max_passes=passes, max_seconds=sec, test_every_forwards=test_every)
                    if not os.path.exists(self.summaries_root): os.mkdir(self.summaries_root)
                    if not os.path.exists(self.summary_dir): os.mkdir(self.summary_dir)
                    bench.render(f'{video_path} __TEMP__', scale=vid_scale, fps=fps, progress=False)
                    os.rename(f'{video_path} __TEMP__.mp4', f'{video_path}.mp4')


        self.run_optimizer = run_optimizer

    def quickrun(self):
        opt = lambda p, lr: torch.optim.SGD(p, lr)
        self.run_optimizer(opt, "SGD", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.SGD(p, lr, momentum=0.9, nesterov=True)
        self.run_optimizer(opt, "NAG(0.95)", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adam(p, lr)
        self.run_optimizer(opt, "Adam", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adam(p, lr, betas=(0.95, 0.95))
        self.run_optimizer(opt, "Adam(0.95,0.95)", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adagrad(p, lr)
        self.run_optimizer(opt, "Adagrad", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.RMSprop(p, lr)
        self.run_optimizer(opt, "RMSprop", tune=True, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.SOAP(max_dim=2048), tz.m.LR(lr))
        self.run_optimizer(opt, "SOAP", tune=True, max_dim=None)


    def run(self, stochastic=True, non_stochastic=True, vr=True, qn=True, newton=True, zo=True, noop=True):
        if noop: self.run_noop()
        if stochastic: self.run_stochastic()
        if non_stochastic: self.run_non_stochastic()
        if vr: self.run_vr()
        if qn: self.run_qn()
        if newton: self.run_newton()
        if zo: self.run_zo()

    def run_noop(self):
        opt = lambda p, lr: tz.Optimizer(p, tz.m.LR(0))
        self.run_optimizer(opt, "Noop", tune=False, max_dim=None)

    def run_stochastic(self):
        opt = lambda p, lr: torch.optim.SGD(p, lr)
        self.run_optimizer(opt, "SGD", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.SGD(p, lr, momentum=0.9, nesterov=True)
        self.run_optimizer(opt, "NAG(0.9)", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.SGD(p, lr, momentum=0.99, nesterov=True)
        self.run_optimizer(opt, "NAG(0.99)", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adagrad(p, lr)
        self.run_optimizer(opt, "Adagrad", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.RMSprop(p, lr)
        self.run_optimizer(opt, "RMSprop", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adam(p, lr)
        self.run_optimizer(opt, "Adam", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.Adam(p, lr, betas=(0.95, 0.95))
        self.run_optimizer(opt, "Adam(0.95, 0.95)", tune=True, max_dim=None)

        opt = lambda p, lr: torch.optim.AdamW(p, lr)
        self.run_optimizer(opt, "AdamW", tune=True, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.GGT(), tz.m.LR(lr))
        self.run_optimizer(opt, "GGT", tune=True, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.SOAP(max_dim=2048), tz.m.LR(lr))
        self.run_optimizer(opt, "SOAP", tune=True, max_dim=None)

        # PSGD Kron
        import pytorch_optimizer
        opt = lambda p, lr: pytorch_optimizer.Kron(p, lr, memory_save_mode="smart_one_diag", store_triu_as_line=False)
        self.run_optimizer(opt, "PSGD Kron", tune=True, max_dim=None)

        # Muon
        opt = lambda p, lr: tz.Optimizer(
            p,
            tz.m.NAG(0.95),
            tz.m.Split(
                lambda x: x.ndim >= 2,
                true=tz.m.Orthogonalize(),
                false=[tz.m.Adam(0.9, 0.95), tz.m.Mul(1/66)]),
            tz.m.LR(lr)
        )
        self.run_optimizer(opt, "Muon", tune=True, max_dim=None)

    def run_non_stochastic(self):
        opt = lambda p, lr: tz.Optimizer(p, tz.m.BirginMartinezRestart(tz.m.PolakRibiere()), tz.m.StrongWolfe(a_init='first-order', c2=0.1))
        self.run_optimizer(opt, "PolakRibiere-StrongWolfe", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.AdaptiveHeavyBall())
        self.run_optimizer(opt, "AdaptiveHeavyBall", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.AdGD())
        self.run_optimizer(opt, "AdGD", tune=False, max_dim=None)

        opt = lambda p, lr: torch.optim.Rprop(p, lr)
        self.run_optimizer(opt, "Rprop", tune=True, max_dim=None)

    def run_vr(self):
        opt = lambda p, lr: tz.Optimizer(p, tz.m.SVRG(self.passes//4), tz.m.LBFGS(), tz.m.Backtracking())
        self.run_optimizer(opt, "SVRG-LBFGS-Backtracking", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.Online(tz.m.LBFGS()), tz.m.Backtracking())
        self.run_optimizer(opt, "OnlineLBFGS-Backtracking", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.SVRG(self.passes//4), tz.m.Adam(), tz.m.LR(lr))
        self.run_optimizer(opt, "SVRG-Adam", tune=True, max_dim=None)


    def run_qn(self):
        from torchzero.optim.wrappers.scipy import ScipyMinimize

        opt = lambda p, lr: torch.optim.LBFGS(p, line_search_fn='strong_wolfe')
        self.run_optimizer(opt, "LBFGS", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.LBFGS(), tz.m.RelativeWeightDecay(0.2), tz.m.Backtracking())
        self.run_optimizer(opt, "LBFGS-RelativeWeightDecay-Backtracking", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.BFGS(), tz.m.Backtracking())
        self.run_optimizer(opt, "BFGS-Backtracking", tune=False, max_dim=5000)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.LevenbergMarquardt(tz.m.RestartOnStuck(tz.m.SR1(inverse=False))))
        self.run_optimizer(opt, "LevenbergMarquardt(SR1)", tune=False, max_dim=5000)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.RestartOnStuck(tz.m.ShorR()), tz.m.StrongWolfe(a_init='quadratic'))
        self.run_optimizer(opt, "ShorR-StrongWolfe", tune=False, max_dim=5000)

    def run_newton(self):
        opt = lambda p, lr: tz.Optimizer(p, tz.m.Newton(), tz.m.Backtracking())
        self.run_optimizer(opt, "Newton-Backtracking", tune=False, max_dim=400, num_extra_passes=lambda ndim: ndim+1)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.Newton(eigval_fn=lambda x: x.abs().clip(min=1e-8)), tz.m.Backtracking())
        self.run_optimizer(opt, "SPFN-Backtracking", tune=False, max_dim=400, num_extra_passes=lambda ndim: ndim+1)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.LevenbergMarquardt(tz.m.Newton()))
        self.run_optimizer(opt, "LevenbergMarquardt(Newton)", tune=False, max_dim=400, num_extra_passes=lambda ndim: ndim+1)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.NewtonCGSteihaug(hvp_method='fd_forward'))
        self.run_optimizer(opt, "NewtonCGSteihaug", tune=False, max_dim=None)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.SixthOrder5P())
        self.run_optimizer(opt, "SixthOrder5P", tune=False, max_dim=400, num_extra_passes=lambda ndim: 5*ndim+5)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.experimental.NewtonNewton(), tz.m.Backtracking())
        self.run_optimizer(opt, "NewtonNewton-Backtracking", tune=False, max_dim=50, num_extra_passes=lambda ndim: ndim**2)

        opt = lambda p, lr: tz.Optimizer(p, tz.m.experimental.HigherOrderNewton())
        self.run_optimizer(opt, "HigherOrderNewton", tune=False, max_dim=10, num_extra_passes=lambda ndim: ndim**3)


    def run_zo(self):
        from torchzero.optim.wrappers.scipy import ScipyMinimize
        opt = lambda p, lr: ScipyMinimize(p, 'powell')
        self.run_optimizer(opt, "Powell", tune=False, max_dim=1000)



    def render(self, axsize=(6,3), dpi=300):
        from .plotting import (
            REFERENCE_OPTS,
            bar_chart,
            make_axes,
            plot_sweeps,
            plot_values,
        )

        dir = self.summaries_root
        if not os.path.exists(dir): os.mkdir(dir)

        summary_dir = os.path.join(dir, self.task_name)
        if not os.path.exists(summary_dir): os.mkdir(summary_dir)

        # ----------------------------------- load ----------------------------------- #
        task_path = os.path.join(self.root, self.task_name)
        task = Task.load(task_path, load_loggers=True, decoder=None)

        # ---------------------------------- sweeps ---------------------------------- #
        axes = make_axes(n=len(self.metrics), nrows=len(self.metrics), axsize=axsize, dpi=dpi)
        for ax, (metric, maximize) in zip(axes, self.metrics.items()):
            plot_sweeps(task, metric, maximize, main=None, references=None, n_best=10, yscale=self.yscale, ax=ax)

        plt.savefig(os.path.join(summary_dir, "sweep.png"))
        plt.close()

        # ---------------------------------- losses ---------------------------------- #
        axes = make_axes(n=len(self.metrics), nrows=len(self.metrics), axsize=axsize, dpi=dpi)
        for ax, (metric, maximize) in zip(axes, self.metrics.items()):
            plot_values(task, metric, maximize, main=None, references=None, n_best=10, yscale=self.yscale, ax=ax)

        plt.savefig(os.path.join(summary_dir, "losses.png"))
        plt.close()

        # --------------------------------- bar chart -------------------------------- #
        axes = make_axes(n=len(self.metrics), nrows=len(self.metrics), axsize=axsize, dpi=dpi)
        for ax, (metric, maximize) in zip(axes, self.metrics.items()):
            bar_chart(task, metric, maximize, scale=self.yscale, ax=ax, n=100)

        plt.savefig(os.path.join(summary_dir, "bar.png"))
        plt.close()



