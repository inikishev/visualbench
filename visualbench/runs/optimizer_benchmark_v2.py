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

class MBSOptimizerBenchmark:
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
        root: str = "optimizers",
        print_records: bool = True,
        print_progress: bool = True,
        save: bool = True,
        accelerate: bool = True,
        load_existing: bool = True,
        render_vids: bool = True,

        # pass stuff
        num_extra_passes: float | Callable[[int], float] = 0,
        step_callbacks: "Callable[[Benchmark], Any] | Sequence[Callable[[Benchmark], Any]] | None" = None,
    ):
        if skip is None: skip = ()
        if isinstance(skip, str): skip = (skip, )

        self.root = root
        self.sweep_name = sweep_name
        self.summaries_root = f"{self.root} - summaries"
        self.summary_dir = os.path.join(self.summaries_root, f"{to_valid_fname(self.sweep_name)}")
        self.hyperparam = hyperparam

        def run_bench(bench: "Benchmark", task_name: str, passes: int, sec: float, metrics:str | Sequence[str] | dict[str, bool], vid_scale:int|None, fps=60, binary_mul: float = 1, test_every: int | None = None):
            if task_name in skip: return
            dim = sum(p.numel() for p in bench.parameters() if p.requires_grad)
            if max_dim is not None and dim > max_dim: return

            clean_mem()

            if accelerate and next(bench.parameters()).is_cuda: # skip CPU because accelerator state can't change.
                accelerator = Accelerator()
                bench = accelerator.prepare(bench)

            def logger_fn(value: float):
                if dim > 10_000: clean_mem()
                bench.reset().set_benchmark_mode().set_print_inverval(None)
                opt = opt_fn([p for p in bench.parameters() if p.requires_grad], value)
                bench.run(opt, passes, max_seconds=sec, test_every_forwards=test_every, num_extra_passes=num_extra_passes, step_callbacks=step_callbacks)
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

    def run_linalg(self):
        # ------------------------------- Inverse-16 L1 ------------------------------ #
        # SOAP, PSGD, NAG, Muon, Adam, AdamW, BFGS-Backtracking. SOAP/PSGD are 0.08, Adam 0.10, BFGS is 0.12.
        bench = tasks.Inverse(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Inverse-16 L1', passes=2000, sec=60, metrics='train loss', vid_scale=None)

        # ------------------------- StochasticInverse-16 MSE ------------------------- #
        # ShorR(???), SOAP, NAG, Muon, SGD, LMAdagrad, Adagrad, PSGD. Interesting. ShorR is 0.021, SOAP 0.023, Adam is 0.033.
        bench = tasks.StochasticInverse(16, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'StochasticInverse-16 MSE', passes=2000, sec=60, metrics='test loss', vid_scale=None)

        # --------------------------- MatrixLogarithm-16 LR -------------------------- #
        # smooth, PSGD 0.02, SOAP 0.03, AdamW 0.05
        bench = tasks.MatrixLogarithm(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MatrixLogarithm-16 L1', passes=2000, sec=60, metrics='train loss', vid_scale=None)


        # maybe

        # ------------------------------ Inverse-16 MSE ------------------------------ #
        # AdaptiveHeavyBall, Newton and QN with up to 1e-13  is good, Adam 5e-4, SOAP 5e-5. Maybe keep for convex testing
        bench = tasks.Inverse(16, criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Inverse-16 MSE', passes=2000, sec=60, metrics='train loss', vid_scale=None)

        # ---------------------------- MoorePenrose-16 L1 ---------------------------- #
        # weird mix, but reasonably big spacing between algos, so maybe as a weirder kind of problem with clean lr to loss curve, best is Adam
        bench = tasks.MoorePenrose(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MoorePenrose-16 L1', passes=2000, sec=60, metrics='train loss', vid_scale=None)

        # ---------------------------- Drazin-fielder16 L1 --------------------------- #
        # hard, only few managed to reach 2 - LBFGS and ShorR. Then we have BFGS with 1348, Adam has 2233
        bench = tasks.Drazin(data.get_fielder(16)[0], criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Drazin-fielder16 L1', passes=2000, sec=60, metrics='train loss', vid_scale=None)

        # -------------------------- StochasticRLstsq-10 MSE ------------------------- #
        # smooth, big gaps, Adagrad is best, not sure if this is a good proxy for generalization
        bench = tasks.StochasticRLstsq(10, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'StochasticRLstsq-10 MSE', passes=2000, sec=60, metrics='train loss', vid_scale=None)


    def run_visual(self):
        # ------------------------------- neural drawer ------------------------------ #
        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=nn.ReLU, bn=True), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ReLU+bn', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=nn.ELU), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ELU', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

        bench = tasks.NeuralDrawer(data.SPIRAL96, models.MLP(2, 3, [16,16,16,16,16,16,16], act_cls=models.act.Sine), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Sine', passes=2000, sec=60, metrics='train loss', vid_scale=2, fps=30)

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

        # ----------------------- Sine Approximator - Tanh 7-4 ---------------------- #
        # ndim = 15
        # 4.2s ~ 1m. 24s.
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=7, resolution=(384,768),
        ) # NO CUDA

        self.run_bench(bench, 'Visual - Sine Approximator - Tanh 7-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # ----------------------- Particle minmax ---------------------- #
        # ndim = 64
        # 2s ~ 40s
        bench = tasks.ClosestFurthestParticles(32, spread=0.75) # NO CUDA
        self.run_bench(bench, 'Visual - Particle min-max', passes=2_000, sec=60, metrics='train loss', vid_scale=1)

    def run_visual_extra(self):
        # ----------------------------------- moons ---------------------------------- #
        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2]),).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2], act_cls=nn.ReLU, bn=True)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ReLU+bn', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP(2,1,[2,2,2,2,2,2,2]), batch_size=16, n_samples=2048, test_split=1024).to(CUDA_IF_AVAILABLE)
        bench_name= "Visual - Moons BS-16 - MLP(2-2-2-2-2-2-2-2-1)-ELU"
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics='test loss', vid_scale=2, test_every=1)

        # ------------------------------- lines drawer ------------------------------- #
        bench = tasks.LinesDrawer(data.WEEVIL96, 100, loss=_unbatched_ssim).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - LinesDrawer SSIM', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ----------------------------- partition drawer ----------------------------- #
        bench = tasks.PartitionDrawer(data.WEEVIL96, 100).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - PartitionDrawer', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ------------------------- Colorization (1.3th power) ------------------------- #
        bench = tasks.Colorization.small(power=1.3).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (1.3th power)', passes=2_000, sec=60, metrics='train loss', vid_scale=8)

        # ----------------------- Sine Approximator - LeakyReLU 10-4 ---------------------- #
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=10, act=F.leaky_relu, resolution=(384,768),
        ) # NO CUDA
        self.run_bench(bench, 'Visual - Sine Approximator - LeakyReLU 10-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)


    def run_2d(self):
        bench = tasks.FunctionDescent('booth')
        self.run_bench(bench, '2D - booth', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('ill')
        self.run_bench(bench, '2D - ill', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('rosen10')
        self.run_bench(bench, '2D - rosenbrock-10', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1000, sec=30, metrics='train loss', vid_scale=1, fps=30)

        bench = tasks.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2000, sec=60, metrics='train loss', vid_scale=1, fps=30)

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


