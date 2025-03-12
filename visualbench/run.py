from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch
from myai.jupyter_tools import clean_mem
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig
from myai.python_tools import performance_context
from torch.nn import functional as F

from ._utils.runs import REFERENCE_OPTS
from ._utils.runs_plotting import _print_best, plot_lr_search_curve, plot_metric
from ._utils.search import _search_for_visualization
from .benchmark import Benchmark
from .data import ATTNGRAD96, QRCODE96, SANIC96, TEST96, get_qrcode, get_randn
from .tasks import (
    QEP,
    QR,
    CaliforniaHousing,
    Convex,
    Eigen,
    FunctionDescent,
    Inverse,
    InverseInverse,
    LUPivot,
    MatrixSign,
    Mnist1d,
    Mnist1dAutoencoder,
    RNNArgsort,
    SelfRecurrent,
    SynthSeg1d,
    Sorting,
    models,
)
from .tasks.linalg._linalg_utils import _full01

_train_loss = {"train loss": False}
_test_train_loss = {"test loss": False, "train loss": False}
_train_test_loss = {"train loss": False, "test loss": False}


def descent_previews(opt_fn: Callable, has_lr = True, fns: dict[str, int] = dict(booth=200,rosen=1000,rastrigin=1000)): # pylint:disable=dangerous-default-value
    ... # TODO


def run_bench(opt_name:str, opt_fn: Callable, show=True, save=True, extra:Sequence[str]|str|None =(), has_lr=True, print_time = False):
    torch.manual_seed(0)
    if extra is None: extra = []
    elif isinstance(extra, str): extra = [extra]
    else: extra = list(extra)
    ref = list(REFERENCE_OPTS) + extra

    randn = get_randn()
    fig = Fig()

    # queue = []
    def _search(
        bench: Benchmark,
        name: str,
        target_metrics: dict[str, bool],
        max_passes: int,
        max_seconds: float,
        log_scale: dict[str, bool] | None = None,
        lr_binary_search_steps: int = 7,
        test_every_forwards: int | None = None,
        batched: dict[str, bool] | None = None,
        smoothing: dict[str, int | None] | None = None,
    ):
        clean_mem()
        kw = {}
        if not has_lr: kw['log10_lrs'] = None

        # run binary search
        with performance_context(name, 2) if print_time else nullcontext():

            bench._print_timeout = True # print if ran out of time
            bench.search(
                task_name=name,
                opt_name=opt_name,
                target_metrics=target_metrics,
                optimizer_fn=opt_fn,
                max_passes=max_passes,
                max_seconds=max_seconds,
                smoothing = smoothing,
                batched = batched,
                test_every_forwards=test_every_forwards,
                lr_binary_search_steps=lr_binary_search_steps,
                **kw,
            )

        # plotting
        for metric in reversed(target_metrics):
            log_scale_value = False
            if (log_scale is not None) and (metric in log_scale) and log_scale[metric]: log_scale_value = True

            (plot_metric(task_name=name, metric=metric, opts = opt_name, log_scale=log_scale_value, fig = fig.add(f'{name} {metric}'), show = False, opts_all_lrs=False, ref=ref)
            .legend(size=12))
            (plot_lr_search_curve(task_name=name, metric=metric, opts = opt_name, log_scale=log_scale_value, fig = fig.add(f'{name} {metric} lrs'), show = False, ref = ref)
            .legend(size=12))


    # note that some benchmarks are a bit faster on CPU
    # however optimizer logic like newton schulz in muon is way slower so in the end CUDA is faster

    # ----------------------------------- PATHS ---------------------------------- #
    bench = FunctionDescent('booth')
    bench = _search_for_visualization(bench, opt_fn, max_passes=200, max_seconds=5)
    bench.plot(fig=fig.add(f'Booth 200 passes lr={bench._info["lr"]}'), show=False) # type:ignore

    bench = FunctionDescent('rosen')
    bench = _search_for_visualization(bench, opt_fn, max_passes=200, max_seconds=5)
    bench.plot(fig=fig.add(f'Rosenbrock 200 passes lr={bench._info["lr"]}'), show=False) # type:ignore

    bench = FunctionDescent('goldstein_price')
    bench = _search_for_visualization(bench, opt_fn, max_passes=200, max_seconds=5)
    bench.plot(fig=fig.add(f'Goldstein-Price 200 passes lr={bench._info["lr"]}'), show=False) # type:ignore

    bench = FunctionDescent('spiral_short', x0 = (0.15, 0.07))
    bench = _search_for_visualization(bench, opt_fn, max_passes=200, max_seconds=5)
    bench.plot(fig=fig.add(f'Spiral 200 passes lr={bench._info["lr"]}'), show=False) # type:ignore

    # --------------------------- SYNTHETIC OBJECTIVES --------------------------- #
    # ---------------------------- Convex512  --------------------------- #
    # basic booth-like convex objective
    bench = Convex().cuda()
    _search(bench, 'Convex', _train_loss, max_passes=2_000, max_seconds=30, log_scale={'train loss': True})

    # ---------------------------- Inverse L1 randn-64 --------------------------- #
    # L1 bench for a difference, favours  MARS
    bench = Inverse(randn, F.l1_loss).cuda()
    _search(bench, 'Inverse L1 randn-64', _train_loss, max_passes=2_000, max_seconds=30, log_scale={'train loss': True})

    # ------------------------ InverseInverse randn-64 ------------------------ #
    # diabolical kron and muon and soap lead
    bench = InverseInverse(randn).cuda()
    _search(bench, 'InverseInverse randn-64', _train_loss, max_passes=2_000, max_seconds=30, log_scale={'train loss': True},)

    # -------------------------- MatrixSign attngrad-96 -------------------------- #
    # L-BFGS insane lead but then SOAP, rprop, kron, muon so might be unique, good values range is quite thin
    bench = MatrixSign(ATTNGRAD96).cuda()
    _search(bench, 'MatrixSign attngrad-96', _train_loss, max_passes=2_000, max_seconds=30, log_scale={'train loss': True},)

    # ------------------------------- ML OBJECTIVES ------------------------------ #
    # we search train losses, and then test losses which usually only makes very few additional evaluations

    # --------------------- Mnist1d MLP([40,40,40,40]) --------------------- #
    # soap-kron-muon (this is better/harder than california housing)
    bench = Mnist1d(models.MLP([40,40,40,40])).cuda()
    _search(
        bench = bench,
        name = "Mnist1d MLP([40,40,40,40])",
        target_metrics = _test_train_loss,
        max_passes=2_000,
        max_seconds=60,
        log_scale={'train loss': True},
    )

    # ---------------------------- BATCHED OBJECTIVES ---------------------------- #
    # --------------------- Mnist1d MLP([64,96,128,256]) bs64 --------------------- #
    # soap-kron-muon again, now batched
    bench = Mnist1d(models.MLP([64,96,128,256]), batch_size = 64).cuda()
    _search(
        bench = bench,
        name = "Mnist1d MLP([64,96,128,256]) bs64",
        target_metrics = _test_train_loss,
        max_passes=4_000,
        max_seconds=120,
        test_every_forwards=10,
        smoothing={'train loss': 8},
        batched={'train loss': True},
        log_scale={'train loss': True},
    )

    # ------------------------- Mnist1d RNN(40, 2) bs128 ------------------------- #
    # huge kron and soap lead + muon and mars next
    bench = Mnist1d(models.Mnist1dRNN(40, 2, torch.nn.RNN), batch_size=128,).cuda()
    _search(
        bench = bench,
        name = "Mnist1d RNN(40, 2) bs128",
        target_metrics = _test_train_loss,
        max_passes=4_000,
        max_seconds=120,
        test_every_forwards=20,
        smoothing={'train loss': 8},
        batched={'train loss': True},
    )

    # ------------------- Mnist1d ConvNet([64,96,128,256]) bs32 ------------------ #
    # huge kron and soap lead + muon and mars next
    bench = Mnist1d(models.Mnist1dConvNet([64,96,128,256]), batch_size=32, test_batch_size=512).cuda()
    _search(
        bench = bench,
        name = 'Mnist1d ConvNet([64,96,128,256])) bs32',
        target_metrics = _test_train_loss,
        max_passes=6_000,
        max_seconds=360,
        test_every_forwards=20,
        smoothing={'train loss': 8},
        batched={'train loss': True},
        log_scale={'train loss': True},
    )

    # ------------------- Mnist1dConvNetAutoencoder([32, 64, 128, 256]) bs128 ------------------ #
    #
    bench = Mnist1dAutoencoder(models.Mnist1dConvNetAutoencoder([64,96,128,256]), batch_size=32, test_batch_size=512).cuda()
    _search(
        bench = bench,
        name = 'Mnist1dConvNetAutoencoder([32,64,128,256]) bs32',
        target_metrics = _test_train_loss,
        max_passes=2_000,
        max_seconds=120,
        test_every_forwards=20,
        smoothing={'train loss': 8},
        batched={'train loss': True},
        log_scale={'train loss': True, 'test loss': True},
    )


    # -------------- SynthSeg1d ConvNetAutoencoder([64,96,128]) bs64 ------------- #
    bench = SynthSeg1d(
        model = models.Mnist1dConvNetAutoencoder([64,96,128], clip_length = 32, real_out=5),
        batch_size=64,
        test_batch_size=512,
    ).cuda()
    _search(
        bench = bench,
        name = 'SynthSeg1d ConvNetAutoencoder([64,96,128]) bs64',
        target_metrics = _test_train_loss,
        max_passes=4_000,
        max_seconds=240,
        test_every_forwards=20,
        smoothing={'train loss': 8},
        batched={'train loss': True},
        log_scale={'train loss': False, 'test loss': False},
    )


    # --------------------------------- PLOTTING --------------------------------- #
    # for fn in queue: fn()
    if show: fig.show(axsize = (12, 6), dpi=100, ncols = 2)
    if save:
        print()
        print(f'finished, saving `summary - {opt_name}.jpg`...')
        fig.savefig(f'summary - {opt_name}.jpg', axsize = (12, 6), dpi=300, ncols = 2)
    fig.close()