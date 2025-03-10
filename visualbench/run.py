from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch
from glio.jupyter_tools import clean_mem
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
    Eigen,
    FunctionDescent,
    Inverse,
    InverseInverse,
    LUPivot,
    MatrixSign,
    Mnist1d,
    RNNArgsort,
    SelfRecurrent,
    Sorting,
    models,
)
from .tasks.linalg._linalg_utils import _full01

_train_loss = {"train loss": False}
_test_train_loss = {"test loss": False, "train loss": False}
_train_test_loss = {"train loss": False, "test loss": False}

def run_bench(opt_name:str, opt_fn: Callable, show=True, save=True, extra:Sequence[str]|str|None =(), has_lr=True, skip_batched=False, print_time = False):
    torch.manual_seed(0)
    if extra is None: extra = []
    elif isinstance(extra, str): extra = [extra]
    else: extra = list(extra)
    ref = list(REFERENCE_OPTS) + extra

    qrcode = get_qrcode()
    randn = get_randn()
    fig = Fig()

    queue = []
    def _search(
        bench: Benchmark,
        name: str,
        target_metrics: dict[str, bool],
        max_passes: int,
        max_seconds: float,
        log_scale: bool,
        lr_binary_search_steps: int = 7,
        test_every_forwards: int | None = None,
        batched: dict[str, bool] | None = None,
        smoothing: dict[str, int | None] | None = None,
        # existing_files_count_towards_steps=True,
        # max_files = 17, # 7 base lr10s + 5 and 5 binary search lrs
        # print_time = print_time,
    ):
        clean_mem()
        kw = {}
        if not has_lr: kw['lrs10'] = None
        with performance_context(name, 2) if print_time else nullcontext():
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
                # existing_files_count_towards_steps=existing_files_count_towards_steps,
                # max_files = max_files,
                **kw,
            )

        for metric in reversed(target_metrics):
            def _queue_plot():
                (plot_metric(task_name=name, metric=metric, opts = opt_name, log_scale=log_scale, fig = fig.add(f'{name} {metric}'), show = False, opts_all_lrs=False, ref=ref)
                .legend(size=12))
                (plot_lr_search_curve(task_name=name, metric=metric, opts = opt_name, log_scale=log_scale, fig = fig.add(f'{name} {metric} lrs'), show = False, ref = ref)
                .legend(size=12))
            queue.append(_queue_plot)

    # note that some benchmarks are a bit faster on CPU
    # however optimizer logic like newton schulz in muon is way slower so in the end CUDA is faster

    # ----------------------------------- paths ---------------------------------- #
    bench = FunctionDescent('booth')
    bench = _search_for_visualization(bench, opt_fn, max_passes=200)
    bench.plot(fig=fig.add('Booth 200 passes'), show=False) # type:ignore

    bench = FunctionDescent('rosen')
    bench = _search_for_visualization(bench, opt_fn, max_passes=1000)
    bench.plot(fig=fig.add('Rosenbrock 1000 passes'), show=False) # type:ignore

    # ------------------------------- QEP qrcode-96 ------------------------------ #
    # for testing if optimizer is good at exploiting curvature
    bench = QEP(qrcode, qrcode.flip(-1), qrcode.flip(-2)).cuda()
    _search(bench, 'QEP qrcode-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True)

    # ------------------------- SelfRecurrent attngrad-96 ------------------------ #
    # every good optimizer is 2x better here
    bench = SelfRecurrent(ATTNGRAD96, n = 5, init = _full01).cuda()
    _search(bench, 'SelfRecurrent attngrad-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True)

    # ----------------------------- LUPivot qrcode-96 ---------------------------- #
    # crazy kron and soap lead
    bench = LUPivot(qrcode, sinkhorn_iters=4).cuda()
    _search(bench, 'LUPivot qrcode-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True)

    # ------------------------------- Eigen test-96 ------------------------------ #
    # crazy kron and muon lead
    bench = Eigen(TEST96).cuda()
    _search(bench, 'Eigen test-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True,)

    # -------------------------------- QR test-96 -------------------------------- #
    # Favors rprop for a difference
    bench = QR(TEST96).cuda()
    _search(bench, 'QR test-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True)

    # ---------------------------- Inverse L1 randn-64 --------------------------- #
    # L1 bench for a difference, favours  MARS
    bench = Inverse(randn, F.l1_loss).cuda()
    _search(bench, 'Inverse L1 randn-64', _train_loss, max_passes=2000, max_seconds=30, log_scale=True)

    # ------------------------ InverseInverse L1 randn-64 ------------------------ #
    # diabolical kron and muon and soap lead
    bench = InverseInverse(randn, loss = F.l1_loss).cuda()
    _search(bench, 'InverseInverse L1 randn-64', _train_loss, max_passes=2000, max_seconds=30, log_scale=True,)

    # -------------------------- MatrixSign attngrad-96 -------------------------- #
    # L-BFGS insane lead but then SOAP, rprop, kron, muon so might be unique
    bench = MatrixSign(ATTNGRAD96)
    _search(bench, 'MatrixSign attngrad-96', _train_loss, max_passes=2000, max_seconds=30, log_scale=True,)

    # ---------------------------------- Sorting --------------------------------- #
    # MARS leads, then kron, then Adam
    bench = Sorting(sinkhorn_iters=4)
    _search(bench, 'Sorting', _train_loss, max_passes=2000, max_seconds=30, log_scale=True,)

    # --------------------- CaliforniaHousing MLP16-16-16-16 --------------------- #
    # soap-kron-muon, not big advantage tho
    # we search train losses, and then test losses which usually only makes very few additional evaluations
    bench = CaliforniaHousing(models.MLP([16,16,16,16])).cuda()
    _search(
        bench = bench,
        name = "CaliforniaHousing MLP([16,16,16,16])",
        target_metrics = _test_train_loss,
        max_passes=2000,
        max_seconds=60,
        log_scale=True,
    )

    # ------------------------ Mnist1d RecurrentMLP(40, 7) ----------------------- #
    # kron leads in train, SOAP in test, mayb be good for testing generalization
    bench = Mnist1d(models.RecurrentMLP(40, 7)).cuda()
    _search(
        bench = bench,
        name = "Mnist1d RecurrentMLP(40, 7)",
        target_metrics = _test_train_loss,
        max_passes=2000,
        max_seconds=60,
        log_scale=False,
    )

    if not skip_batched:
        # ------------------------- Mnist1d RNN(40, 2) bs128 ------------------------- #
        # huge kron and soap lead + muon and mars next
        bench = Mnist1d(models.Mnist1dRNN(40, 2, torch.nn.RNN), batch_size=128,).cuda()
        _search(
            bench = bench,
            name = "Mnist1d RNN(40, 2) bs128",
            target_metrics = _test_train_loss,
            max_passes=2000,
            max_seconds=60,
            test_every_forwards=10,
            smoothing={'train loss': 3},
            batched={'train loss': True},
            log_scale=False,
        )

        # ------------------- Mnist1d ConvNet([32,64,128,256]) bs32 ------------------ #
        # huge kron and soap lead + muon and mars next
        bench = Mnist1d(models.Mnist1dConvNet([32,64,128,256]), batch_size=32, test_batch_size=512).cuda()
        _search(
            bench = bench,
            name = 'Mnist1d ConvNet([32,64,128,256])) bs32',
            target_metrics = _test_train_loss,
            max_passes=2000,
            max_seconds=60,
            test_every_forwards=10,
            smoothing={'train loss': 3},
            batched={'train loss': True},
            log_scale=False,
        )

    # --------------------------------- PLOTTING --------------------------------- #
    for fn in queue: fn()
    if show: fig.show(axsize = (12, 6), dpi=300, ncols = 2)
    if save:
        print(f'finished, saving `summary - {opt_name}.jpg`...')
        fig.savefig(f'summary - {opt_name}.pdf', axsize = (12, 6), dpi=300, ncols = 2)
    fig.close()