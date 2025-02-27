from collections.abc import Callable, Sequence

import torch
from glio.jupyter_tools import clean_mem
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig

from ._utils.runs_plotting import plot_lr_search_curve, plot_metric, _print_best
from .benchmark import Benchmark
from .data import ATTNGRAD96, QRCODE96, SANIC96, TEST96, get_qrcode, get_randn
from .tasks import QEP, Eigen, LSTMArgsort, LUPivot, SelfRecurrent, InverseInverse, CaliforniaHousing, Mnist1d
from .tasks import models
from .tasks.linalg._linalg_utils import _full01
from ._utils.runs import REFERENCE_OPTS


_trainloss = {"train loss": False}
_train_test_loss = {"train loss": False, "test loss": False}

def run_bench(opt_name:str, opt_fn: Callable, show=True, save=True, extra:Sequence[str]|str|None =(), has_lr=True):
    if extra is None: extra = []
    elif isinstance(extra, str): extra = [extra]
    else: extra = list(extra)
    ref = list(REFERENCE_OPTS) + extra

    qrcode = get_qrcode()
    fig = Fig()

    def _search(
        bench: Benchmark,
        name: str,
        target_metrics: dict[str, bool],
        max_passes: int,
        max_seconds: float,
        log_scale: bool,
        lr_binary_search_steps: int = 7,
        test_every_forwards: int | None = None,
    ):
        clean_mem()
        kw = {}
        if not has_lr: kw['lrs10'] = None
        bench.search(
            task_name=name,
            opt_name=opt_name,
            target_metrics=target_metrics,
            optimizer_fn=opt_fn,
            max_passes=max_passes,
            max_seconds=max_seconds,
            test_every_forwards=test_every_forwards, lr_binary_search_steps=lr_binary_search_steps,
            **kw,
        )
        (plot_metric(task_name=name, opts = opt_name, log_scale=log_scale, fig = fig.add(f'{name} losses'), show = False, opts_all_lrs=False, ref=ref)
         .legend(size=12))
        (plot_lr_search_curve(task_name=name, opts = opt_name, log_scale=log_scale, fig = fig.add(f'{name} lrs'), show = False, ref = ref)
         .legend(size=12))


    # ------------------------------------ QEP ----------------------------------- #
    # for testing if optimizer is good at exploiting curvature
    bench = QEP(qrcode, qrcode.flip(-1), qrcode.flip(-2)).cuda()
    _search(bench, 'QEP qrcode-96', _trainloss, max_passes=2000, max_seconds=30, log_scale=True)

    # ------------------------------ MatrixLogarithm ----------------------------- #
    # every good optimizer is 2x better here
    bench = SelfRecurrent(ATTNGRAD96, n = 5, init = _full01).cuda()
    _search(bench, 'SelfRecurrent attngrad-96', _trainloss, max_passes=2000, max_seconds=30, log_scale=True)

    # ---------------------------------- LUPivot --------------------------------- #
    # crazy kron and soap lead
    bench = LUPivot(qrcode).cuda()
    _search(bench, 'LUPivot qrcode-96', _trainloss, max_passes=2000, max_seconds=30, log_scale=True)

    # ----------------------------------- Eigen ---------------------------------- #
    # crazy kron and muon lead
    bench = Eigen(TEST96).cuda()
    _search(bench, 'Eigen test-96', _trainloss, max_passes=2000, max_seconds=30, log_scale=True)

    # ------------------------------ INVERSEINVERSE ------------------------------ #
    # crazy kron and muon and soap lead
    bench = InverseInverse(get_randn()).cuda()
    _search(bench, 'InverseInverse randn-64', _trainloss, max_passes=2000, max_seconds=30, log_scale=True)

    # -------------------------------- LSTMArgsort ------------------------------- #
    # all good optimizers are at the top + its mini-batch
    bench = LSTMArgsort().cuda()
    _search(bench, 'LSTMArgsort', _trainloss, max_passes=2000, max_seconds=30, log_scale=False)

    # --------------------- CaliforniaHousing MLP16-16-16-16 --------------------- #
    # soap-kron-muon, not big advantage tho
    bench = CaliforniaHousing(models.MLP([16,16,16,16]))
    _search(bench, 'CaliforniaHousing MLP16-16-16-16', _train_test_loss, max_passes=2000, max_seconds=60, log_scale=True)

    # ------------------------- Mnist1d RecurrentMLP40-7 ------------------------- #
    # kron leads in train, SOAP in test, mayb be good for testing generalization
    bench = Mnist1d(models.RecurrentMLP(40, 7))
    _search(bench, 'Mnist1d RecurrentMLP40-7', _train_test_loss, max_passes=2000, max_seconds=60, log_scale=False)

    # --------------------- Mnist1d ConvNet32-64-128-256 bs32 -------------------- #
    # kron soap train, muon test
    bench = Mnist1d(models.Mnist1dConvNet(), batch_size=32)
    _search(bench, 'Mnist1d ConvNet32-64-128-256 bs32', _train_test_loss, max_passes=2000, max_seconds=60, test_every_forwards=10, log_scale=False)

    # -------------------------- Mnist1d LSTM40-2 bs128 -------------------------- #
    # huge kron and soap lead + muon and mars next
    bench = Mnist1d(models.Mnist1dConvNet(), batch_size=128)
    _search(bench, 'Mnist1d LSTM40-2 bs128', _train_test_loss, max_passes=2000, max_seconds=60, test_every_forwards=10, log_scale=False)

    # --------------------------------- PLOTTING --------------------------------- #
    if show: fig.show(axsize = (12, 6), dpi=300, ncols = 2)
    if save: fig.savefig(f'summary: {opt_name}.jpg', axsize = (12, 6), dpi=300, ncols = 2)
    fig.close()