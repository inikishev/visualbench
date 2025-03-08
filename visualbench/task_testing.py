from typing import Literal
import time

from collections.abc import Callable, Sequence

import torch
from heavyball import CachedPSGDKron, PrecondSchedulePaLMForeachSOAP
from myai.loaders.image import imreadtensor
from myai.plt_tools import Fig
from myai.python_tools import performance_context
from myai.torch_tools import count_params
from mystuff.found.torch.optim.Muon import Muon
from pytorch_optimizer import MARS

from ._utils.runs_plotting import _print_best, plot_lr_search_curve, plot_metric
from .benchmark import Benchmark
from .data import ATTNGRAD96, SANIC96, TEST96, get_qrcode, get_randn

OPTS = {
    "Adam": torch.optim.Adam,
    "Rprop": torch.optim.Rprop,
    "Adagrad": torch.optim.Adagrad,
    "NAG": lambda p,lr: torch.optim.SGD(p, lr, momentum = 0.9, nesterov=True),
    "SGD": torch.optim.SGD,
    "PrecondSchedulePaLMSOAP": PrecondSchedulePaLMForeachSOAP,
    "Kron": lambda p,lr: CachedPSGDKron(p, lr, store_triu_as_line=False),
    "Muon": Muon,
    "MARS-AdamW": MARS,
    "L-BFGS": lambda p, lr: torch.optim.LBFGS(p,lr,line_search_fn='strong_wolfe'),
}

_randn = get_randn()
_qrcode = get_qrcode()


MATRICES = {
    "randn-64": _randn,
    "attngrad-96": ATTNGRAD96,
    "sanic-96": SANIC96,
    "test-96": TEST96,
    "qrcode-96": _qrcode,
}



def test_benchmark(
    name,
    bench_fn: Callable[..., Benchmark],
    target_metrics: dict[str, bool],
    pass_mats=False,
    max_passes=2000,
    max_seconds=30,
    test_every_batches = None,
    lr_binary_search_steps=7,
    skip_opt: str | Sequence[str] | None = (),
    log_scale = True,
    progress: Literal['full', 'reduced', 'none'] = 'reduced',
    print_achievements = True,
    debug=False,
):
    start_time = time.perf_counter()

    if skip_opt is None: skip_opt = ()
    if isinstance(skip_opt, str): skip_opt = (skip_opt, )

    # test all matrices
    mats = MATRICES if pass_mats else {None:None}
    for m_name, m in mats.items():
        if pass_mats: b = bench_fn(m)
        else: b = bench_fn()

        # check that it runs on CUDA
        bench_name = f'{name} {m_name}' if m_name is not None else name
        print(f'\ntesting "{bench_name}" with {count_params(b)} params running on {b.device}')

        # test all opts
        for opt_name, cls in OPTS.items():
            if opt_name in skip_opt: continue

            kw:dict = dict(log10_lrs = (0, )) if opt_name == 'L-BFGS' else {}
            # try:
            b.search(
                task_name = bench_name,
                opt_name = opt_name,
                target_metrics = target_metrics,
                optimizer_fn = cls,
                max_passes = max_passes,
                max_seconds = max_seconds,
                test_every_batches = test_every_batches,
                lr_binary_search_steps = lr_binary_search_steps,
                root = 'bench tests',
                debug = debug,
                progress = progress,
                print_achievements = print_achievements,
                **kw
            )
            # except Exception as e:
            #     print(f'BENCHMARK FAILED {e!r}')

        sec = time.perf_counter() - start_time
        # print/plot results
        print()
        print('RESULTS:')
        _print_best(bench_name, root = 'bench tests')

        fig = Fig()
        for m in target_metrics:
            plot_metric(bench_name, None, log_scale=log_scale, metric=m, show=False, ref='all', fig = fig.add(m), root = 'bench tests')
            plot_lr_search_curve(bench_name, None, log_scale=log_scale, metric=m, show=False, ref='all', fig = fig.add(m), root = 'bench tests')

        fig.figtitle(bench_name).savefig(f'{bench_name} - {sec:.2f}s..jpg', axsize = (12, 6), dpi=300, ncols=2).close()