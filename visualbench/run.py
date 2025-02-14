import copy
import math
import os
from collections import OrderedDict

import numpy as np
import torch
from glio.jupyter_tools import clean_mem
from myai.loaders.text import txtread, txtwrite
from myai.loaders.yaml import yamlread, yamlwrite
from myai.plt_tools import Fig
from scipy.ndimage import gaussian_filter1d

from .benchmark import Benchmark
from .tasks import (
    GCN,
    XOR,
    CharLSTM,
    Convex,
    DiffusionMNIST1D,
    MatrixInverse,
    MNIST1D_ConvNet_Fullbatch,
    MNIST1D_ConvNet_Minibatch,
    MNIST1D_LSTM_Minibatch,
    MNIST1D_MLP_Minibatch,
    MNIST1D_ResNet_Fullbatch,
    StyleTransfer,
    TorchGeometricDataset,
    Rosenbrock,
)

REFERENCE_OPTS = ("SGD", "AdamW")

def _ensure_float(x):
    if isinstance(x, torch.Tensor): return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

def _round_significant(x: float, nsignificant: int):
    if x > 1: return round(x, nsignificant)
    if x == 0: return x

    v = 1
    for n in range(100):
        v /= 10
        if abs(x) > v: return round(x, nsignificant + n)

    return x # x is nan
    #raise RuntimeError(f"wtf {x = }")


class Task:
    """task
    Args:
        name (_type_): name
        new_fn (_type_): _description_
        max_passes (_type_): _description_
        max_steps (_type_): _description_
        max_search_time (_type_): _description_
        min_loss (_type_, optional): _description_. Defaults to None.
        test_every_sec (_type_, optional): _description_. Defaults to None.
    """
    def __init__(self, name, new_fn, max_passes, max_time, max_search_time, min_loss = 0., min_search_iters = 2, max_search_iters = 5, test_every_sec = None, test_every_steps=None, root='runs',):

        self.name = name
        self.new_fn = new_fn
        self.max_time = max_time
        self.max_search_time = max_search_time
        self.max_passes = max_passes
        self.min_loss = min_loss
        self.test_every_sec = test_every_sec
        self.test_every_steps = test_every_steps
        self.max_search_iters = max_search_iters
        self.min_search_iters = min_search_iters
        self.root = root

        self.task_dir = os.path.join(root, name)
        if not os.path.exists(self.task_dir): os.mkdir(self.task_dir)

        self.info_file = os.path.join(self.task_dir, 'info.yaml')
        self._needs_yaml_update = False
        if os.path.exists(self.info_file):
            self.info = yamlread(self.info_file)
        else:
            self._needs_yaml_update = True
            self.info = {
                0.1:
                    {
                        'lowest loss': float('inf'),
                        'best run': '',
                    },
                0.25:
                    {
                        'lowest loss': float('inf'),
                        'best run': '',
                    },
                0.5:
                    {
                        'lowest loss': float('inf'),
                        'best run': '',
                    },
                0.75:
                    {
                        'lowest loss': float('inf'),
                        'best run': '',
                    },
                1:
                    {
                        'lowest loss': float('inf'),
                        'best run': '',
                    },
                }


    def _check_res(self, res: "Result"):
        for rel_point, info in self.info.items():
            if rel_point == 1:
                frac_losses = res.any_loss[1]
            else:
                abs_point = self.max_passes * rel_point

                # index of loss at this point (needed because loss is not logged every pass due to backward passes)
                idx = np.argmax(res.any_loss[0] >= abs_point)
                if idx < 1: continue # skip fractions that are too low

                # losses up to this point
                frac_losses = res.any_loss[1, :idx]

            # compare it
            min = frac_losses.min()
            if min < info['lowest loss']:
                info['lowest loss'] = _ensure_float(min)

                print(f'{self.name} - {self.opt_name} reached new {rel_point*100}% lowest loss {_round_significant(min, 3)} with lr {_round_significant(res.lr, 3)}!')

                info['best run'] = os.path.join(self.task_dir, self.opt_name, f'{res.lr}.npz')
                self._needs_yaml_update = True

        # info1 = self.info[1]
        # if res.score < info1['best score']:
        #     print(f'{self.name} - {self.opt_name} reached new best score {_round_significant(res.score, 3)} with lr {_round_significant(res.lr, 3)}!')
        #     info1['best score'] = res.score
        #     info1['best run'] = os.path.join(self.task_dir, self.opt_name, f'{res.lr}.npz')
        #     self._needs_yaml_update = True


    def search(self, opt_name: str, opt_fn, lrs_log10 = (1, 0, -1, -2, -3, -4, -5)):
        self.opt_name = opt_name
        path = os.path.join(self.root, self.name, opt_name)
        if os.path.exists(path):
            print(f"WARNING {path} ALREDAY EXISTS")
            raise FileExistsError(path)

        time_passed = 0

        res: list[Result] = []
        for lr10 in lrs_log10 if lrs_log10 is not None else [1]:
            lr = 10 ** lr10
            bench: Benchmark = self.new_fn()

            bench._log_projections = False # for performance
            bench._log_params = False
            bench._save_edge_params = False

            opt = opt_fn(bench.parameters(), lr)
            bench.run(opt, max_passes = self.max_passes, max_time = self.max_time, min_loss = None, test_every_sec=self.test_every_sec, test_every_steps=self.test_every_steps)
            time_passed += bench._time_passed

            res.append(Result(bench, self, opt_name, lr))
            self._check_res(res[-1])


        for search_iter in range(self.max_search_iters if lrs_log10 is not None else 0):

            # sort by score
            res.sort(key = lambda r: r.score)


            # best lrs
            ind = 0
            new_lr10 = None
            lr1_10 = res[ind].lr10; lr2_10 = res[ind+1].lr10
            while True:

                # sort by lrs
                res_sorted = sorted(res, key = lambda r: r.lr)

                if lr1_10 == res_sorted[0].lr10: new_lr10 = res_sorted[0].lr10 - 1 # smallest lr is best, try smaller one
                elif lr1_10 == res_sorted[-1].lr10: new_lr10 = res_sorted[-1].lr10 + 1 # largest lr is best, try larger one
                else:
                    # unless lr it outside or 1st iter, break
                    if (time_passed > self.max_search_time) and (search_iter >= self.min_search_iters): break
                    new_lr10 = lr1_10 + (lr2_10 - lr1_10) / 2

                # check if lr already evaluated
                evaluated_lr10s = [_round_significant(r.lr10, 3) for r in res]
                if _round_significant(new_lr10, 3) in evaluated_lr10s:
                    ind+=1
                    if ind >= len(res): break
                    lr1_10 = res[ind].lr10; lr2_10 = res[ind+1].lr10
                else:
                    break

            if new_lr10 is None: break

            lr = 10 ** new_lr10
            bench: Benchmark = self.new_fn()
            opt = opt_fn(bench.parameters(), lr)
            bench.run(opt, max_passes = self.max_passes, max_time = self.max_time, min_loss = None, test_every_sec=self.test_every_sec, test_every_steps=self.test_every_steps)
            time_passed += bench._time_passed

            res.append(Result(bench, self, opt_name, lr))
            self._check_res(res[-1])

        if self._needs_yaml_update:
            yamlwrite(self.info, self.info_file)

        return res

def _nan_to_num(v):
    if isinstance(v, (int,float)):
        if not np.isfinite(v): return float('inf')
        return v

    if not isinstance(v, np.ndarray): v = np.asarray(v)
    return np.nan_to_num(v, nan = float('inf'), posinf = float('inf'), neginf = float('inf'))

class Result:
    def __init__(self, bench: Benchmark, task: Task, opt_name: str, lr: float):
        # store info
        self.task_name = task.name
        self.opt_name = opt_name
        self.lr = lr
        self.lr10 = math.log10(lr)
        self.max_passes = task.max_passes

        # store losses
        self.train_loss = np.array(list(bench.logger['train loss'].items())).T
        """(2, n_passes) i.e. (keys, values)"""
        assert self.train_loss.shape[0] == 2
        self.lowest_loss = _ensure_float(bench._lowest_loss) # only needed for plotting

        if 'test loss' in bench.logger:
            self.test_loss = np.array(list(bench.logger['test loss'].items())).T
            """(2, n_passes) i.e. (keys, values)"""
        else: self.test_loss = None

        self.any_loss = self.test_loss if self.test_loss is not None else self.train_loss
        """(2, n_passes) i.e. (keys, values)"""
        assert self.any_loss.shape[0] == 2

        # calculate number of steps before converging to min_loss
        if self.lowest_loss > task.min_loss: self.converged_steps = None
        else: self.converged_steps = self.any_loss[0, np.argmax(self.any_loss[1] <= task.min_loss)]

        # calculate score, lower is always better
        if self.converged_steps is None: self.score = self.lowest_loss
        else:
            # this is guaranteed to be less then min_loss, and early first good idx is smaller
            #self.score = float(task.min_loss - (1 - (self.converged_steps / (task.max_passes/2))))
            self.score = self.lowest_loss


    def save(self, root):
        """i think this is faster than pickle"""
        if not os.path.exists(root): os.mkdir(root)

        task_path = os.path.join(root, self.task_name)
        if not os.path.exists(task_path): os.mkdir(task_path)

        opt_path = os.path.join(task_path, self.opt_name)
        if not os.path.exists(opt_path): os.mkdir(opt_path)

        data = {
            k:_nan_to_num(v) for k,v in self.__dict__.items() if not (k.startswith('_') or isinstance(v, (str,dict)) or v is None)
        }
        out_path = os.path.join(opt_path, f'{self.lr}')
        if os.path.exists(out_path): print(f'overwriting {out_path}')
        np.savez(out_path, **data) # type:ignore

    @classmethod
    def load(cls, file):
        """this loads without any methods but they are not needed anyway"""
        data = np.load(file)
        instance = cls.__new__(cls)

        for k,v in data.items():
            if v.ndim == 0: setattr(instance, k, float(v))
            else: setattr(instance, k, v)

        for attr in ["test_loss", "converged_steps"]:
            if not hasattr(instance, attr): setattr(instance, attr, None)

        instance.opt_name = os.path.basename(os.path.dirname(file))
        instance.task_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        return instance


def _get_score(x: Result): return x.score
def load_results(path, metric_fn = _get_score):
    """loads rsults sorted by score, 1st result in list is the best"""
    return sorted([Result.load(os.path.join(path, f)) for f in os.listdir(path)], key = metric_fn)


def _plot_loss_(opt_dir, fig:Fig, ymin,ymax, plotted:set, prefix='', metric = 'any_loss', kw=None, postprocess=None):
    """helper function to plot loss from opt_dir

    Args:
        opt_dir: path to dir or file
        fig: figure to plot on
        ymax: current ymax, this returns new one if it is none
        plotted: set of already plotted ones
        prefix: for name
    """
    if kw is None: kw = {}

    if os.path.isdir(opt_dir):

        def _metric_fn(x:Result):
            values: np.ndarray = getattr(x, metric)
            return np.nanmin(values[1])

        results = load_results(opt_dir, _metric_fn)
        best = results[0]

    else:
        if not os.path.isfile(opt_dir): raise FileNotFoundError(opt_dir)
        best = Result.load(opt_dir)

    values: np.ndarray = getattr(best, metric)
    if ymax is None: ymax = values[1,0]
    if ymin is None: ymin = values[1].min()
    else: ymin = min(ymin, values[1].min())

    if (best.opt_name, best.lr) not in plotted:
        if postprocess is not None: values = postprocess(values)

        # limit ymax
        values = values.copy()
        values[1] = np.where(values[1] < ymax*1.1, values[1], np.nan)

        fig.linechart(
            *values,
            label=f'{prefix}{best.opt_name} {_round_significant(best.lr, 3)} - {_round_significant(float(np.nanmin(values[1])), 3)}',
            **kw,
            )
        plotted.add((best.opt_name, best.lr))

    return ymin,ymax

def plot_loss(task, opts = None, root = 'runs', log_scale=False, metric = 'train_loss', fig = None, show=True, postprocess=None):
    """
    plot loss curves of all reference opts, best opts, and given opts.

    Args:
        task: name of task
        opts: string or list of strings with extra optimizers to plot
        root: root dir
        log_scale: whether to use log scale
    """
    task_dir = os.path.join(root, task)

    if fig is None: fig = Fig()

    task_info = yamlread(os.path.join(task_dir, 'info.yaml'))
    ymax = None
    ymin = None
    #ymin = task_info[1]['lowest loss']

    plotted = set()


    # plot opts (1st for consistent coloring)
    if opts is None: opts = []
    if isinstance(opts, str): opts = [opts]
    for opt in opts:
        opt_dir = os.path.join(task_dir, opt)
        ymin,ymax = _plot_loss_(opt_dir, fig, ymin,ymax, plotted, metric=metric, postprocess=postprocess)

    # plot reference opts
    for opt_name in REFERENCE_OPTS:
        opt_dir = os.path.join(task_dir, opt_name)
        if not os.path.exists(opt_dir): continue
        ymin,ymax = _plot_loss_(opt_dir, fig, ymin,ymax, plotted, metric=metric, kw=dict(linewidth=0.5), postprocess=postprocess)

    # plot best opts
    for point, info in reversed(list(task_info.items())):
        best_run = info['best run']
        if len(best_run) == 0: continue
        opt_name = os.path.basename(os.path.dirname(best_run))
        ymin,ymax = _plot_loss_(best_run, fig, ymin,ymax, plotted, prefix=f"(best {point*100}%) ", metric=metric, kw=dict(linewidth=0.5), postprocess=postprocess)


    # expand ymin/ymax by 10%
    assert ymin is not None
    assert ymax is not None

    d = (ymax - ymin)*0.05
    if log_scale:
        # fig.ylim(10**(ymin), 10**(ymax))
        fig.yscale('symlog', linthresh = 1e-8)
    else:
        fig.ylim(ymin-d, ymax+d)

    fig.preset(
        xlabel = 'forward/backward passes',
        ylabel = metric.replace('_', ' '),
        title = task,
        ticks=False,
    ).legend(size=12)
    if show: fig.show()


def _gaussian_smooth(x, sigma):
    if sigma is None: return x
    return gaussian_filter1d(x, sigma, mode='nearest', truncate=4.0)

def run(name:str, opt_fn, has_lr=True, root='runs', save=True, show=True, savefig=True, blacklist = (), whitelist=None):
    if not os.path.exists(root): os.mkdir(root)
    fig = Fig()

    def _search(task:Task, log_scale=False, train_sigma=None, test_sigma=None):
        if task.name in blacklist: return
        if whitelist is not None and task.name not in whitelist: return
        clean_mem()
        if has_lr: res = task.search(name, opt_fn)
        else: res = task.search(name, opt_fn, None)
        if save:
            for i in res: i.save(root)

        def _train_post(x): return _gaussian_smooth(x, train_sigma)
        def _test_post(x): return _gaussian_smooth(x, test_sigma)

        plot_loss(
            res[0].task_name,
            res[0].opt_name,
            metric="train_loss",
            log_scale=log_scale,
            fig=fig.add(task.name),
            show=False,
            postprocess=_train_post,
        )
        if res[0].test_loss is not None:
            plot_loss(
                res[0].task_name,
                res[0].opt_name,
                metric="test_loss",
                log_scale=log_scale,
                fig=fig.add(task.name),
                show=False,
                postprocess=_test_post,
            )
        return res



    # ---------------------------------- CONVEX ---------------------------------- #
    task = Task(
        "Convex 512",
        Convex,
        max_passes=2000,
        max_time=10,
        max_search_time=30,
        min_loss=-1,
    )
    _search(task, True)

    # ----------------------------------- ROSEN ---------------------------------- #
    task = Task(
        "Rosenbrock 4096",
        Rosenbrock,
        max_passes=2000,
        max_time=10,
        max_search_time=30,
        min_loss=-1,
    )
    _search(task, True)

    # ---------------------------------- INVERSE --------------------------------- #
    task = Task(
        "Matrix Inverse",
        lambda: MatrixInverse(torch.randn(64,64), make_images=False),
        max_passes=2000,
        max_time=10,
        max_search_time=30,
        min_loss=-1,
    )
    _search(task, True)

    # ------------------------------------ XOR ----------------------------------- #
    task = Task(
        "XOR",
        XOR,
        max_passes=2000,
        max_time=20,
        max_search_time=40,
        min_loss=-1,
    )
    _search(task, True)

    # ------------------------------- STYLE TRANFER ------------------------------ #
    bench = StyleTransfer(
        "/var/mnt/ssd/Файлы/Изображения/Сохраненное/тест.jpg",
        "/var/mnt/ssd/Файлы/Изображения/Сохраненное/sanic.jpg", save_images=False).cuda()

    def style_transfer_fn():
        bench.reset() #type:ignore
        return bench

    task = Task(
        "Style Transfer",
        style_transfer_fn,
        max_passes=2000,
        max_time=40,
        max_search_time=120,
        min_search_iters=1,
        max_search_iters=2,
        min_loss=-1,
    )
    _search(task, True)

    # --------------------------------- CHAR LSTM -------------------------------- #
    task = Task(
        "CharLSTM",
        lambda: CharLSTM(64).cuda(),
        max_passes=2000,
        max_time=30,
        max_search_time=60,
        min_loss=-1,
    )
    _search(task, False, train_sigma=3)

    # ------------------------------- MNIST-1D MLP ------------------------------- #
    bench = MNIST1D_MLP_Minibatch().cuda()
    model = copy.deepcopy(bench.model)

    def mnist1d_mlp_fn():
        bench.reset(copy.deepcopy(model)) # type:ignore # pylint:disable=too-many-function-args
        bench.model.train() # type:ignore
        return bench

    task = Task(
        "MNIST-1D MLP",
        mnist1d_mlp_fn,
        max_passes=2000,
        max_time=30,
        max_search_time=60,
        min_loss=-1,
        test_every_steps = 10,
    )

    _search(task, False, train_sigma=2)


    # # ------------------------------- MNIST-1D LSTM ------------------------------ #
    # bench = MNIST1D_LSTM_Minibatch(32, 3)
    # model = copy.deepcopy(bench.model)

    # def mnist1d_lstm_fn():
    #     bench.reset(copy.deepcopy(model))
    #     bench.model.lstm.flatten_parameters() # type:ignore
    #     bench.model.train()
    #     return bench

    # task = Task(
    #     "MNIST-1D LSTM",
    #     mnist1d_lstm_fn,
    #     max_passes=2000,
    #     max_time=60,
    #     max_search_time=360,
    #     max_search_iters=3,
    #     min_loss=-1,
    #     test_every_steps = 10,
    # )
    # _search(task, False)


    # ------------------------------ MNIST1D-ConvNet ------------------------------ #
    bench = MNIST1D_ConvNet_Minibatch().cuda()
    model = copy.deepcopy(bench.model)

    def mnist1d_convnet_fn():
        bench.reset(copy.deepcopy(model)) # type:ignore # pylint:disable=too-many-function-args
        bench.model.train() # type:ignore
        return bench

    task = Task(
        "MNIST-1D ConvNet",
        mnist1d_convnet_fn,
        max_passes=2000,
        max_time=30,
        max_search_time=60,
        min_loss=-1,
        test_every_steps = 10,
    )
    _search(task, False, train_sigma=3)

    # # ------------------------- MNIST1D-ConvNet-Fullbatch ------------------------ #
    # bench = MNIST1D_ConvNet_Fullbatch([1, 16, 32, 64, 128], fixed_bn=True).cuda()
    # model = copy.deepcopy(bench.model)

    # def mnist1d_convnet_fullbatch_fn():
    #     bench.reset(copy.deepcopy(model)) # type:ignore # pylint:disable=too-many-function-args
    #     bench.model.train() # type:ignore
    #     return bench

    # task = Task(
    #     "MNIST-1D ConvNet fullbatch",
    #     mnist1d_convnet_fullbatch_fn,
    #     max_passes=2000,
    #     max_time=40,
    #     max_search_time=80,
    #     min_loss=-1,
    #     test_every_steps = 1,
    # )
    # _search(task, False)

    # --------------------------------- DIFFUSION -------------------------------- #
    # bench = DiffusionMNIST1D().cuda()
    # model = copy.deepcopy(bench.model)
    # def diffusion_fn():
    #     bench.reset(copy.deepcopy(model)) # type:ignore # pylint:disable=too-many-function-args
    #     bench.model.train() # type:ignore
    #     return bench

    # task = Task(
    #     'MNIST-1D Diffusion U-Net',
    #     diffusion_fn,
    #     max_passes=2000,
    #     max_time=40,
    #     max_search_time=80,
    #     min_loss=-1,
    # )
    # _search(task, False, train_sigma=6)

    # --------------------------------- Graph NN --------------------------------- #
    bench = TorchGeometricDataset(lambda i,o: GCN(i, o, (8, 8), act=torch.nn.functional.relu, dropout=0.5)).cuda()
    def graph_fn():
        bench.reset() # type:ignore # pylint:disable=too-many-function-args
        return bench

    task = Task(
        'GCN Cora',
        graph_fn,
        max_passes=2000,
        max_time=50,
        max_search_time=100,
        min_loss=-1,
    )
    _search(task, False, train_sigma=1, test_sigma=3)


    # ----------------------------------- SAVE ----------------------------------- #
    fig.figtitle(f'{name}', size = 14)
    if show: fig.show(ncols = 2, axsize = (12, 6))
    if savefig: fig.savefig(f'{name}.png', ncols = 2, axsize = (12, 6), dpi=300)