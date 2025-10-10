"""tiny benchmark pretty useless"""
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import monai.losses
import torch
from kornia.losses import ssim_loss
from sklearn.datasets import make_swiss_roll
from torch import nn
from torch.nn import functional as F

from .. import data, models, tasks
from ..utils import CUDA_IF_AVAILABLE
from .benchpack import OptimizerBenchPack

if TYPE_CHECKING:
    from ..benchmark import Benchmark

LOSSES = ("train loss", "test loss")

def _unbatched_ssim(x,y):
    return ssim_loss(x[None,:], y[None,:],5)

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

        init_fn = lambda opt_fn, bench, value: opt_fn([p for p in bench.parameters() if p.requires_grad], value)
    ):
        kwargs = locals().copy()
        del kwargs["self"], kwargs["__class__"]
        super().__init__(**kwargs)

    def run(self, ML=True, synthetic=True, visual=True, twod=True, stochastic=True, *, extra_visual=False):
        if twod:
            self.run_2d()

        if visual:
            self.run_visual()

        if synthetic:
            self.run_synthetic()

        if ML:
            self.run_real()
            self.run_ml()
            if stochastic: self.run_mls()

        if extra_visual:
            self.run_visual_extra()

    def run_synthetic(self):
        # basic
        # ------------------------------ Rosenbrock-256 ------------------------------ #
        bench = tasks.projected.Rosenbrock(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Rosenbrock 384', passes=2000, sec=30, metrics='train loss', vid_scale=4)

        # ---------------------------- IllConditioned-256 ---------------------------- #
        bench = tasks.RotatedQuadratic(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Rotated quadratic 384', passes=2000, sec=30, metrics='train loss', vid_scale=None)

        # -------------------- Nonsmooth Chebyshev-Rosenbrock 384 -------------------- #
        bench = tasks.ChebushevRosenbrock(dim=384, p=100, pd_fn=torch.abs).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Nonsmooth Chebyshev-Rosenbrock 384', passes=2000, sec=30, metrics='train loss', vid_scale=None)

        # --------------------------------- rastrigin -------------------------------- #
        bench = tasks.Rastrigin(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Rastrigin 384', passes=2000, sec=30, metrics='train loss', vid_scale=None)


        # good linalg
        # ------------------------------- Inverse-16 L1 ------------------------------ #
        # SOAP, PSGD, NAG, Muon, Adam, AdamW, BFGS-Backtracking. SOAP/PSGD are 0.08, Adam 0.10, BFGS is 0.12.
        bench = tasks.Inverse(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse-16 L1', passes=2000, sec=30, metrics='train loss', vid_scale=None)

        # --------------------------- MatrixLogarithm-16 L1 -------------------------- #
        # smooth, PSGD 0.02, SOAP 0.03, AdamW 0.05
        bench = tasks.MatrixLogarithm(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - MatrixLogarithm-16 L1', passes=2000, sec=30, metrics='train loss', vid_scale=None)


        # maybe linalg
        # ------------------------------ Inverse-16 MSE ------------------------------ #
        # AdaptiveHeavyBall, Newton and QN with up to 1e-13  is good, Adam 5e-4, SOAP 5e-5. Maybe keep for convex testing
        bench = tasks.Inverse(data.get_fielder(16)[0], criterion=F.mse_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Inverse-fielder16 MSE', passes=2000, sec=30, metrics='train loss', vid_scale=16)

        # ---------------------------- MoorePenrose-16 L1 ---------------------------- #
        # weird mix, but reasonably big spacing between algos, so maybe as a weirder kind of problem with clean lr to loss curve, best is Adam
        bench = tasks.MoorePenrose(16, criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - MoorePenrose-16 L1', passes=2000, sec=30, metrics='train loss', vid_scale=None)

        # ---------------------------- Drazin-fielder16 L1 --------------------------- #
        # hard, only few managed to reach 2 - LBFGS and ShorR. Then we have BFGS with 1348, Adam has 2233
        bench = tasks.Drazin(data.get_fielder(16)[0], criterion=F.l1_loss).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'S - Drazin-fielder16 L1', passes=2000, sec=30, metrics='train loss', vid_scale=16)

        # -------------------------- StochasticRLstsq-10 MSE ------------------------- #
        # smooth, big gaps, Adagrad is best, not sure if this is a good proxy for generalization
        bench = tasks.StochasticRLstsq(10, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'SS - StochasticRLstsq-10 MSE', passes=2000, sec=30, metrics='test loss', vid_scale=None)


    def run_visual(self):
        # ------------------------------- neural drawer ------------------------------ #
        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ReLU+bn', passes=2000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ELU), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ELU', passes=2000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,12,12,12,12,12,12,12,3], act_cls=models.act.Sine), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Sine', passes=2000, sec=60, metrics='train loss', vid_scale=2)

        bench = tasks.NeuralDrawer(data.WEEVIL96, models.MLP([2,1000,3], act_cls=nn.ReLU, ortho_init=True), expand=48).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Wide ReLU', passes=2000, sec=60, metrics='train loss', vid_scale=2)

        # ------------------------------- Colorization ------------------------------- #
        # ndim  = 1024
        # 3.2s. ~ 1m. 4s.
        bench = tasks.Colorization.small().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=2_000, sec=60, metrics='train loss', vid_scale=4)

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
        # ndim = 29
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

        # ----------------------------- partition drawer ----------------------------- #
        bench = tasks.PartitionDrawer(data.WEEVIL96, 100).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - PartitionDrawer', passes=2000, sec=60, metrics='train loss', vid_scale=4)


    def run_real(self):
        # ---------------------------- Human heart dipole ---------------------------- #
        # ndim = 8
        # 3.3s. ~ 1m. 6s.
        bench = tasks.HumanHeartDipole() # NO CUDA
        self.run_bench(bench, "Real - Human heart dipole", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

        # ---------------------------- Propane combustion ---------------------------- #
        # ndim = 11
        # 3.3s. ~ 1m. 6s.
        bench = tasks.PropaneCombustion() # NO CUDA
        self.run_bench(bench, "Real - Propane combustion", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

        # -------------------------------- Muon coeffs ------------------------------- #
        # ndim = 15
        # 9.1s. ~ 3m. 3s.
        bench = tasks.MuonCoeffs(resolution=(512, 512)) # NO CUDA
        self.run_bench(bench, 'Real - Muon coefficients', passes=2_000, sec=120, metrics='train loss', vid_scale=1, binary_mul=0.75)

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = 600
        # 4.4s. ~ 1m. 30s.
        bench = tasks.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Real - Alpha Evolve B1', passes=4_000, sec=90, metrics='train loss', vid_scale=1)

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = 49,152
        # 14s. ~ 4m. 40s.
        # 9+4=13 ~ 3m.
        bench = tasks.StyleTransfer(data.FROG96, data.GEOM96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Real - Style Transfer', passes=2_000, sec=120, metrics='train loss', binary_mul=0.4, vid_scale=2)


    def run_ml(self):
        # --------------------- TinyConvNet (full-batch MNIST-1D) -------------------- #
        # strong overfitting, may be good to study generalization
        # ndim = 4,098
        # 4.6s. ~ 1m. 32s.
        bench = tasks.datasets.Mnist1d(models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - TinyConvNet", passes=2_000, sec=120, metrics = LOSSES, vid_scale=None)

        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = tasks.WavePINN(tasks.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - FLS', passes=2_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)

    def run_mls(self):
        # stochastic
        # ---------------------------- Logistic regression --------------------------- #
        # ndim = 385
        # 5s. ~ 1m. 40s.
        bench = tasks.datasets.Covertype(models.MLP([54, 7]), batch_size=1).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Covertype BS-1 - Logistic Regression'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # --------------------------- Matrix factorization --------------------------- #
        bench = tasks.MFMovieLens("/var/mnt/hdd/datasets/MovieLens 100K", batch_size=32, device='cuda').cuda()
        bench_name = 'MLS - MovieLens BS-32 - Matrix Factorization'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # ------------------------------- MLP (MNIST-1D) ------------------------------ #
        # ndim = 56,874
        # 9.4s ~ 2m. 28s.
        bench = tasks.datasets.Mnist1d(
            models.MLP([40, 64, 96, 128, 256, 10], act_cls=nn.ELU),
            batch_size=64
        ).to(CUDA_IF_AVAILABLE)
        bench_name = "MLS - MNIST-1D BS-64 - MLP(40-64-96-128-256-10)"
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=20, metrics = "test loss", vid_scale=None, binary_mul=0.75)

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = tasks.datasets.Mnist1d(
            models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128,
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MNIST-1D BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=20, metrics='test loss', vid_scale=None, binary_mul=0.5)

        # --------------------- TinyConvNet (MNIST-1D) -------------------- #
        # ndim = 4,098
        # 3.9s. ~ 1m. 18s.
        bench = tasks.datasets.Mnist1d(models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU), batch_size=32).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLS - MNIST-1D BS-32 - TinyConvNet", passes=2_000, sec=60, test_every=10, metrics = "test loss", vid_scale=None)

        # ----------------------- Sparse Autoencoder (MNIST-1D) ---------------------- #
        # 8.0s ~ 2m. 30s.
        bench = tasks.datasets.Mnist1dAutoencoding(
            models.vision.ConvNetAutoencoder(1, 1, 1, 40, hidden=(64,96,128,256), sparse_reg=0.1),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet', passes=2_000, sec=120, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.75)

        # ---------------------------- ConvNet (SynthSeg) ---------------------------- #
        # 18.8s ~ 6m. 12s.
        # 9+3=12 ~ 3m. 44s.
        bench = tasks.datasets.SynthSeg1d(
            models.vision.ConvNetAutoencoder(1, 1, 5, 32, hidden=(64,96,128)),
            criterion = monai.losses.DiceFocalLoss(softmax=True),
            num_samples=10_000, batch_size=64, test_batch_size=512
        ).cuda()
        self.run_bench(bench, 'MLS - SynthSeg BS-64 - ConvNet', passes=4_000, sec=240, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.3)


    def run_2d(self):
        bench = tasks.FunctionDescent('booth')
        self.run_bench(bench, '2D - booth', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('ill')
        self.run_bench(bench, '2D - ill', passes=200, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = tasks.FunctionDescent('rosen10')
        self.run_bench(bench, '2D - rosenbrock-10', passes=1000, sec=30, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1000, sec=30, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2000, sec=60, metrics='train loss', vid_scale=1)

        bench = tasks.FunctionDescent('spiral')
        self.run_bench(bench, '2D - spiral', passes=2000, sec=60, metrics='train loss', vid_scale=1)



    # ----------------------------------- extra ---------------------------------- #
    def run_visual_extra(self):
        # ----------------------------------- moons ---------------------------------- #
        bench = tasks.Moons(models.MLP([2,2,2,2,2,2,2,2,1]),).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP([2,2,2,2,2,2,2,2,1], act_cls=nn.ReLU, bn=True)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ReLU+bn', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = tasks.Moons(models.MLP([2,2,2,2,2,2,2,2,1]), batch_size=16, n_samples=2048, train_split=1024).to(CUDA_IF_AVAILABLE)
        bench_name= "Visual - Moons BS-16 - MLP(2-2-2-2-2-2-2-2-1)-ELU"
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics='test loss', vid_scale=2, test_every=1)

        # ------------------------------- lines drawer ------------------------------- #
        bench = tasks.LinesDrawer(data.WEEVIL96, 100, loss=_unbatched_ssim).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - LinesDrawer SSIM', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ------------------------- Colorization (1.3th power) ------------------------- #
        bench = tasks.Colorization.small(power=1.3).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (1.3th power)', passes=2_000, sec=60, metrics='train loss', vid_scale=8)

        # ----------------------- Sine Approximator - LeakyReLU 10-4 ---------------------- #
        bench = tasks.FunctionApproximator(
            tasks.FunctionApproximator.SINE(8), n_skip=4, depth=10, act=F.leaky_relu, resolution=(384,768),
        ) # NO CUDA
        self.run_bench(bench, 'Visual - Sine Approximator - LeakyReLU 10-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # -------------------------- deformable registration ------------------------- #
        bench = tasks.DeformableRegistration(data.FROG96, grid_size=(5,5)).cuda()
        self.run_bench(bench, 'Visual - DeformableRegistration', passes=2_000, sec=60, metrics='train loss', vid_scale=2)
