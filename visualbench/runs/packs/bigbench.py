"""tiny benchmark pretty useless"""
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import monai.losses
import torch
from kornia.losses import ssim_loss
from sklearn.datasets import make_swiss_roll
from torch import nn
from torch.nn import functional as F

import visualbench as vb
from visualbench.utils import CUDA_IF_AVAILABLE
from ..benchpack import OptimizerBenchPack

LOSSES = ("train loss", "test loss")

def _unbatched_ssim(x,y):
    return ssim_loss(x[None,:], y[None,:],5)

class BigBench(OptimizerBenchPack):
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
        step_callbacks: "Callable[[vb.Benchmark], Any] | Sequence[Callable[[vb.Benchmark], Any]] | None" = None,

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


    def run_2d(self):
        bench = vb.FunctionDescent('booth')
        self.run_bench(bench, '2D - booth', passes=1_000, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = vb.FunctionDescent('ill2')
        self.run_bench(bench, '2D - ill2', passes=1_000, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = vb.FunctionDescent('ill4')
        self.run_bench(bench, '2D - ill4', passes=1_000, sec=10, metrics='train loss', vid_scale=1, fps=10)

        bench = vb.FunctionDescent('rosen10')
        self.run_bench(bench, '2D - rosenbrock-10', passes=1_000, sec=10, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent('rosen')
        self.run_bench(bench, '2D - rosenbrock', passes=1_000, sec=10, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent('dipole_field')
        self.run_bench(bench, '2D - dipole field', passes=1_000, sec=10, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent('around')
        self.run_bench(bench, '2D - around', passes=1_000, sec=10, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent(vb.test_functions.booth + vb.test_functions.noise * 5)
        self.run_bench(bench, '2D - booth + noise', passes=1_000, sec=20, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent('rosenabs')
        self.run_bench(bench, '2D - rosenbrock abs', passes=2_000, sec=20, metrics='train loss', vid_scale=1)

        bench = vb.FunctionDescent('spiral')
        self.run_bench(bench, '2D - spiral', passes=2_000, sec=20, metrics='train loss', vid_scale=1)



    def run_visual(self):
        # ------------------------------- NeuralDrawer ------------------------------- #
        # ndim = 1,843
        # AdamW - 103s, SOAP - 178s
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ReLU+bn', passes=5_000, sec=60, metrics='train loss', vid_scale=2)

        # ndim = 1,843
        # AdamW - 95s, SOAP - 164s
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), batch_size=32, expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer BS-16 - ReLU+bn', passes=5_000, sec=60, metrics='train loss', vid_scale=2)

        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ELU), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - ELU', passes=5_000, sec=60, metrics='train loss', vid_scale=2)

        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,12,12,12,12,12,12,12,3], act_cls=vb.models.act.Sine), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Sine', passes=5_000, sec=60, metrics='train loss', vid_scale=2)

        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,1000,3], act_cls=nn.ReLU, ortho_init=True), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - Wide ReLU', passes=5_000, sec=60, metrics='train loss', vid_scale=2)

        # ------------------------------- Colorization ------------------------------- #
        bench = vb.Colorization.small().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        bench = vb.Colorization.small(order=2).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (2nd order)', passes=10_000, sec=120, metrics='train loss', vid_scale=4)

        bench = vb.Colorization.small(power=1).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (L1)', passes=10_000, sec=120, metrics='train loss', vid_scale=4)

        # ------------------------------- Graph layout ------------------------------- #
        bench = vb.GraphLayout(vb.GraphLayout.GRID()).to(CUDA_IF_AVAILABLE)
        bench_name = 'Visual - Graph layout optimization'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.

        # ----------------------------------- t-SNE ---------------------------------- #
        X, y = make_swiss_roll(1000, noise=0.1, hole=True, random_state=0)
        bench = vb.TSNE(X, y).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - t-SNE', passes=2_000, sec=90, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.

        # ----------------------- Sine Approximator - Tanh 7-4 ---------------------- #
        bench = vb.FunctionApproximator(
            vb.FunctionApproximator.SINE(8), n_skip=4, depth=7, resolution=(384,768),
        ) # NO CUDA
        bench_name = 'Visual - Sine Approximator - Tanh 7-4'
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # ----------------------- Particle minmax ---------------------- #
        bench = vb.ClosestFurthestParticles(32, spread=0.75) # NO CUDA
        self.run_bench(bench, 'Visual - Particle min-max', passes=2_000, sec=60, metrics='train loss', vid_scale=1)

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        bench = vb.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Alpha Evolve B1', passes=4_000, sec=90, metrics='train loss', vid_scale=1)

        # ------------------------------ Style transfer ------------------------------ #
        bench = vb.StyleTransfer(vb.data.FROG96, vb.data.GEOM96).to(CUDA_IF_AVAILABLE)
        bench_name = "Visual - Style Transfer"
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', vid_scale=2)

        # ---------------------------------- Drawers --------------------------------- #
        # CirclesDrawer - looks interesting
        # ndim = 1,404
        # AdamW - 60s, SOAP - 74s
        bench = vb.CirclesDrawer(vb.data.WEEVIL96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - CirclesDrawer', passes=5_000, sec=60, metrics='train loss', vid_scale=2, test_every=10)

        # PartitionDrawer - interesting enough and super fast
        # NOTE: weight decay is very bad for this one
        # ndim = 501
        # AdamW - 41s , SOAP - 53s
        bench = vb.PartitionDrawer(vb.data.WEEVIL96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - PartitionDrawer', passes=5_000, sec=60, metrics='train loss', vid_scale=4)

        # CurvesDrawer - too slow
        # GaborFiltersDrawer - slow
        # LinesDrawer - too easy
        # LinesDrawer(per_line_thickness) - too easy
        # RectanglesDrawer - easy and boring
        # TrianglesDrawer - boring
        # VoronoiDrawer - too hard / arbitrary


        # ------------------------------ Polynomial fit ------------------------------ #
        bench = vb.FitData(*vb.tasks.FitData.DATA(), vb.tasks.FitData.POLY(8)) # NO CUDA!
        self.run_bench(bench, 'Visual - Polynomial fit', passes=2_000, sec=60, metrics='train loss', vid_scale=1)



    def run_synthetic(self):
        # ------------------------------ Rosenbrock-384 ------------------------------ #
        bench = vb.projected.Rosenbrock(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rosenbrock 384', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        # ---------------------------- IllConditioned-256 ---------------------------- #
        bench = vb.RotatedQuadratic(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rotated quadratic 384', passes=2_000, sec=30, metrics='train loss', vid_scale=None)

        # ------------------------------- Rastrigin-384 ------------------------------ #
        bench = vb.projected.Rastrigin(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rastrigin 384', passes=2_000, sec=30, metrics='train loss', vid_scale=None)

        # ---------------------------------- Linalg ---------------------------------- #
        bench = vb.Inverse(vb.data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - Inverse MSE', passes=2_000, sec=60, metrics='train loss', vid_scale=2)

        bench = vb.StochasticInverse(vb.data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticInverse', passes=2_000, sec=60, metrics='test loss', vid_scale=2)

        bench = vb.LeastSquares(vb.data.FROG96, vb.data.SANIC96, ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - LeastSquares', passes=2_000, sec=60, metrics='train loss', vid_scale=2)



    def run_real(self):
        # ---------------------------- Human heart dipole ---------------------------- #
        # ndim = 8
        # 3.3s. ~ 1m. 6s.
        bench = vb.HumanHeartDipole() # NO CUDA
        self.run_bench(bench, "Real - Human heart dipole", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

        # ---------------------------- Propane combustion ---------------------------- #
        # ndim = 11
        # 3.3s. ~ 1m. 6s.
        bench = vb.PropaneCombustion() # NO CUDA
        self.run_bench(bench, "Real - Propane combustion", passes=2_000, sec=60, metrics='train loss', vid_scale=None)

        # -------------------------------- Muon coeffs ------------------------------- #
        # ndim = 15
        # 9.1s. ~ 3m. 3s.
        bench = vb.MuonCoeffs(resolution=(512, 512)) # NO CUDA
        self.run_bench(bench, 'Real - Muon coefficients', passes=2_000, sec=120, metrics='train loss', vid_scale=1, binary_mul=0.75)

        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = 600
        # 4.4s. ~ 1m. 30s.
        bench = vb.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Real - Alpha Evolve B1', passes=4_000, sec=90, metrics='train loss', vid_scale=1)

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = 49,152
        # 14s. ~ 4m. 40s.
        # 9+4=13 ~ 3m.
        bench = vb.StyleTransfer(vb.data.FROG96, vb.data.GEOM96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Real - Style Transfer', passes=2_000, sec=120, metrics='train loss', binary_mul=0.4, vid_scale=2)


    def run_ml(self):
        # --------------------- TinyConvNet (full-batch MNIST-1D) -------------------- #
        # strong overfitting, may be good to study generalization
        # ndim = 4,098
        # 4.6s. ~ 1m. 32s.
        bench = vb.Mnist1d(vb.models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - MNIST-1D FB - TinyConvNet", passes=2_000, sec=120, metrics = LOSSES, vid_scale=None)

        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # 22s. ~ 7m. 20s.
        # 9+3=12 ~ 4m. 20s.
        bench = vb.WavePINN(vb.WavePINN.FLS(2, 1, hidden_size=256, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - FLS', passes=2_000, sec=240, metrics='train loss', binary_mul=0.3, vid_scale=4)

    def run_mls(self):
        # stochastic
        # ---------------------------- Logistic regression --------------------------- #
        # ndim = 385
        # 5s. ~ 1m. 40s.
        bench = vb.Covertype(vb.models.MLP([54, 7]), batch_size=1).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Covertype BS-1 - Logistic Regression'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # --------------------------- Matrix factorization --------------------------- #
        bench = vb.MFMovieLens("/var/mnt/hdd/datasets/MovieLens 100K", batch_size=32, device='cuda').cuda()
        bench_name = 'MLS - MovieLens BS-32 - Matrix Factorization'
        self.run_bench(bench, bench_name, passes=2_000, sec=60, test_every=10, metrics='test loss', vid_scale=None)

        # ------------------------------- MLP (MNIST-1D) ------------------------------ #
        # ndim = 56,874
        # 9.4s ~ 2m. 28s.
        bench = vb.Mnist1d(
            vb.models.MLP([40, 64, 96, 128, 256, 10], act_cls=nn.ELU),
            batch_size=64
        ).to(CUDA_IF_AVAILABLE)
        bench_name = "MLS - MNIST-1D BS-64 - MLP(40-64-96-128-256-10)"
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=20, metrics = "test loss", vid_scale=None, binary_mul=0.75)

        # ------------------------------- RNN (MNIST-1D) ------------------------------ #
        # ndim = 20,410
        # 11s. ~ 3m. 30s.
        bench = vb.Mnist1d(
            vb.models.RNN(1, 10, hidden_size=40, num_layers=2, rnn=torch.nn.RNN),
            batch_size=128,
        ).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MNIST-1D BS-128 - RNN(2x40)'
        self.run_bench(bench, bench_name, passes=4_000, sec=120, test_every=20, metrics='test loss', vid_scale=None, binary_mul=0.5)

        # --------------------- TinyConvNet (MNIST-1D) -------------------- #
        # ndim = 4,098
        # 3.9s. ~ 1m. 18s.
        bench = vb.Mnist1d(vb.models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU), batch_size=32).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLS - MNIST-1D BS-32 - TinyConvNet", passes=2_000, sec=60, test_every=10, metrics = "test loss", vid_scale=None)

        # ----------------------- Sparse Autoencoder (MNIST-1D) ---------------------- #
        # 8.0s ~ 2m. 30s.
        bench = vb.Mnist1dAutoencoding(
            vb.models.vision.ConvNetAutoencoder(1, 1, 1, 40, encoder=(64,96,128,256), sparse_reg=0.1),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet', passes=2_000, sec=120, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.75)

        # ---------------------------- ConvNet (SynthSeg) ---------------------------- #
        # 18.8s ~ 6m. 12s.
        # 9+3=12 ~ 3m. 44s.
        bench = vb.SynthSeg1d(
            vb.models.vision.ConvNetAutoencoder(1, 1, 5, 32, encoder=(64,96,128)),
            criterion = monai.losses.DiceFocalLoss(softmax=True),
            num_samples=10_000, batch_size=64, test_batch_size=512
        ).cuda()
        self.run_bench(bench, 'MLS - SynthSeg BS-64 - ConvNet', passes=4_000, sec=240, test_every=50, metrics='test loss', vid_scale=None, binary_mul=0.3)



    # ----------------------------------- extra ---------------------------------- #
    def run_visual_extra(self):
        # ----------------------------------- moons ---------------------------------- #
        bench = vb.Moons(vb.models.MLP([2,2,2,2,2,2,2,2,1]),).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ELU', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = vb.Moons(vb.models.MLP([2,2,2,2,2,2,2,2,1], act_cls=nn.ReLU, bn=True)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Moons FB - MLP(2-2-2-2-2-2-2-2-1)-ReLU+bn', passes=2_000, sec=90, metrics="train loss", vid_scale=2)

        bench = vb.Moons(vb.models.MLP([2,2,2,2,2,2,2,2,1]), batch_size=16, n_samples=2048, train_split=1024).to(CUDA_IF_AVAILABLE)
        bench_name= "Visual - Moons BS-16 - MLP(2-2-2-2-2-2-2-2-1)-ELU"
        self.run_bench(bench, bench_name, passes=2_000, sec=90, metrics='test loss', vid_scale=2, test_every=1)

        # ------------------------------- lines drawer ------------------------------- #
        bench = vb.LinesDrawer(vb.data.WEEVIL96, 100, loss=_unbatched_ssim).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - LinesDrawer SSIM', passes=2000, sec=60, metrics='train loss', vid_scale=4, fps=30)

        # ------------------------- Colorization (1.3th power) ------------------------- #
        bench = vb.Colorization.small(power=1.3).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (1.3th power)', passes=2_000, sec=60, metrics='train loss', vid_scale=8)

        # ----------------------- Sine Approximator - LeakyReLU 10-4 ---------------------- #
        bench = vb.FunctionApproximator(
            vb.FunctionApproximator.SINE(8), n_skip=4, depth=10, act=F.leaky_relu, resolution=(384,768),
        ) # NO CUDA
        self.run_bench(bench, 'Visual - Sine Approximator - LeakyReLU 10-4', passes=2_000, sec=120, metrics='train loss', vid_scale=1)

        # -------------------------- deformable registration ------------------------- #
        bench = vb.DeformableRegistration(vb.data.FROG96, grid_size=(5,5)).cuda()
        self.run_bench(bench, 'Visual - DeformableRegistration', passes=2_000, sec=60, metrics='train loss', vid_scale=2)
