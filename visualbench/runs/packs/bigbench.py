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
        num_binary: int = 5,
        num_expansions: int = 12,
        rounding = 1,
        fixed_hyperparams: dict | None = None,
        max_dim: int | None = None,
        tune: bool = True,
        skip:str | Sequence[str] | None = None,

        # storage
        root: str = "BigBench",
        print_records: bool = True,
        print_progress: bool = True,
        print_time: bool = False,
        save: bool = True,
        accelerate: bool = True,
        accelerate_kwargs: dict[str, Any] | None = None,
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

    def run(self, ML=True, synthetic=True, visual=True, twod=True, stochastic=True):
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
        # AdamW - ?, SOAP - ?
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ReLU, bn=True), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - MLP(2-16-16-16-16-16-16-16-3 ReLU+bn)', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = 1,731
        # Adam - ?, SOAP - ?
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ELU), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - MLP(2-16-16-16-16-16-16-16-3 ELU)', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = 1,731
        # AdamW - ?, SOAP - ?
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,16,16,16,16,16,16,16,3], act_cls=nn.ELU), batch_size=32, expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer BS-16 - MLP(2-16-16-16-16-16-16-16-3 ELU)', passes=20_000, sec=120, test_every=10, metrics='test loss', vid_scale=2)

        # ndim = 1,011
        # Adam - ?, SOAP - ?
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,12,12,12,12,12,12,12,3], act_cls=vb.models.act.Sine), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - MLP(2-12-12-12-12-12-12-12-3 Sine)', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = 6,147
        # Adam - ?, SOAP - ?
        bench = vb.NeuralDrawer(vb.data.WEEVIL96, vb.models.MLP([2,1024,3], act_cls=nn.ReLU, ortho_init=True), expand=24).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - NeuralDrawer - MLP(2-1024-3 ReLU)', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ------------------------------- Colorization ------------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.Colorization.small().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.Colorization.tiny(order=2).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Colorization (2nd order)', passes=20_000, sec=120, metrics='train loss', vid_scale=10)


        # ------------------------------- Graph layout ------------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.GraphLayout(vb.GraphLayout.GRID()).to(CUDA_IF_AVAILABLE)
        bench_name = 'Visual - Graph layout optimization'
        self.run_bench(bench, bench_name, passes=2_000, sec=120, metrics='train loss', vid_scale=1) # 4.4s. ~ 1m. 30s.


        # ----------------------- Particle minmax ---------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.ClosestFurthestParticles(32, spread=0.75) # NO CUDA
        self.run_bench(bench, 'Visual - Particle min-max', passes=2_000, sec=120, metrics='train loss', vid_scale=1)


        # ------------------------------ Alpha Evolve B1 ----------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.AlphaEvolveB1().to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - Alpha Evolve B1', passes=20_000, sec=120, metrics='train loss', vid_scale=1)

        # ---------------------------------- Drawers --------------------------------- #
        # CirclesDrawer - looks interesting
        # ndim = 1,404
        # Adam - ?, SOAP - ?
        bench = vb.CirclesDrawer(vb.data.WEEVIL96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - CirclesDrawer', passes=20_000, sec=120, metrics='train loss', vid_scale=2, test_every=10)

        # PartitionDrawer - interesting enough and super fast
        # NOTE: weight decay is very bad for this one
        # ndim = 501
        # Adam - ?, SOAP - ?
        bench = vb.PartitionDrawer(vb.data.WEEVIL96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Visual - PartitionDrawer', passes=20_000, sec=120, metrics='train loss', vid_scale=4)

        # CurvesDrawer - too slow
        # GaborFiltersDrawer - slow
        # LinesDrawer - too easy
        # LinesDrawer(per_line_thickness) - too easy
        # RectanglesDrawer - easy and boring
        # TrianglesDrawer - boring
        # VoronoiDrawer - too hard / arbitrary


        # ------------------------------ Polynomial fit ------------------------------ #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.FitData(*vb.tasks.FitData.DATA(), vb.tasks.FitData.POLY(8)) # NO CUDA!
        self.run_bench(bench, 'Visual - Polynomial fit', passes=20_000, sec=120, metrics='train loss', vid_scale=1)


        # ----------------------------------- Kato ----------------------------------- #
        bench = vb.Kato(vb.data.WEEVIL96).to(CUDA_IF_AVAILABLE)
        # ndim = 27,648
        # Adam - ?, SOAP - >
        self.run_bench(bench, 'Visual - Kato', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ------------------------------------ NSC ----------------------------------- #
        bench = vb.NormalScalarCurvature().to(CUDA_IF_AVAILABLE)
        # ndim = 16,384
        # Adam - ?, SOAP - ?
        self.run_bench(bench, 'Visual - NormalScalarCurvature', passes=20_000, sec=120, metrics='train loss', vid_scale=2)


    def run_synthetic(self):
        # ------------------------------ Rosenbrock-384 ------------------------------ #
        # Adam - ?, SOAP - ?
        bench = vb.projected.Rosenbrock(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rosenbrock 384', passes=5_000, sec=120, metrics='train loss', vid_scale=4)

        # ---------------------------- IllConditioned-256 ---------------------------- #
        # Adam - ?, SOAP - ?
        bench = vb.RotatedQuadratic(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rotated quadratic 384', passes=2_000, sec=120, metrics='train loss', vid_scale=None)

        # ------------------------------- Rastrigin-384 ------------------------------ #
        # Adam - ?, SOAP - ?
        bench = vb.projected.Rastrigin(384).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Synthetic - Rastrigin 384', passes=2_000, sec=120, metrics='train loss', vid_scale=None)

        # ---------------------------------- Linalg ---------------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.Inverse(vb.data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - Inverse', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.StochasticInverse(vb.data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticInverse', passes=20_000, sec=120, metrics='test loss', vid_scale=2)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.LeastSquares(vb.data.FROG96, vb.data.SANIC96).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - LeastSquares', passes=2_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.MatrixIdempotent(A=vb.data.SANIC96, n=10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - MatrixIdempotent-10', passes=20_000, sec=120, metrics='train loss', vid_scale=2)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.StochasticMatrixIdempotent(A=vb.data.SANIC96, n=10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - StochasticMatrixIdempotent-10', passes=20_000, sec=120, metrics='test loss', vid_scale=2)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.TensorSpectralNorm(vb.data.get_lowrank([10,20,30,100], 10)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - TensorSpectralNorm', passes=20_000, sec=120, metrics='train loss', vid_scale=None)

        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.Schur(vb.data.FROG96, 10).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'Linalg - Schur', passes=20_000, sec=120, metrics='train loss', vid_scale=None)


    def run_real(self):
        # ---------------------------- Human heart dipole ---------------------------- #
        # ndim = 8
        # Adam - ?, SOAP - ?
        bench = vb.HumanHeartDipole() # NO CUDA
        self.run_bench(bench, "Real - Human heart dipole", passes=20_000, sec=120, metrics='train loss', vid_scale=None)

        # ---------------------------- Propane combustion ---------------------------- #
        # ndim = 11
        # Adam - ?, SOAP - ?
        bench = vb.PropaneCombustion() # NO CUDA
        self.run_bench(bench, "Real - Propane combustion", passes=20_000, sec=120, metrics='train loss', vid_scale=None)

        # -------------------------------- Muon coeffs ------------------------------- #
        # ndim = 15
        # Adam - ?, SOAP - ?
        bench = vb.MuonCoeffs(resolution=(512, 512)) # NO CUDA
        self.run_bench(bench, 'Real - Muon coefficients', passes=20_000, sec=120, metrics='train loss', vid_scale=1)

        # ------------------------------ Style transfer ------------------------------ #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.StyleTransfer(vb.data.FROG96, vb.data.GEOM96).to(CUDA_IF_AVAILABLE)
        bench_name = "Real - Style Transfer"
        self.run_bench(bench, bench_name, passes=20_000, sec=120, metrics='train loss', vid_scale=2)


    def run_ml(self):
        # --------------------- TinyConvNet (full-batch MNIST-1D) -------------------- #
        # strong overfitting, may be good to study generalization
        # ndim = 4,098
        # Adam - ?, SOAP - ?
        bench = vb.Mnist1d(vb.models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "ML - TinyConvNet", passes=20_000, sec=120, metrics = LOSSES, vid_scale=None)

        # ------------------------------ PINN (Wave PDE) ----------------------------- #
        # ndim = 132,611
        # Adam - ?, SOAP - ?
        bench = vb.WavePINN(vb.WavePINN.FLS(2, 1, hidden_size=512, n_hidden=3)).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Wave PDE - FLS', passes=20_000, sec=240, metrics='train loss', vid_scale=4)

        # -------------------------------- LogisticRegression ------------------------------- #
        bench = vb.Collinear(vb.models.MLP([128, 10]), n_features=128, train_split=0.99).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'ML - Logistic Regression', passes=2_000, sec=120, metrics='train loss', vid_scale=None)


    def run_mls(self):
        # --------------------------------- LR BS-128 -------------------------------- #
        # ndim = 1290
        # AdamW - 23s, SOAP - 35S
        bench = vb.Collinear(vb.models.MLP([128, 10]), n_features=128, batch_size=128).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - Mini-batch Logistic Regression BS-128', passes=20_000, sec=120, metrics='test loss', test_every=10, vid_scale=None)

        # ------------------------ Online logistic regression ------------------------ #
        # ndim = 385
        # Adam - ?, SOAP - ?
        bench = vb.Collinear(vb.models.MLP([128, 10]), n_features=128, batch_size=1).to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - Online Logistic Regression'
        self.run_bench(bench, bench_name, passes=20_000, sec=120, test_every=10, metrics='test loss', vid_scale=None)

        # --------------------------- Matrix factorization --------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.MFMovieLens("/run/media/jj/HDD/datasets/MovieLens 100K", batch_size=32, device='cuda').to(CUDA_IF_AVAILABLE)
        bench_name = 'MLS - MovieLens BS-32 - Matrix Factorization'
        self.run_bench(bench, bench_name, passes=20_000, sec=120, test_every=10, metrics='test loss', vid_scale=None)

        # --------------------- TinyConvNet (MNIST-1D) -------------------- #
        # ndim = 4,098
        # Adam - ?, SOAP - ?
        bench = vb.Mnist1d(vb.models.vision.TinyConvNet(40, 1, 10, act_cls=nn.ELU), batch_size=32).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, "MLS - MNIST-1D BS-32 - TinyConvNet", passes=20_000, sec=120, test_every=10, metrics = "test loss", vid_scale=None)

        # ----------------------- Sparse Autoencoder (MNIST-1D) ---------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.Mnist1dAutoencoding(
            vb.models.vision.ConvNetAutoencoder(1, 1, 1, 40, encoder=(64,96,128,256), sparse_reg=0.1),
            batch_size=32, test_batch_size=256
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - MNIST-1D Sparse Autoencoder BS-32 - ConvNet', passes=20_000, sec=120, test_every=50, metrics='test loss', vid_scale=None)

        # ---------------------------- ConvNet (SynthSeg) ---------------------------- #
        # ndim = ?
        # Adam - ?, SOAP - ?
        bench = vb.SynthSeg1d(
            vb.models.vision.ConvNetAutoencoder(1, 1, 5, 32, encoder=(64,96,128)),
            criterion = monai.losses.DiceFocalLoss(softmax=True),
            num_samples=10_000, batch_size=64, test_batch_size=512
        ).to(CUDA_IF_AVAILABLE)
        self.run_bench(bench, 'MLS - SynthSeg BS-64 - ConvNet', passes=20_000, sec=240, test_every=50, metrics='test loss', vid_scale=None)


