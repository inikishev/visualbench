from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, cast

from . import projected
from .alpha_evolve_b1 import AlphaEvolveB1
from .char_rnn import CharRNN
from .colorization import Colorization
from .covering import RigidBoxCovering
from .cutest import CUTEst
from .datasets import *
from .drawing import (
    LayerwiseNeuralDrawer,
    LinesDrawer,
    NeuralDrawer,
    PartitionDrawer,
    RectanglesDrawer,
)
from .function_approximator import FunctionApproximator
from .function_descent import (
    FunctionDescent,
    MetaLearning,
    NeuralDescent,
    SimultaneousFunctionDescent,
    test_functions,
    TEST_FUNCTIONS,
)
from .glimmer import Glimmer
from .gmm import GaussianMixtureNLL
from .graph_layout import GraphLayout
from .hadamard import Hadamard
from .kato import Kato
from .lennard_jones_clusters import LennardJonesClusters
from .linalg import *
from .matrix_factorization import MFMovieLens
from .minpack2 import HumanHeartDipole, PropaneCombustion
from .muon_coeffs import MuonCoeffs
from .normal_scalar_curvature import NormalScalarCurvature
from .operations import Sorting
from .optimal_control import OptimalControl
from .packing import BoxPacking, RigidBoxPacking, SpherePacking

# # from .gnn import GraphNN
from .particles import *
from .pde import WavePINN
from .registration import AffineRegistration, DeformableRegistration
from .rnn import RNNArgsort
from .smale7 import Smale7
from .steiner import SteinerSystem
from .style_transfer import StyleTransfer
from .synthetic import (
    Ackley,
    ChebushevRosenbrock,
    Rastrigin,
    Rosenbrock,
    RotatedQuadratic,
    Sphere,
)
from .tsne import TSNE

if TYPE_CHECKING or find_spec('gpytorch') is not None:
    from .guassian_processes import GaussianProcesses
else:
    GaussianProcesses = None

