from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal, cast

from . import projected
from .alpha_evolve_b1 import AlphaEvolveB1
from .char_rnn import CharRNN
from .colorization import Colorization
from .cutest import CUTEst
from .data_fit import FitData
from .datasets import *
from .function_approximator import FunctionApproximator
from .function_descent import (
    TEST_FUNCTIONS,
    DecisionSpaceDescent,
    FunctionDescent,
    MetaLearning,
    MultiAgentDescent,
    SimultaneousFunctionDescent,
    test_functions,
)
from .glimmer import Glimmer
from .gmm import GaussianMixtureNLL
from .kato import Kato
from .lennard_jones_clusters import LennardJonesClusters
from .linalg import *
from .matrix_factorization import MFMovieLens
from .minpack2 import HumanHeartDipole, PropaneCombustion
from .muon_coeffs import MuonCoeffs
from .operations import Sorting

# # from .gnn import GraphNN
from .particles import *
from .pde import WavePINN
from .registration import AffineRegistration, DeformableRegistration
from .rnn import RNNArgsort
from .slop import *
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

