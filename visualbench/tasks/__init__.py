from .colorization import Colorization
from .datasets import *
from .drawing import LinesDrawer, PartitionDrawer, RectanglesDrawer
from .function_descent import FunctionDescent, test_functions
from .graph_layout import (
    GraphLayout,
    barbell_graph,
    complete_graph,
    grid_graph,
    watts_strogatz_graph,
)
from .lennard_jones_clusters import LennardJonesClusters
from .linalg import *
from .lstm import RNNArgsort
from .marbles import MARBLE_COURSE, MarbleRace
from .operations import Sorting
from .optimal_control import OptimalControl
from .packing import BoxPacking, SpherePacking, SquishyBoxPacking

# from .gnn import GraphNN
from .particles import *
from .smale7 import Smale7
from .style_transfer import StyleTransfer
from .synthetic import (
    AlphaBeta1,
    Convex,
    NonlinearMatrixFactorization,
    Rosenbrock,
    SelfRecurrent,
    Sphere,
)
from .katos import KatosProblem
from .normal_scalar_curvature import NormalScalarCurvature