from .colorization import Colorization
from .covering import RigidBoxCovering
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
from .hadamard import Hadamard
from .katos import KatosProblem
from .lennard_jones_clusters import LennardJonesClusters
from .linalg import *
from .normal_scalar_curvature import NormalScalarCurvature
from .operations import Sorting
from .optimal_control import OptimalControl
from .packing import BoxPacking, RigidBoxPacking, SpherePacking

# # from .gnn import GraphNN
from .particles import *
from .rnn import RNNArgsort
from .smale7 import Smale7
from .steiner import SteinerSystem
from .style_transfer import StyleTransfer
from .synthetic import (
    IllConditioned,
    QuadraticForm,
    Rosenbrock,
    Sphere,
    PowellSingular,
    VariablyDimensional,
    BroydenTridiagonal,
)
