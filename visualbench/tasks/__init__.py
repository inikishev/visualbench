from .box_packing import BoxPacking, RotatingBoxPacking
from .char_rnn import CharLSTM
from .decision_boundary import DecisionBoundary, HeavyRegMoons2D
from .diffusion import DiffusionMNIST1D
from .function_descent import FunctionDescent
from .gymnasium import Gymnasium
from .image_poly import PolynomialReconstructor
from .image_rectanges import RectangleReconstructor
from .inverse import MatrixInverse
from .mnist1d import (
    MNIST1D,
    MNIST1D_ConvNet_Fullbatch,
    MNIST1D_ConvNet_Minibatch,
    MNIST1D_LogisticRegression,
    MNIST1D_LSTM_Minibatch,
    MNIST1D_MLP_Minibatch,
    MNIST1D_ResNet_Fullbatch,
    MNIST1D_ResNet_Minibatch,
    MNIST1D_ResNet_Online,
    MNIST1D_SparseAutoencoder,
)
from .optimal_control import OptimalControl

from .sphere import Sphere
from .sphere_packing import SpherePacking
from .style_transfer import StyleTransfer
from .xor import XOR, DelayedXOR
from .gnn import TorchGeometricDataset, GCN

from .synthetic import Convex, Rosenbrock, NonlinearMatrixFactorization