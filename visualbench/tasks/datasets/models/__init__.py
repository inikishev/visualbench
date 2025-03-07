from collections.abc import Callable

import torch

from .basic import MLP, RecurrentMLP, Mnist1dConvNet, Mnist1dRNN, Mnist1dRecurrentConvNet
from .ode import NeuralODE

ModelClass = Callable[[int,int], torch.nn.Module]