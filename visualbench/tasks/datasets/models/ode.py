from functools import partial

import torch
from sklearn.datasets import fetch_california_housing, make_moons
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint


class _ODELinear(nn.Module):
    def __init__(self, width, act = F.softplus, bn = True):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act = act
        self.bn = nn.BatchNorm1d(width, track_running_stats=False) if bn else nn.Identity()

    def forward(self, t, z):
        return self.bn(self.act(self.linear(z)))


class _NeuralODE(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 1, width = 8, act = F.softplus, bn=True, T = 10., steps = 2, adjoint = False):
        super().__init__()
        self.in_layer = nn.Linear(in_channels, width)
        self.ode_func = _ODELinear(width, act = act, bn=bn)
        self.head = nn.Linear(width, out_channels)
        self.T = T
        self.adjoint = adjoint
        self.t = nn.Buffer(torch.linspace(0, self.T, steps))# integration times from 0 to T)

    def forward(self, x):
        z0 = self.in_layer(x)

        if self.adjoint: zT = odeint_adjoint(self.ode_func, z0, self.t, method='rk4')[1]  # [1] selects t=T # type:ignore
        else: zT = odeint(self.ode_func, z0, self.t, method = 'rk4')[1]

        out = self.head(zT)
        return out

def NeuralODE(width = 8, act = F.softplus, bn=True, T = 10., steps = 2, adjoint = False):
    return partial(_NeuralODE, width = width, act = act, bn = bn, T = T, steps = steps, adjoint = adjoint)