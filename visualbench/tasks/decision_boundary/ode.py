import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint

from ..._utils import CUDA_IF_AVAILABLE
from .decision_boundary import DecisionBoundary


class ODELinear(nn.Module):
    def __init__(self, width, act = F.softplus):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.act = act

    def forward(self, t, z):
        return self.act(self.linear((z - z.mean()) / z.std()))


class NeuralODE(nn.Module):
    def __init__(self, input_dim = 2, width = 8, output_dim = 1, act = F.softplus, T = 10., steps = 2, adjoint = False):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, width)
        self.ode_func = ODELinear(width, act = act)
        self.head = nn.Linear(width, output_dim)
        self.T = T
        self.adjoint = adjoint
        self.t = nn.Buffer(torch.linspace(0, self.T, steps))# integration times from 0 to T)

    def forward(self, x):
        z0 = self.in_layer(x)

        if self.adjoint: zT = odeint_adjoint(self.ode_func, z0, self.t, method='rk4')[1]  # [1] selects t=T # type:ignore
        else: zT = odeint(self.ode_func, z0, self.t, method = 'rk4')[1]

        out = self.head(zT)
        return out



def MoonsODE(width=8, noise=0.2, n_samples=2048, batch_size = None, act = F.softplus, adjoint=False, T=10., steps = 2, resolution=192, device = CUDA_IF_AVAILABLE):
    from sklearn.datasets import make_moons
    return DecisionBoundary(
        *make_moons(n_samples=n_samples, noise = noise, random_state=0),
        model = NeuralODE(width=width, T = T, steps=steps, act = act, adjoint=adjoint),
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits,
        batch_size=batch_size,
        resolution=resolution,
        device = device,
    )