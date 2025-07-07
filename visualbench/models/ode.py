import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint, odeint_adjoint

class _ODELinear(nn.Module):
    def __init__(self, width, act_cls = nn.Softplus, layer_norm=False):
        super().__init__()
        self.linear = nn.Linear(width + 1, width)
        self.act = act_cls()
        self.layer_norm = nn.LayerNorm(width) if layer_norm else nn.Identity()

    def forward(self, t, z: torch.Tensor):
        z = torch.cat([z, torch.full((z.size(0), 1), fill_value=float(t), device=z.device, dtype=z.dtype)], dim=1)
        return self.layer_norm(self.act(self.linear(z)))

# test 'dopri5', 'adams'
class NeuralODE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, act_cls = F.softplus, layer_norm=False, T = 10., steps = 2, adjoint = False, method = 'implicit_adams'):
        super().__init__()
        self.in_layer = nn.Linear(in_channels, width)
        self.ode_func = _ODELinear(width, act_cls = act_cls, layer_norm=layer_norm)
        self.head = nn.Linear(width, out_channels)
        self.T = T
        self.adjoint = adjoint
        self.t = nn.Buffer(torch.linspace(0, self.T, steps))# integration times from 0 to T)
        self.method = method

    def forward(self, x):
        z0 = self.in_layer(x)

        if self.adjoint: zT = odeint_adjoint(self.ode_func, z0, self.t, method=self.method)[1]  # [1] selects t=T # type:ignore
        else: zT = odeint(self.ode_func, z0, self.t, method=self.method)[1]

        out = self.head(zT)
        return out

