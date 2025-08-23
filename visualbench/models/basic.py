import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, Literal, TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid

from ..utils import nn_tools
if TYPE_CHECKING:
    from ..benchmark import Benchmark

def _visualize_linear(linear: nn.Module, benchmark: "Benchmark", vis_shape: tuple[int,int] | None, max_tiles:int):
    if vis_shape is None: return
    if not hasattr(linear, "weight"): return

    weight = cast(torch.Tensor, linear.weight)
    channels = weight.view(-1, *vis_shape)[:max_tiles].unsqueeze(1)
    grid = make_grid(channels, nrow=max(math.ceil(math.sqrt(channels.size(0))), 1), padding=1, pad_value=channels.amax().item())
    benchmark.log_image("1st layer weights", grid, to_uint8=True, log_difference=True)

class Regularized(nn.Module):
    def __init__(self, model: nn.Module, l1:float|None=None, l2:float|None=None):
        super().__init__()
        self.model = model
        self.l1 = l1
        self.l2 = l2

    def forward(self, x):
        ret = self.model(x)
        penalty = 0
        if self.l1 is not None and self.l1 != 0:
            penalty = sum(p.abs().sum() for p in self.model.parameters()) * self.l1
        if self.l2 is not None and self.l2 != 0:
            penalty = penalty + sum(p.pow(2).sum() for p in self.model.parameters()) * self.l2
        return ret, penalty

    def after_get_loss(self, benchmark: "Benchmark"):
        if hasattr(self.model, "after_get_loss"):
            self.model.after_get_loss(benchmark) #pyright:ignore[reportCallIssue]

class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int | Iterable[int] | None,
        act_cls: Callable | None = nn.ELU,
        bn: bool = False,
        dropout: float = 0,
        vis_shape: tuple[int,int] | None = None,
        max_tiles: int = 100,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        if hidden is None: hidden = []
        channels = [in_channels] + list(hidden) + [out_channels]

        layers = []
        for i,o in zip(channels[:-2], channels[1:-1]):
            layer = [cls(i, o, not bn), act_cls() if act_cls is not None else nn.Identity(), nn.BatchNorm1d(o) if bn else nn.Identity(), nn.Dropout(dropout) if dropout>0 else nn.Identity()]
            layers.append(nn_tools.Sequential(*layer))

        self.layers = nn_tools.Sequential(*layers)
        self.head = cls(channels[-2], channels[-1])
        self.vis_shape = vis_shape
        self.max_tiles = max_tiles

    def forward(self, x):
        x = x.flatten(1,-1)
        for l in self.layers: x = l(x)
        return self.head(x)

    def after_get_loss(self, benchmark: "Benchmark"):
        _visualize_linear(self.layers[0][0], benchmark, self.vis_shape, self.max_tiles)



class RecurrentMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        n_passes: int,
        merge: bool = True,
        act_cls: Callable | None = nn.ELU,
        dropout: float = 0,
        bn=True,
        vis_shape: tuple[int,int] | None = None,
        max_tiles: int = 100,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        self.n_passes = n_passes
        if merge and in_channels == width: self.first = nn.Identity()
        else: self.first = cls(in_channels, width)

        linear = cls(width,width, not bn)
        self.rec = nn_tools.Sequential(linear, act_cls() if act_cls is not None else nn.Identity())
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(width) if bn else nn.Identity() for _ in range(n_passes)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.vis_shape = vis_shape
        self.max_tiles = max_tiles
        self.head = cls(width,out_channels)

    def forward(self, x):
        x = x.flatten(1,-1)
        x = self.first(x)
        for bn in self.batch_norms: x = self.drop(bn(self.rec(x)))
        return self.head(x)

    def after_get_loss(self, benchmark: "Benchmark"):
        if isinstance(self.first, nn.Identity): layer = self.rec[0]
        else: layer = self.first
        _visualize_linear(layer, benchmark, self.vis_shape, self.max_tiles)

class RNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, rnn: Callable[..., nn.Module]=nn.LSTM):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = rnn(in_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
         # from (batch_size, 40) to (batch_size, 40, 1) otherwise known as (batch_size, seq_length, input_size)
        if x.ndim == 2 and self.in_channels == 1: x = x.unsqueeze(2)

        out, _ = self.rnn(x) # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :] # last timestep's output (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, num_classes)
        return out
