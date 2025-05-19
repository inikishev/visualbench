from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import nn_tools


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int | Iterable[int] | None,
        act: Callable | None = F.relu,
        bn: bool = False,
        dropout: float = 0,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        if hidden is None: hidden = []
        channels = [in_channels] + list(hidden) + [out_channels]

        layers = []
        for i,o in zip(channels[:-2], channels[1:-1]):
            layer = [cls(i, o, not bn), act if act is not None else nn.Identity(), nn.BatchNorm1d(o) if bn else nn.Identity(), nn.Dropout(dropout) if dropout>0 else nn.Identity()]
            layers.append(nn_tools.Sequential(*layer))

        self.layers = nn_tools.Sequential(*layers)
        self.head = cls(channels[-2], channels[-1])

    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.head(x)

class RecurrentMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        n_passes: int,
        merge: bool = True,
        act: Callable | None = F.leaky_relu,
        dropout: float = 0,
        bn=True,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        self.n_passes = n_passes
        if merge and in_channels == width: self.first = nn.Identity()
        else: self.first = cls(in_channels, width)

        linear = cls(width,width, not bn)
        self.rec = nn_tools.Sequential(linear, act if act is not None else nn.Identity())
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(width) if bn else nn.Identity() for _ in range(n_passes)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.head = cls(width,out_channels)

    def forward(self, x):
        x = self.first(x)
        for bn in self.batch_norms: x = self.drop(bn(self.rec(x)))
        return self.head(x)

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
