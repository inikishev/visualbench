from functools import partial
from typing import Literal, Any
from collections.abc import Callable

import torch
from myai import nn as mynn
from torch import nn
from torch.nn import functional as F


class _MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, act: Callable | None = F.relu, bn = False, dropout:float=0, cls: Callable = nn.Linear):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        if hidden is None: hidden = []
        channels = [in_channels] + list(hidden) + [out_channels]

        layers = []
        for i,o in zip(channels[:-2], channels[1:-1]):
            layer = [cls(i, o, not bn), act if act is not None else nn.Identity(), nn.BatchNorm1d(o) if bn else nn.Identity(), nn.Dropout(dropout) if dropout>0 else nn.Identity()]
            layers.append(mynn.Sequential(*layer))

        self.layers = mynn.Sequential(*layers)
        self.head = cls(channels[-2], channels[-1])

    def forward(self, x):
        for l in self.layers: x = l(x)
        return self.head(x)

def MLP(hidden, act: Callable | None = F.relu, bn = False, dropout:float=0, cls: Callable = nn.Linear):
    return partial(_MLP, hidden = hidden, act = act, bn = bn, dropout = dropout, cls = cls)

class _RecurrentMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        n_passes,
        merge: bool = True,
        act: Callable | None = F.leaky_relu,
        dropout: float = 0,
        bn=True,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        self.n_passes = n_passes
        if merge and in_channels == width: in_channels = None
        self.first = cls(in_channels, width) if in_channels is not None else nn.Identity()
        linear = cls(width,width, not bn)
        self.rec = mynn.Sequential(linear, act if act is not None else nn.Identity())
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(width) if bn else nn.Identity() for _ in range(n_passes)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.head = cls(width,out_channels)

    def forward(self, x):
        x = self.first(x)
        for bn in self.batch_norms: x = self.drop(bn(self.rec(x)))
        return self.head(x)

def RecurrentMLP(width, n_passes, merge: bool = True, act: Callable | None = F.leaky_relu, dropout: float = 0, bn = True, cls: Callable = nn.Linear):
    return partial(_RecurrentMLP, width=width, n_passes=n_passes, merge=merge, act=act, dropout=dropout, bn = bn, cls=cls)


class _Mnist1dRNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, rnn: Callable[..., nn.Module]=nn.LSTM):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = rnn(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        x = x.unsqueeze(2) # from (batch_size, 40) to (batch_size, 40, 1) otherwise known as (batch_size, seq_length, input_size)
        out, _ = self.rnn(x) # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :] # last timestep's output (batch_size, hidden_size)
        out = self.fc(out) # (batch_size, num_classes)
        return out

def Mnist1dRNN(hidden_size, num_layers, rnn: Callable[..., nn.Module] = nn.LSTM):
    return partial(_Mnist1dRNN, hidden_size=hidden_size, num_layers=num_layers, rnn = rnn)

_DEPTH_TO_OUT_SIZE = {
    4: 2,
    5: 1,
    3: 5,
}
class _Mnist1dConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden = (32, 64, 128, 256), act = 'relu', norm = 'fixedbn', dropout=None):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        channels = [1] + list(hidden) # in_channels is always 1 cos conv net

        self.enc = torch.nn.Sequential(
            *[mynn.ConvBlock(i, o, 2, 2, act=act, norm=norm, ndim=1, dropout=dropout) for i, o in zip(channels[:-1], channels[1:])]
        )
        self.head = mynn.LinearBlock(channels[-1]*_DEPTH_TO_OUT_SIZE[len(hidden)], out_channels, flatten=True)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.enc(x)
        return self.head(x)

def Mnist1dConvNet(hidden = (32, 64, 128, 256), act = 'relu', norm = 'fixedbn', dropout=None):
    """only for mnist1d"""
    return partial(_Mnist1dConvNet, hidden = hidden, act = act, norm = norm, dropout=dropout)


class _Mnist1dRecurrentConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, width: int = 64,num: int = 4, act = 'relu', norm = 'fixedbn'):
        super().__init__()
        self.first = mynn.ConvBlock(1, width, 2, 2, act=act, norm=norm, ndim = 1)
        self.rec = mynn.ConvBlock(width, width, 2, 2, act=act, norm=norm, ndim = 1)
        self.head = mynn.LinearBlock(width*_DEPTH_TO_OUT_SIZE[num+1], out_channels, flatten=True)
        self.num = num

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.first(x)
        for _ in range(self.num): x = self.rec(x)
        return self.head(x)

def Mnist1dRecurrentConvNet(width = 64, num = 4, act = 'relu', norm = 'fixedbn'):
    return partial(_Mnist1dRecurrentConvNet, width = width, num = num, act = act, norm = norm)

class _Mnist1dConvNetAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden = (32, 64, 128, 256), act = 'relu', norm = 'fixedbn', dropout=None):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        channels = [1] + list(hidden) # in_channels is always 1 cos conv net

        self.enc = nn.Sequential(
            *[mynn.ConvBlock(i, o, 2, 2, act=act, norm=norm, ndim=1, dropout=dropout) for i, o in zip(channels[:-1], channels[1:])]
        )

        rev = list(reversed(channels))
        self.dec = nn.Sequential(
            *[mynn.ConvTransposeBlock(i, o, 3, 2, act=act, norm=norm, ndim=1, dropout=dropout) for i, o in zip(rev[:-2], rev[1:-1])]
        )

        self.head = nn.Sequential(
            mynn.ConvTransposeBlock(rev[-2], rev[-2], 2, 2, act = act, norm = norm, dropout = dropout, ndim = 1),
            mynn.ConvBlock(rev[-2], rev[-1], 2, ndim = 1)
        )

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.enc(x)
        x = self.dec(x)
        return self.head(x)[:,0,:40]

def Mnis1dConvNetAutoencoder(hidden = (32, 64, 128, 256), act: Any = 'relu', norm: Any = 'fixedbn', dropout = None):
    return partial(_Mnist1dConvNetAutoencoder, hidden = hidden, act = act, norm = norm, dropout = dropout)


def Unet1d(channels = (32, 64, 128, 256), skip_mode:Any='cat'):
    from myai.nn.nets.unet import UNet
    return partial(UNet, ndim = 1, channels = channels[1:], first_out_channels=channels[0], skip_mode=skip_mode)