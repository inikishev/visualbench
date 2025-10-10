import math
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F


def _raise_torchvision(*args, **kwargs):
    raise ModuleNotFoundError("torchvision is required for linear layer visualization")

if find_spec("torchvision") is not None:
    from torchvision.utils import make_grid
else:
    make_grid = _raise_torchvision

if TYPE_CHECKING:
    from ..benchmark import Benchmark

def ConvNd(ndim: int):
    if ndim == 1: return nn.Conv1d
    if ndim == 2: return nn.Conv2d
    if ndim == 3: return nn.Conv3d
    raise ValueError(ndim)

def ConvTransposeNd(ndim: int):
    if ndim == 1: return nn.ConvTranspose1d
    if ndim == 2: return nn.ConvTranspose2d
    if ndim == 3: return nn.ConvTranspose3d
    raise ValueError(ndim)

def MaxPoolNd(ndim: int):
    if ndim == 1: return nn.MaxPool1d
    if ndim == 2: return nn.MaxPool2d
    if ndim == 3: return nn.MaxPool3d
    raise ValueError(ndim)

def DropoutNd(ndim: int):
    if ndim == 1: return nn.Dropout1d
    if ndim == 2: return nn.Dropout2d
    if ndim == 3: return nn.Dropout3d
    raise ValueError(ndim)

def BatchNormNd(ndim: int):
    if ndim == 1: return nn.BatchNorm1d
    if ndim == 2: return nn.BatchNorm2d
    if ndim == 3: return nn.BatchNorm3d
    raise ValueError(ndim)


class TinyConvNet(nn.Module):
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int, act_cls: Callable = nn.ReLU,):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        Conv = ConvNd(len(in_size))
        self.c1 = nn.Sequential(Conv(in_channels, 16, 3, 2), act_cls())
        self.c2 = nn.Sequential(Conv(16, 24, 3, 2), act_cls())
        self.c3 = nn.Sequential(Conv(24, 32, 2, 2), act_cls())

        dummy = torch.randn(1,in_channels,*in_size)
        dummy = self.c3(self.c2(self.c1(dummy))).flatten(1,-1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(dummy.size(1), out_channels))

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.head(x)


class TinyWideConvNet(nn.Module):
    """3,626 params if ndim=1."""
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int,  act_cls: Callable = nn.ReLU, dropout=0.5):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        self.ndim = len(in_size)

        Conv = ConvNd(self.ndim)
        MaxPool = MaxPoolNd(self.ndim)
        Dropout = DropoutNd(self.ndim)

        self.c1 = nn.Sequential(
            Conv(in_channels, 8, kernel_size=5), # ~37
            MaxPool(2), # ~18
            act_cls(),
        )

        self.c2 = nn.Sequential(
            Conv(8, 16, kernel_size=5), # ~15
            MaxPool(2), # ~7
            act_cls(),
            Dropout(dropout),
        )
        self.c3 = nn.Sequential(
            Conv(16, 32, kernel_size=5),
            Dropout(dropout),
        )

        self.linear = nn.Linear(32, out_channels)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        dims = [-i for i in range(1, self.ndim+1)]
        x = self.c3(x).mean(dims)
        return self.linear(x)


class TinyLongConvNet(nn.Module):
    """1,338 params if ndim=1"""
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int, act_cls: Callable = nn.ReLU, dropout=0.0):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        self.ndim = len(in_size)

        Conv = ConvNd(self.ndim)
        Dropout = DropoutNd(self.ndim)
        BatchNorm = BatchNormNd(self.ndim)

        self.c1 = nn.Sequential(
            Conv(in_channels, 4, kernel_size=2, bias=False),
            act_cls(),
            BatchNorm(4, track_running_stats=False),

            Conv(4, 4, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(4, track_running_stats=False),
        )

        self.c2 = nn.Sequential(
            Conv(4, 8, kernel_size=2, bias=False),
            act_cls(),
            BatchNorm(8, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),

            Conv(8, 8, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(8, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.c3 = nn.Sequential(
            Conv(8, 16, kernel_size=2, bias=False),
            act_cls(),
            BatchNorm(16, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),

            Conv(16, 16, kernel_size=2, stride=2, bias=False),
            BatchNorm(16, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.linear = nn.Linear(16, out_channels)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)

        dims = [-i for i in range(1, self.ndim+1)]
        x = self.c3(x).mean(dims)
        return self.linear(x)


class ConvNet(nn.Module):
    """134,410 params if ndim=1"""
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int, widths=(32, 64, 128, 256), kernel_size=3, act_cls:Callable = nn.ReLU, dropout=0.2):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        self.ndim = len(in_size)

        Conv = ConvNd(self.ndim)
        Dropout = DropoutNd(self.ndim)
        BatchNorm = BatchNormNd(self.ndim)
        MaxPool = MaxPoolNd(self.ndim)

        layers = []
        widths = [in_channels] + list(widths)
        for i,o in zip(widths[:-1], widths[1:]):

            layers.append(
                nn.Sequential(
                    Conv(i, o, kernel_size=kernel_size, bias=False),
                    MaxPool(2),
                    act_cls(),
                    BatchNorm(o, track_running_stats=False),
                    Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )

        self.layers = nn.Sequential(*layers)
        dummy = torch.randn(1,in_channels,*in_size)
        dummy = self.layers(dummy).flatten(1,-1)
        self.linear = nn.Linear(dummy.size(1), out_channels)

    def forward(self, x):
        if x.ndim == self.ndim + 1: x = x.unsqueeze(1)
        x = self.layers(x)
        return self.linear(x.flatten(1,-1))




class FastConvNet(nn.Module):
    """? params if ndim=1"""
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int, act_cls: Callable = nn.ReLU, dropout=0.2):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        ndim = len(in_size)

        Conv = ConvNd(ndim)
        Dropout = DropoutNd(ndim)
        BatchNorm = BatchNormNd(ndim)

        self.c1 = nn.Sequential(
            Conv(in_channels, 64, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(64, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c2 = nn.Sequential(
            Conv(64, 96, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(96, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c3 = nn.Sequential(
            Conv(96, 128, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(128, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c4 = nn.Sequential(
            Conv(128, 256, kernel_size=2, stride=2, bias=False),
            act_cls(),
            BatchNorm(256, track_running_stats=False),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        dummy = torch.randn(1,in_channels,*in_size)
        dummy = self.c4(self.c3(self.c2(self.c1(dummy)))).flatten(1,-1)
        self.linear = nn.Linear(dummy.size(1), out_channels)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.linear(x)



class MobileNet(nn.Module):
    """6,228 params if ndim=1"""
    def __init__(self, in_size:int | Sequence[int], in_channels:int, out_channels:int, act_cls: Callable = nn.ReLU, dropout=0.5):
        super().__init__()
        if isinstance(in_size, int): in_size = (in_size, )
        self.ndim = len(in_size)

        Conv = ConvNd(self.ndim)
        Dropout = DropoutNd(self.ndim)

        self.c1 = nn.Sequential(
            Conv(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ChannelShuffle(8),
            act_cls(),
        )

        self.c2 = nn.Sequential(
            Conv(32, 32, kernel_size=3, groups=32, padding=1),
            act_cls(),
            Dropout(dropout) if dropout > 0 else nn.Identity(),

            Conv(32, 64, kernel_size=1, groups=8, padding=1, ),
            nn.ChannelShuffle(16),
            act_cls(),
            Dropout(dropout) if dropout > 0 else nn.Identity(),

            Conv(64, 128, kernel_size=3, stride=2, padding=1, groups=64, ),
            act_cls(),
            Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c3 = nn.Sequential(
            Conv(128, 10, kernel_size=1),
        )

        dummy = torch.randn(1,in_channels,*in_size)
        dummy = self.c3(self.c2(self.c1(dummy))).flatten(1,-1)
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(384, out_channels))

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        dims = [-i for i in range(1, self.ndim+1)]
        return x.mean(dims)

def convblocknd(in_channels, out_channels, kernel_size, stride, padding, act_cls, bn: bool, dropout:float|None, transpose=False, ndim:int=2):
    ConvCls = ConvTransposeNd(ndim) if transpose else ConvNd(ndim)
    # UCPAND
    modules: list = [ConvCls(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not bn)]
    if act_cls is not None:
        modules.append(act_cls())

    if bn:
        modules.append(BatchNormNd(ndim)(out_channels, track_running_stats=False))

    if dropout is not None and dropout != 0:
        modules.append(DropoutNd(ndim)(dropout))

    return nn.Sequential(*modules)


class ConvNetAutoencoder(nn.Module):
    def __init__(
        self,
        ndim:int,
        in_channels: int,
        out_channels: int,
        out_size: int,
        hidden=(32, 64, 128, 256),
        act_cls:Callable=nn.ReLU,
        bn=True,
        dropout=None,
        sparse_reg: float | None = None,
        squeeze:bool=True,

    ):
        super().__init__()
        if isinstance(hidden, int): hidden = [hidden]
        channels = [in_channels] + list(hidden) # in_channels is always 1 cos conv net

        self.enc = nn.Sequential(
            *[convblocknd(i, o, 2, 2, 0, act_cls=act_cls, bn=bn, dropout=dropout, ndim=ndim) for i, o in zip(channels[:-1], channels[1:])]
        )


        rev = list(reversed(channels))
        self.dec = nn.Sequential(
            *[convblocknd(i, o, 3, 2, 0, act_cls=act_cls, bn=bn, dropout=dropout, transpose=True, ndim=ndim) for i, o in zip(rev[:-2], rev[1:-1])]
        )

        self.head = nn.Sequential(
            *convblocknd(rev[-2], rev[-2], 2, 2, 0, act_cls=act_cls, bn=bn, dropout=dropout, transpose=True, ndim=ndim),
            convblocknd(rev[-2], out_channels, 2, 1, 0, act_cls=None, bn=False, dropout=None, ndim=ndim)
        )

        self.sparse_reg = sparse_reg
        self.out_size = out_size
        self.squeeze = squeeze
        self.out_channels = out_channels
        self.ndim = ndim
        self.x_vis = None

    def forward(self, x):
        if x.ndim == self.ndim+1: x = x.unsqueeze(1)
        if self.x_vis is None: self.x_vis = x[:100]

        features = self.enc(x)
        x = self.dec(features)

        x = self.head(x)

        x = x[:,:,:self.out_size]
        if self.ndim >= 2: x = x[:,:,:,:self.out_size]
        if self.ndim == 3: x = x[:,:,:,:,:self.out_size]

        if self.squeeze and self.out_channels == 1:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.sparse_reg is not None: return x, features.abs().mean() * self.sparse_reg

        return x

    @torch.no_grad
    def after_get_loss(self, benchmark: "Benchmark"):
        if self.ndim in (1, 3): return

        x = self.x_vis
        features = self.enc(x)
        x = self.dec(features)
        x = self.head(x)

        grid = make_grid(x, nrow=max(math.ceil(math.sqrt(x.size(0))), 1), padding=1, normalize=True, scale_each=True)
        benchmark.log_image("outputs", grid, to_uint8=True)



