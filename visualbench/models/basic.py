import itertools
import math
import random
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import nn_tools


def _raise_torchvision(*args, **kwargs):
    raise ModuleNotFoundError("torchvision is required for linear layer visualization")

if find_spec("torchvision") is not None:
    from torchvision.utils import make_grid
else:
    make_grid = _raise_torchvision


if TYPE_CHECKING:
    from ..benchmark import Benchmark
    from ..tasks.datasets import DatasetBenchmark

def _visualize_linear(linear: nn.Module, benchmark: "Benchmark", vis_shape: tuple[int,int] | None, max_tiles:int):
    if vis_shape is None: return
    if not hasattr(linear, "weight"): return

    weight = cast(torch.Tensor, linear.weight)
    channels = weight.view(-1, *vis_shape)[:max_tiles].unsqueeze(1)
    grid = make_grid(channels, nrow=max(math.ceil(math.sqrt(channels.size(0))), 1), padding=1, pad_value=channels.amax().item())
    benchmark.log_image("1st layer weights", grid, to_uint8=True, log_difference=True)

class Regularized(nn.Module):
    """Wrapper around another model which adds l1 and l2 regularization when the model is used in a benchmark.

    for example

    ```python
    vb.models.Regularized(
        vb.models.MLP([10, 128, 3]),
        l2=1e-2,
    )
    ```
    """
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
    """Multi-layer perceptorn.

    for example, if we have 10 inputs, 3 outputs, and want two hidden layer of sizes 128 and 64:

    ```python
    model = vb.models.MLP([10, 128, 64, 3])
    ```

    Args:
        channels:
            list of widths of linear layers. First value is number of input channels and last is number of output channels.
        act_cls: activation function class. Defaults to nn.ReLU.
        bn: if True enables batch norm. Defaults to False.
        dropout: dropout probability. Defaults to 0.
        ortho_init: if true ises orthgonal init. defaults to False.
        cls: you can change it from using nn.Linear to some other class with same API. Defaults to nn.Linear.
    """
    def __init__(
        self,
        channels: Iterable[int],
        act_cls: Callable | None = nn.ReLU,
        bn: bool = False,
        dropout: float = 0,
        ortho_init: bool = False,
        cls: Callable = nn.Linear,

        # vis_shape: tuple[int,int] | None = None,
        # max_tiles: int = 100,
    ):
        super().__init__()
        channels = list(channels)

        layers = []

        # if len(channels) = 2, this entire thing is skipped (is empty) so we get only head
        for i,o in zip(channels[:-2], channels[1:-1]):
            layers.append(cls(i, o, not bn))
            if act_cls is not None: layers.append(act_cls())
            if bn: layers.append(nn.BatchNorm1d(o))
            if dropout > 0: layers.append(nn.Dropout1d(dropout))

        self.layers = nn_tools.Sequential(*layers)
        self.head = cls(channels[-2], channels[-1])
        # self.vis_shape = vis_shape
        # self.max_tiles = max_tiles

        if ortho_init:
            generator=torch.Generator().manual_seed(0)
            for p in self.parameters():
                if p.ndim >= 2:
                    torch.nn.init.orthogonal_(p, generator=generator)

    def forward(self, x: torch.Tensor):
        if x.ndim > 2:
            x = x.flatten(1,-1)

        for l in self.layers: x = l(x)
        return self.head(x)

    # def after_get_loss(self, benchmark: "Benchmark"):
    #     _visualize_linear(self.layers[0][0], benchmark, self.vis_shape, self.max_tiles)



class RecurrentMLP(nn.Module):
    """Neural net which passes input through the same layer(s) multiple times.

    Input is passed through in-block defined by ``in_channels``,

    then ``n_passes`` times it goes through hidden block defined by ``hidden_channels``,

    and then it is passed to out-block defined by ``out_channels``.

    Args:
        in_channels (int | Iterable[int] | None):
            Width(s) of in-block. First value is number of input channels.
            If None - no in-block (it goes straight to hidden block).
        hidden_channels (int | Iterable[int]):
            Width(s) of hidden block. If sequence, first value must be equal to last value!!!
        out_channels (int | Iterable[int] | None):
            Width(s) of out-block. Last value is number of output channels.
            If None - no out-block (uses output of last pass of hidden block).
        n_passes (int): number of times input should go through the hidden block.
        act_cls (Callable | None, optional): activation function class. Defaults to nn.ELU.
        dropout (float, optional): dropout probability. Defaults to 0.
        bn (bool, optional): if True enables batch norm. Defaults to True.
        cls (Callable, optional):
            you can change it from using nn.Linear to some other class with same API. Defaults to nn.Linear.
    """
    def __init__(
        self,
        in_channels: int | Iterable[int] | None,
        hidden_channels: int | Iterable[int],
        out_channels: int | Iterable[int] | None,
        n_passes: int,
        act_cls: Callable | None = nn.ReLU,
        dropout: float = 0,
        bn=True,
        ortho_init: bool = False,
        cls: Callable = nn.Linear,
    ):
        super().__init__()
        if isinstance(in_channels, int): in_channels = [in_channels, ]
        if isinstance(out_channels, int): out_channels = [out_channels, ]

        if isinstance(hidden_channels, int): hidden_channels = [hidden_channels, ]
        else: hidden_channels = list(hidden_channels)

        # in-block
        if in_channels is None: self.in_block = nn.Identity()
        else:
            in_channels = list(in_channels) + [hidden_channels[0], ]
            self.in_block = MLP(in_channels, act_cls=act_cls, dropout=dropout, bn=bn, ortho_init=ortho_init, cls=cls)

        # hidden block
        if len(hidden_channels) == 1: hidden_channels = [hidden_channels[0], hidden_channels[0]]
        self.hidden_block = MLP(hidden_channels, act_cls=act_cls, bn=bn, dropout=dropout, ortho_init=ortho_init, cls=cls)

        # out-block
        if out_channels is None: self.in_block = nn.Identity()
        else:
            out_channels = [hidden_channels[-1], ] + list(out_channels)
            self.out_block = MLP(out_channels, act_cls=act_cls, dropout=dropout, bn=bn, ortho_init=ortho_init, cls=cls)

        self.n_passes = n_passes

    def forward(self, x: torch.Tensor):
        if x.ndim > 2:
            x = x.flatten(1,-1)

        x = self.in_block(x)

        for _ in range(self.n_passes):
            x = self.hidden_block(x)

        return self.out_block(x)

class RNN(nn.Module):
    """passes input to RNN (assumes input is single channel sequence) and then takes last timesteps output and passes to a linear layer.

    Args:
        in_channels (int): sequence length
        out_channels (int): output channels.
        hidden_size (int): hidden size of the rnn.
        num_layers (int): number of layers in rnn
        rnn (Callable[..., nn.Module], optional):
            rnn class like nn.RNN, nn.LSTM or nn.GRU or something else. Defaults to nn.LSTM.
        all_layers (bool, optional):
            if True, passes outputs of all layers for last time step,
            if False (default), only passes last layer output

    """
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, rnn: Callable[..., nn.Module]=nn.LSTM, all_layers:bool=False):

        super().__init__()
        self.in_channels = in_channels
        self.rnn = rnn(in_channels, hidden_size, num_layers, batch_first=True)
        self.all_layers = all_layers

        if all_layers: hidden_size = hidden_size * num_layers
        self.fc = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor):
        if x.ndim == 2 and self.in_channels == 1:
            # from (batch_size, 40) to (batch_size, 40, 1) otherwise known as (batch_size, seq_length, in_channels)
            x = x.unsqueeze(2)
        else:
            # from (batch_size, in_channels, seq_length) to (batch_size, seq_length, in_channels)
            x = x.swapaxes(1,2)

        out, hn = self.rnn(x) # out: (batch_size, seq_length, hidden_size)
        if isinstance(hn, tuple): hn = hn[0] # lstm also returns cell states

        if self.all_layers:
            # hidden is (num_layers, batch_size, hidden_size)
            # outpit of all layers for last time step
            x = hn.swapaxes(0,1).flatten(1,-1) # (batch_size, num_layers * hidden_size)

        else:
            # out is (batch_size, seq_len, hidden_size)
            # output of last layer for all time steps
            x = out[:, -1, :] # last timestep's output (batch_size, hidden_size)

        return self.fc(x) # (batch_size, num_classes)


def _make_colors(n, seed):
    # generate colors for boxes
    colors = []
    i = 2
    while len(colors) < i:
        colors = list(itertools.product(np.linspace(0, 255, n).tolist(), repeat=3)) # type:ignore
        if (255., 255., 255.) in colors: colors.remove((255., 255., 255.))
        i+=1

    rng = random.Random(seed)
    rgbs = rng.sample(colors, k = n)

    # remove almost black colors
    for i in range(len(rgbs)): # pylint:disable=consider-using-enumerate
        while sum(rgbs[i]) < 150: # type:ignore
            rgbs[i] = rng.sample(colors, k = 1)[0]

    return rgbs

@torch.no_grad
def _scatter_2d_latents_(
    benchmark: "DatasetBenchmark | Any",
    encoder: nn.Module,
    resolution: tuple[int, int],
    colors,
    prep_fn=None,
):
    device = next(iter(encoder.parameters())).device

    latents_list = []
    labels_list = []

    if hasattr(benchmark, "data_tensors") and benchmark.data_tensors is not None:
        dl = (benchmark.data_tensors, )
    else:
        assert benchmark._dltrain is not None
        dl = benchmark._dltrain

    for batch in dl:
        if isinstance(batch, torch.Tensor):
            x = batch
        else:
            if len(batch) == 1:
                x = batch[0]
            else:
                x, y = batch
                labels_list.append(y)

        if prep_fn is not None:
            x = prep_fn(x)

        latent = encoder(x.to(device)).squeeze()
        assert latent.shape[-1] == 2
        while latent.ndim < 2: latent = latent.unsqueeze(0)
        latents_list.append(latent)

    latents = torch.cat(latents_list, 0).nan_to_num(0,1,-1) # (B, 2)
    latents -= latents.amin(0, keepdim=True)
    latents /= latents.amax(0, keepdim=True).clip(min=1e-10)
    latents[:, 0] = (latents[: , 0] * resolution[0]).clip(max=resolution[0] - 1)
    latents[:, 1] = (latents[: , 1] * resolution[1]).clip(max=resolution[1] - 1)
    latents = latents.long()


    if len(labels_list) == 0:
        frame = torch.zeros(resolution, device=latents.device, dtype=torch.uint8)
        frame[latents[:, 0], latents[:, 1]] = 255

    else:
        labels = torch.stack(labels_list).to(device=device).squeeze()
        frame = torch.zeros((3, *resolution), device=latents.device, dtype=torch.uint8)
        if labels.is_floating_point():
            labels = ((labels - labels.amin()) * (255 / labels.amax())).clip(0, 255).to(torch.uint8)
            frame[0, latents[:, 0], latents[:, 1]] = labels # red
            frame[2, latents[:, 0], latents[:, 1]] = 255-labels # blue
        else:
            if colors is None:
                unique = len(torch.unique(labels, sorted=False))
                colors = torch.tensor(_make_colors(unique, 0)).to(device=frame.device, dtype=torch.uint8)
            points = colors[labels]
            frame[0, latents[:, 0], latents[:, 1]] = points[:, 0]
            frame[1, latents[:, 0], latents[:, 1]] = points[:, 1]
            frame[2, latents[:, 0], latents[:, 1]] = points[:, 2]

    benchmark.log_image("latents", frame, to_uint8=False)
    return colors

class Autoencoder(nn.Module):
    """Can visualize latents if they are 2d"""
    def __init__(self, encoder: nn.Module, decoder: nn.Module, resolution: tuple[int, int] | None = (256, 256)):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.is_2d = False
        self.resolution = resolution # visualize_latents
        self.colors = None

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        self.is_2d = (x.ndim <= 2) and (x.size(-1) == 2)
        x = self.decoder(x)
        return x

    @torch.no_grad
    def after_get_loss(self, benchmark: "DatasetBenchmark"):
        if not (benchmark.training and benchmark.make_images):
            return

        if self.is_2d and self.resolution is not None:
            self.colors = _scatter_2d_latents_(benchmark, self.encoder, self.resolution, self.colors)
