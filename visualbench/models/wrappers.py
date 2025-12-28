from collections.abc import Sequence
from functools import partial
import torch

from ..utils import totensor
from ..rng import RNG

def _get_lr(opt):
    return next(iter(opt.param_groups))["lr"]

def _decay_lr_(opt):
    for p in opt.param_groups:
        p["lr"] *= 0.99

class ConstantInput(torch.nn.Module):
    """wraps another model and passes it constant input"""
    @torch.no_grad
    def __init__(self, model: torch.nn.Module, input: int | Sequence[int] | torch.Tensor, learnable: bool = False,  noise:float=0, seed: int | None = 0):
        super().__init__()

        # generate input
        device = next(iter(model.parameters())).device
        self.rng = RNG(seed)
        if not isinstance(input, torch.Tensor): input = torch.randn(input, device=device, generator=self.rng.torch(device))
        input = input.to(device)

        if learnable:
            self.input = torch.nn.Parameter(input.requires_grad_(True))
        else:
            self.input = torch.nn.Buffer(input.requires_grad_(False))

        self.input_mad = torch.nn.Buffer((input - input.mean()).abs().mean())

        self.model = model
        self.noise = noise

    def forward(self):
        input = self.input
        if self.noise != 0:
            noise = torch.randn(
                input.size(), device=input.device, dtype=input.dtype, generator=self.rng.torch(input.device))
            input = input + noise * self.noise * self.input_mad
        return self.model(input)


class RandomInput(torch.nn.Module):
    """wraps another model and passes it random input"""
    @torch.no_grad
    def __init__(self, model: torch.nn.Module, input_size: int | Sequence[int], seed: int | None =0):
        super().__init__()

        self.model = model
        self.input_size = input_size
        self.rng = RNG(seed)

    def forward(self):
        p = next(iter(self.model.parameters()))
        input = torch.randn(self.input_size, device=p.device, dtype=p.dtype, generator=self.rng.torch(p.device))
        return self.model(input)
