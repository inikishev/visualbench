from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class ScalarAffine(nn.Module):
    """computes ``w * x + b`` where ``w`` and ``b`` are learnable scalars.

    if ``enable=False`` then this is a no-op and does nothing."""
    def __init__(self, enable: bool = True):
        super().__init__()
        if enable:
            self.w = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True), requires_grad=True)
            self.b = nn.Parameter(torch.tensor(0., dtype=torch.float32, requires_grad=True), requires_grad=True)
        else:
            self.w = self.b = None

    def forward(self, x: torch.Tensor):
        if self.w is not None:
            assert self.b is not None
            return x * self.w + self.b
        return x

class Sine(nn.Module):
    """computes ``sine(x)``"""
    def forward(self, x: torch.Tensor): return x.sin()

class SineApprox(nn.Module):
    """computes ``sine(x)``"""
    def forward(self, x: torch.Tensor): return -x * x.abs() + x

class Abs(nn.Module):
    """computes ``abs(x)``"""
    def forward(self, x: torch.Tensor): return x.abs()

class Square(nn.Module):
    """computes ``x^2`` and optionally copies sign of x"""
    def __init__(self, copysign:bool=False):
        super().__init__()
        self.copysign = copysign
    def forward(self, x: torch.Tensor):
        if self.copysign: return x.square().copysign(x)
        return x.square()

class Exp(nn.Module):
    """computes ``exp(x)`` and optionally copies sign of x"""
    def __init__(self, copysign:bool=False):
        super().__init__()
        self.copysign = copysign
    def forward(self, x: torch.Tensor):
        if self.copysign: return x.exp().copysign(x)
        return x.exp()

class Sqrt(nn.Module):
    """computes ``sqrt(abs(x))`` and optionally copies sign of x"""
    def __init__(self, copysign:bool=False):
        super().__init__()
        self.copysign = copysign
    def forward(self, x: torch.Tensor):
        if self.copysign: return x.abs().sqrt().copysign(x)
        return x.abs().sqrt()

class Hyperexp(nn.Module):
    """computes ``x^x`` and optionally copies sign of x"""
    def __init__(self, copysign:bool=False):
        super().__init__()
        self.copysign = copysign
    def forward(self, x: torch.Tensor):
        x_abs = x.abs()
        if self.copysign: return (x_abs**x_abs).copysign(x)
        return x_abs**x_abs

class HyperexpSoftplus(nn.Module):
    """computes ``softplus(x)^softplus(x)``"""
    def forward(self, x: torch.Tensor):
        x = F.softplus(x) # pylint:disable=not-callable
        return x**x

class LearnablePower(nn.Module):
    """computes ``abs(x)^power`` where ``power`` is learnable."""
    def __init__(self, init=1.0, copysign:bool=False):
        super().__init__()
        self.power = nn.Parameter(torch.tensor(init, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.copysign = copysign

    def forward(self, x: torch.Tensor):
        if self.copysign: x.abs().pow(self.power).copysign(x)
        return x.abs().pow(self.power)

class PowersDifference(nn.Module):
    def __init__(self, weighted:bool=False):
        super().__init__()
        self.p1 = nn.Parameter(torch.tensor(2., dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.p2 = nn.Parameter(torch.tensor(1.5, dtype=torch.float32, requires_grad=True), requires_grad=True)

        self.a1 = ScalarAffine(weighted)
        self.a2 = ScalarAffine(weighted)

    def forward(self, x: torch.Tensor):
        return self.a1(x).abs() ** self.p1 - self.a2(x).abs() ** self.p2


class WeightedHyperexp(nn.Module):
    def __init__(self,):
        super().__init__()
        self.a1 = ScalarAffine()
        self.a2 = ScalarAffine()

    def forward(self, x: torch.Tensor):
        return self.a1(x).abs() ** self.a2(x).abs()

class DivAdd(nn.Module):
    def __init__(self, denom=1.0, min=1e-6, learnable:bool=True, weighted:bool=False):
        super().__init__()
        self.denom = nn.Parameter(torch.tensor(denom, dtype=torch.float32, requires_grad=learnable), requires_grad=learnable)
        self.min = min

        self.na = ScalarAffine(weighted)
        self.da = ScalarAffine(weighted)

    def forward(self, x: torch.Tensor):
        return self.na(x) / (self.da(x).abs() + self.denom.pow(2).clip(min=self.min))

class Spike(nn.Module):
    def __init__(self, min=1e-6, weighted:bool=False):
        super().__init__()
        self.num = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.denom = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.min = min

        self.na = ScalarAffine(weighted)
        self.da = ScalarAffine(weighted)


    def forward(self, x: torch.Tensor):
        return (self.na(x).abs() + self.num) / (self.da(x).abs() + self.denom.pow(2).clip(min=self.min))

class Gaussian(nn.Module):
    """computes ``exp(-x^2)``"""
    def __init__(self, ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.square().neg().exp()

class ActNet(nn.Module):
    """repeatedly applies activation function with learnable per-step weights!

    ```
    x1 = act1(x0)
    x2 = act2(w1 * x1)
    x3 = act3(w2 * x2)
    etc
    ```
    """
    def __init__(self, depth:int, act_cls: Callable = nn.ELU):
        super().__init__()
        self.depth = depth
        self.weights = nn.Parameter(torch.ones(depth-1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.biases = nn.Parameter(torch.zeros(depth-1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.acts = nn.ModuleList(act_cls() for _ in range(depth))

    def forward(self, x:torch.Tensor):
        x = self.acts[0](x)
        for w,b,a in zip(self.weights, self.biases, self.acts[1:]):
            x = a(w*x+b)
        return x

class Lambda(nn.Module):
    """computes ``exp(-x^2)``"""
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor):
        return self.fn(x)