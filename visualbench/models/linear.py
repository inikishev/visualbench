import torch
from torch import nn

class Rank1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(in_channels))
        self.w_out = nn.Parameter(torch.randn(out_channels))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

    def forward(self, x: torch.Tensor):
        x = (x * self.w_in).mean(-1, keepdim=True)
        x = x * self.w_out
        if self.b is not None: x = x + self.b
        return x

class Rank1PlusIdentity(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, learnable_scale:bool=False):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(in_channels))
        self.w_out = nn.Parameter(torch.randn(out_channels))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(1.))
        else:
            self.scale = None

    def forward(self, x: torch.Tensor):
        x_out = (x * self.w_in).mean(-1, keepdim=True)
        x_out = x_out * self.w_out

        if self.b is not None: x_out = x_out + self.b
        if self.scale is not None: x = x * self.scale

        if x.size(-1) < x_out.size(-1):
            x = torch.nn.functional.pad(x, (x_out.size(-1) - x.size(-1), 0))
        elif x_out.size(-1) < x.size(-1):
            x = x[..., :x_out.size(-1)]

        return x + x_out

class Diagonal(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.diag = nn.Parameter(torch.randn(min(in_channels, out_channels)))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

    def forward(self, x: torch.Tensor):
        x = x[..., :self.diag.size(-1)] * self.diag[..., :x.size(-1)]
        if self.diag.size(-1) > x.size(-1):
            x = torch.nn.functional.pad(x, (self.diag.size(-1) - x.size(-1), 0))

        if self.b is not None: x = x + self.b
        return x

def _maybe_call(fn, x):
    if fn is None: return x
    return fn(x)

class Rank1PlusDiagonal(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, rank1_act=None, diag_act=None, mid_act=None):
        super().__init__()
        self.w_in = nn.Parameter(torch.randn(in_channels))
        self.w_out = nn.Parameter(torch.randn(out_channels))
        self.diag = nn.Parameter(torch.ones(min(in_channels, out_channels), dtype=torch.float32))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

        self.mid_act = mid_act
        self.rank1_act = rank1_act
        self.diag_act = diag_act

    def forward(self, x: torch.Tensor):
        x_rank1 = (x * self.w_in).mean(-1, keepdim=True)
        x_rank1 = _maybe_call(self.mid_act, x_rank1) * self.w_out

        x_diag = x[..., :self.diag.size(-1)] * self.diag[..., :x.size(-1)]
        if x_rank1.size(-1) > x_diag.size(-1):
            x_diag = torch.nn.functional.pad(x_diag, (x_rank1.size(-1) - x_diag.size(-1), 0))

        x_out = _maybe_call(self.rank1_act, x_rank1) + _maybe_call(self.diag_act, x_diag)
        if self.b is not None: x_out = x_out + self.b

        return x_out