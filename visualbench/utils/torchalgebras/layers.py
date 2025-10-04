import torch

from .base import Algebra, AlgebraicTensor


class AlgebraicLinear(torch.nn.Module):
    def __init__(self, algebra: Algebra, in_channels, out_channels, bias=True, dtype = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, dtype=dtype))

        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels, dtype=dtype))
        else:
            self.bias = None

        self.algebra = algebra

    def forward(self, x: torch.Tensor):
        r = AlgebraicTensor(self.weight, self.algebra) @ x.unsqueeze(-1)
        r.data = r.data.squeeze(-1)

        if self.bias is not None:
            r = r + self.bias

        return r.data

