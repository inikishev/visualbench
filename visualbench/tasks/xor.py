from typing import Any

import torch
from torch import nn

from ..benchmark import Benchmark
from ..utils import CUDA_IF_AVAILABLE


def generate_xor_data(num_samples=1000, device: Any = CUDA_IF_AVAILABLE):
    X = torch.randint(0, 2, (num_samples, 2), device = device)
    y = (X[:, 0] ^ X[:, 1]).float().unsqueeze(1)
    return X.float(), y


class XOR_LSTM(nn.Module):
    def __init__(self, input_size = 2, hidden_size = 10, output_size = 1, num_layers = 1, dropout=0.):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :]) # Take output from the last time step
        return out

class XOR(Benchmark):
    """ultra fast LSTM XOR benchmark

    Args:
        hidden_size (int, optional): LSTM hidden size. Defaults to 16.
        num_layers (int, optional): LSTM number of layers. Defaults to 1.
        num_samples (int, optional): number of XOR samples. Defaults to 128.
        dropout (_type_, optional): LSTM dropout. Defaults to 0.
        batched (bool, optional): if False, pregenerates data, otherwise generates random data each time. Defaults to False.
        criterion (_type_, optional): loss function. Defaults to torch.nn.functional.binary_cross_entropy_with_logits.
    """
    x_data: torch.nn.Buffer | None
    y_data: torch.nn.Buffer | None
    def __init__(
        self,
        hidden_size = 16,
        num_layers = 1,
        num_samples = 128,
        dropout = 0.,
        batched = True,
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
    ):

        super().__init__()
        self.lstm = XOR_LSTM(2, hidden_size=hidden_size, num_layers = num_layers, dropout=dropout)
        self.lstm.train()
        self.num_samples = num_samples
        self.batched = batched
        self.criterion = criterion

        if self.batched:
            self.x_data = None
            self.y_data = None
        else:
            x, y = generate_xor_data(num_samples, 'cpu')
            self.register_buffer('x_data', x)
            self.register_buffer('y_data', y)

    def get_loss(self):
        if self.x_data is None or self.y_data is None:
            x, y = generate_xor_data(self.num_samples, self._first_param_device())

        else:
            x,y = self.x_data, self.y_data

        outputs = self.lstm(x.unsqueeze(1))
        loss = self.criterion(outputs, y)
        return loss


def generate_delayed_xor_data(num_samples=128, seq_len=16, device: Any = CUDA_IF_AVAILABLE):
    """Taken from https://github.com/ClashLuke/HeavyBall/blob/main/benchmark/xor_spot.py"""
    b = num_samples
    l = seq_len
    d = torch.float32

    inp = torch.randn((b, l, 1), device=device, dtype=d)
    inp = inp > 0
    zeros = torch.zeros_like(inp)
    zeros[:, torch.randint(0, l, (b,), device=device)] = 1
    zeros[:, torch.randint(0, l, (b,), device=device)] = 1
    target = (inp * zeros).sum(1) % 2
    return torch.stack((inp, zeros + 2), 0).to(d), target.to(d)


class DelayedXOR_LSTM(nn.Module):
    """Taken from https://github.com/ClashLuke/HeavyBall/blob/main/benchmark/xor_spot.py"""
    def __init__(self, size, depth,dropout=0.):
        super().__init__()
        self.embed = nn.Embedding(4, size)
        self.enc = nn.LSTM(size, size, depth,dropout=dropout, batch_first=False)
        self.enc.flatten_parameters()
        self.proj = nn.Sequential(nn.LayerNorm(size),  #
                                  nn.Linear(size, 1))

    def forward(self, inp):
        inp = self.embed(inp.squeeze(-1).long())
        inp = inp[0] + inp[1]
        out, _ = self.enc(inp.transpose(0, 1))
        return self.proj(out[-1, :])


class DelayedXOR(Benchmark):
    """ultra fast delayed XOR from https://github.com/ClashLuke/HeavyBall/blob/main/benchmark/xor_spot.py

    Args:
        hidden_size (int, optional): LSTM hidden size. Defaults to 16.
        num_layers (int, optional): LSTM number of layers. Defaults to 1.
        num_samples (int, optional): number of XOR samples. Defaults to 128.
        dropout (_type_, optional): LSTM dropout. Defaults to 0.
        batched (bool, optional): if False, pregenerates data, otherwise generates random data each time. Defaults to False.
        criterion (_type_, optional): loss function. Defaults to torch.nn.functional.binary_cross_entropy_with_logits.
    """
    x_data: torch.nn.Buffer | None
    y_data: torch.nn.Buffer | None
    def __init__(
        self,
        seq_length = 16,
        hidden_size = 16,
        num_layers = 1,
        num_samples = 128,
        dropout = 0.,
        batched = True,
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
    ):
        super().__init__()
        self.lstm = DelayedXOR_LSTM(hidden_size, num_layers, dropout)
        self.lstm.train()
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.batched = batched
        self.criterion = criterion

        if self.batched:
            self.x_data = None
            self.y_data = None
        else:
            x, y = generate_delayed_xor_data(num_samples, seq_length, device = 'cpu')
            self.register_buffer('x_data', x)
            self.register_buffer('y_data', y)

    def get_loss(self):
        if self.x_data is None or self.y_data is None:
            x, y = generate_delayed_xor_data(self.num_samples, self.seq_length, device = self._first_param_device())

        else:
            x,y = self.x_data, self.y_data

        outputs = self.lstm(x)
        loss = self.criterion(outputs, y)
        return loss
