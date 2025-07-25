from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..benchmark import Benchmark


class RNNArgsort(Benchmark):
    """Very fast RNN training on an argsort objective.

    Doesn't support rendering.
    """
    def __init__(self, seq_len=10, hidden_size=32, batch_size=128, num_layers=1, rnn_cls: Callable[..., nn.Module] = nn.LSTM):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, seq_len)

    def get_loss(self):
        inputs = torch.rand(self.batch_size, self.seq_len, device=self.device)
        targets = torch.argsort(inputs, dim=1)
        rnn_out, _ = self.rnn(inputs.unsqueeze(-1))
        scores = self.fc(rnn_out)  # (batch_size, seq_len, seq_len)
        loss = F.cross_entropy(scores.transpose(1, 2), targets)
        return loss

    def reset(self):
        super().reset()
        if hasattr(self, 'rnn'): self.rnn.flatten_parameters() # type:ignore
        return self