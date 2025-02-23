import torch
import torch.nn as nn
import torch.nn.functional as F
from ..benchmark import Benchmark

class LSTMArgsort(Benchmark):
    def __init__(self, seq_len=10, hidden_size=32, batch_size=128, num_layers=1):
        super().__init__(log_projections=True)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
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
        lstm_input = inputs.unsqueeze(-1)  # Add feature dimension
        lstm_out, _ = self.lstm(lstm_input)
        scores = self.fc(lstm_out)  # (batch_size, seq_len, seq_len)
        loss = F.cross_entropy(scores.transpose(1, 2), targets)
        return loss

    def reset(self):
        super().reset()
        self.lstm.flatten_parameters()