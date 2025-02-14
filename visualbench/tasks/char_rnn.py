"""https://github.com/HomebrewML/HeavyBall/blob/main/benchmark/char_rnn.py"""
from torch import nn
import torch
from visualbench.benchmark import Benchmark
import os
class Take0(nn.Module):
    def forward(self, x):
        return x[0]


class _Char_LSTM(nn.Module):
    def __init__(self, features: int, sequence: int, layers=1,dropout=0.):
        super().__init__()
        self.sequence = sequence
        self.net = nn.Sequential(
            nn.Embedding(256, features),
            nn.LSTM(features, features, layers, dropout=dropout,batch_first=True),  # Removed dropout since num_layers=1
            Take0(),
            nn.Linear(features, 256)
        )

    def forward(self, inp):
        return self.net(inp)


class CharLSTM(Benchmark):
    """i just took it from https://github.com/HomebrewML/HeavyBall/blob/main/benchmark/char_rnn.py

    Args:
        hidden_size (int, optional): LSTM hidden size. Defaults to 16.
        num_layers (int, optional): LSTM number of layers. Defaults to 1.
        num_samples (int, optional): number of XOR samples. Defaults to 128.
        dropout (_type_, optional): LSTM dropout. Defaults to 0.
        batched (bool, optional): if False, pregenerates data, otherwise generates random data each time. Defaults to False.
        criterion (_type_, optional): loss function. Defaults to torch.nn.functional.binary_cross_entropy_with_logits.
    """
    def __init__(
        self,
        features = 512,
        batch_size = 16,

        num_layers = 1,
        dropout = 0.,
        criterion = torch.nn.functional.cross_entropy
    ):
        super().__init__()
        sequence = 256
        batch = batch_size
        self.criterion = criterion

        self.lstm = _Char_LSTM(features, sequence, num_layers, dropout)

        with open(os.path.join(os.path.dirname(__file__), 'shakespeare.txt'), 'rb') as f:
            text = f.read()

        chars = torch.frombuffer(text, dtype=torch.uint8).clone().long()
        # holdout = chars[:(sequence + 1) * batch].view(batch, sequence + 1)
        self.chars = torch.nn.Buffer(chars[(sequence + 1) * batch:])
        self.offsets = torch.nn.Buffer(torch.arange(0, sequence + 1).repeat(batch, 1))

        def data():
            batch_offsets = torch.randint(0, len(self.chars) - sequence - 1, (batch,), device=self.offsets.device)
            batch_offsets = batch_offsets[:, None] + self.offsets
            batch_chars = self.chars[batch_offsets]
            batch_chars = batch_chars.view(batch, sequence + 1)
            src = batch_chars[:, :-1]
            tgt = batch_chars[:, 1:]
            return src, tgt

        self.data = data


    def get_loss(self):
        x,y = self.data()

        outputs = self.lstm(x)
        loss = self.criterion(outputs, y)
        return loss


# bbb = CharLSTM(128, 16, 1).cuda()
# bbb.run(torch.optim.Adam(bbb.parameters(), 1e-2), 1000)