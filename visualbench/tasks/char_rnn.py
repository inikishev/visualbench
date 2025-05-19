import os
from collections.abc import Callable
import torch
from torch import nn

from ..benchmark import Benchmark


class _DefaultRNN(nn.Module):
    def __init__(self, length: int, features: int, layers=1, dropout=0., rnn_cls: Callable = nn.LSTM):
        super().__init__()
        self.emb = nn.Embedding(length, features)
        self.rnn = rnn_cls(features, features, layers, dropout=dropout,batch_first=True)
        self.head = nn.Linear(features, length)

    def forward(self, x):
        x = self.emb(x)
        x = self.rnn(x)[0]
        return self.head(x)


class CharLSTM(Benchmark):
    def __init__(
        self,
        length = 256,
        features = 512,
        batch_size = 16,
        layers = 1,
        dropout = 0.,
        rnn_cls: Callable = nn.LSTM,
        criterion = torch.nn.functional.cross_entropy,
    ):
        super().__init__()
        self.criterion = criterion
        self.lstm = _DefaultRNN(length=length, features=features, layers=layers, dropout=dropout, rnn_cls=rnn_cls)

        with open(os.path.join(os.path.dirname(__file__), 'shakespeare.txt'), 'rb') as f:
            text = f.read()

        chars = torch.frombuffer(text, dtype=torch.uint8).clone().long()
        self.chars = torch.nn.Buffer(chars[(length + 1) * batch_size:])
        self.offsets = torch.nn.Buffer(torch.arange(0, length + 1).repeat(batch_size, 1))

        def dataloader():
            # fast loader stolen from https://github.com/HomebrewML/HeavyBall/blob/main/benchmark/char_rnn.py
            batch_offsets = torch.randint(0, len(self.chars) - length - 1, (batch_size,), device=self.offsets.device)
            batch_offsets = batch_offsets[:, None] + self.offsets
            batch_chars = self.chars[batch_offsets]
            batch_chars = batch_chars.view(batch_size, length + 1)
            src = batch_chars[:, :-1]
            tgt = batch_chars[:, 1:]
            return src, tgt

        self.dataloader = dataloader


    def get_loss(self):
        x,y = self.dataloader()

        outputs = self.lstm(x)
        loss = self.criterion(outputs, y)
        return loss


# bbb = CharLSTM(128, 16, 1).cuda()
# bbb.run(torch.optim.Adam(bbb.parameters(), 1e-2), 1000)