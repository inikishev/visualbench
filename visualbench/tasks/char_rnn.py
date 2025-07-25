import os
import textwrap
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from ..benchmark import Benchmark

def _text_to_image(text, size):
    image = Image.new("RGB", (size,size), (255,255,255))

    d_usr = ImageDraw.Draw(image)
    d_usr.fontmode = "1"
    d_usr.text((0,0), textwrap.fill(text),(0,0,0))

    return np.asarray(image)


class _RNN(nn.Module):
    def __init__(self, features: int, layers=1, dropout=0., rnn_cls: Callable = nn.LSTM, vocab_size: int=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, features)
        self.rnn = rnn_cls(features, features, layers, dropout=dropout,batch_first=True)
        self.head = nn.Linear(features, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.rnn(x)[0]
        return self.head(x)

class _Tinyformer(nn.Module):
    def __init__(self, length, features=64, heads=4, layers=2, vocab_size=256, ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, features)
        self.pos = nn.Parameter(torch.zeros(1, length, features))
        encoder_layer = nn.TransformerEncoderLayer(d_model=features, nhead=heads, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(features, vocab_size)

    def forward(self, x):
        x = self.emb(x) + self.pos[:, :x.size(1), :]
        x = self.enc(x)
        return self.head(x)


class CharRNN(Benchmark):
    """Character-level RNN, learns to predict text.

    Renders:
        predicted text if you pass in ``test_text``"""
    def __init__(
        self,
        length = 256,
        features = 512,
        batch_size = 16,
        layers = 1,
        heads = 4, # transformer only
        dropout = 0.,
        rnn_cls: Callable | Literal['transformer'] = nn.LSTM, # rnn only
        criterion = torch.nn.functional.cross_entropy,
        test_text: str | None = None,
        resolution = 512,
    ):
        super().__init__()
        self.criterion = criterion
        if isinstance(rnn_cls, str):
            self.net = _Tinyformer(features=features, layers=layers, heads=heads, length=length)
        else:
            self.net = _RNN(features=features, layers=layers, dropout=dropout, rnn_cls=rnn_cls)

        with open(os.path.join(os.path.dirname(__file__), 'shakespeare.txt'), 'rb') as f:
            text = f.read()

        chars = torch.frombuffer(text, dtype=torch.uint8).clone().long()
        self.chars = torch.nn.Buffer(chars[(length + 1) * batch_size:])
        self.offsets = torch.nn.Buffer(torch.arange(0, length + 1).repeat(batch_size, 1))

        # fast loader stolen from https://github.com/HomebrewML/HeavyBall/blob/main/benchmark/char_rnn.py
        def dataloader():
            batch_offsets = torch.randint(0, len(self.chars) - length - 1, (batch_size,), device=self.offsets.device, generator=self.rng.torch(device=self.device))
            batch_offsets = batch_offsets[:, None] + self.offsets
            batch_chars = self.chars[batch_offsets]
            batch_chars = batch_chars.view(batch_size, length + 1)
            src = batch_chars[:, :-1]
            tgt = batch_chars[:, 1:]
            return src, tgt

        self.dataloader = dataloader

        # test text
        if test_text is None:
            self.test_text = self.test_ints = None

        else:
            self.test_text = test_text[:length]
            self.test_ints = nn.Buffer(torch.tensor([ord(c) for c in self.test_text], dtype=torch.int64).unsqueeze(0))

        self.history = []
        self.resolution = resolution


    def get_loss(self):
        x,y = self.dataloader()

        outputs: torch.Tensor = self.net(x)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), y.ravel())

        if self._make_images and self.test_ints is not None:
            with torch.no_grad():
                outputs = self.net(self.test_ints.to(x.device))[0]
                text_ints = outputs.argmax(-1).cpu().numpy().astype(np.uint8)
                text_bytes = bytes(text_ints)
                text = text_bytes.decode('utf-8', errors='replace')
                self.history.append(text)
        return loss

    def render(self, file: str, fps: int = 60, scale: int | float = 1, progress=True):
        if self.test_text is None: raise RuntimeError("no text")

        from ..utils._benchmark_video import OpenCVRenderer, _maybe_progress
        with OpenCVRenderer(file, fps, scale=scale) as renderer:

            for text in _maybe_progress(self.history, enable=progress):
                renderer.write(_text_to_image(self.test_text + text, self.resolution))
