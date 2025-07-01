import torch
from torch import nn
from torch.nn import functional as F


class TinyWideConvNet(nn.Module):
    """3,626 params"""
    def __init__(self, act_cls = nn.ELU, dropout=0.5):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5), # ~37
            nn.MaxPool1d(2), # ~18
            act_cls(),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5), # ~15
            nn.MaxPool1d(2), # ~7
            act_cls(),
            nn.Dropout1d(dropout),
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5),
            nn.Dropout1d(dropout),
        )

        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x).mean(-1)
        return self.linear(x)


class TinyLongConvNet(nn.Module):
    """1,338 params"""
    def __init__(self, act_cls = nn.ELU, dropout=0.0):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=2),
            act_cls(),
            nn.BatchNorm1d(4, track_running_stats=False),

            nn.Conv1d(4, 4, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(4, track_running_stats=False),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=2),
            act_cls(),
            nn.BatchNorm1d(8, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv1d(8, 8, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(8, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=2),
            act_cls(),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv1d(16, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(16, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x).mean(-1)
        return self.linear(x)


class ConvNet(nn.Module):
    """134,410 params"""
    def __init__(self, act_cls = nn.ELU, dropout=0.2):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.MaxPool1d(2),
            act_cls(),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.MaxPool1d(2),
            act_cls(),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            act_cls(),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.MaxPool1d(2),
            act_cls(),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )


        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(384, 10))

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.linear(x)




class FastConvNet(nn.Module):
    """134,410 params"""
    def __init__(self, act_cls = nn.ELU, dropout=0.2):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(96, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2, stride=2),
            act_cls(),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )


        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(640, 10))

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return self.linear(x)



class MobileNet(nn.Module):
    """6,228 params"""
    def __init__(self, act_cls = nn.ReLU, dropout=0.5):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ChannelShuffle(8),
            act_cls(),
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, groups=32, padding=1),
            act_cls(),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv1d(32, 64, kernel_size=1, groups=8, padding=1, ),
            nn.ChannelShuffle(16),
            act_cls(),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, groups=64, ),
            act_cls(),
            nn.Dropout1d(dropout) if dropout > 0 else nn.Identity(),
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(128, 10, kernel_size=1),
        )


        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(384, 10))

    def forward(self, x):
        if x.ndim == 2: x = x.unsqueeze(1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x.mean(-1)
