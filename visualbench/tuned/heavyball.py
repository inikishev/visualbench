from functools import partial

import torch
import torchzero as tz


def HeavyBall_tuned1(params, lr):
    """
    Bench: InverseInverse(SANIC96)

    Space: normal

    Hps: `{'lr': 0.36187769677450954, 'momentum': 0.9864351645, 'dampening': 0.6187006388894171}`

    Loss: `0.6734351515769958`
    """
    return torch.optim.SGD(params, lr, momentum=0.9864351645, dampening=0.6187006388894171)

def HeavyBall_tuned2(params, lr):
    """
    Bench: InverseInverse(SANIC96)

    Space: full (-10, 10)

    Hps: `{'lr': 0.014748564858988513, 'momentum': 0.9822590358913175, 'dampening': -7.653852004274594}`

    Loss: `0.7042162418365479`
    """
    return tz.Modular(params, tz.m.HeavyBall(momentum=0.9822590358913175, dampening=-7.653852004274594), tz.m.LR(lr))
