import csv
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import DatasetBenchmark
from ...utils import CUDA_IF_AVAILABLE

# function from PSGD https://github.com/lixilinx/psgd_torch/blob/master/lstm_with_xor_problem.py
def generate_train_data(batch_size=128, seq_len=16, dim_in=2, dim_out=1, seed: int | None=0):
    generator = np.random.default_rng(seed)
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i,:,0] = generator.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(generator.uniform(0,1)*0.1*seq_len))
        i2 = int(np.floor(generator.uniform(0.1)*0.4*seq_len + 0.1*seq_len))
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i,i1,0] == x[i,i2,0]: # XOR
            y[i] = -1.0 # lable 0
        else:
            y[i] = 1.0  # lable 1

    #tranpose x to format (sequence_length, batch_size, dimension_of_input)
    X, y = torch.tensor(np.transpose(x, [1, 0, 2])), torch.tensor(y)

    #tranpose x to format (batch_size, dimension_of_input, sequence_length)
    X = X.permute(1, 2, 0)
    return X, (y + 1) / 2


class XOR(DatasetBenchmark):
    def __init__(
        self,
        model,
        batch_size:int | None,
        test_batch_size:int | None,
        num_samples:int = 10_000,
        seq_len:int=16,
        dim_in:int=2,
        dim_out=1,
        criterion=F.binary_cross_entropy_with_logits,
        train_split: int | float = 0.8,
        device = CUDA_IF_AVAILABLE
    ):
        X, y = generate_train_data(batch_size=num_samples, seq_len=seq_len, dim_in=dim_in, dim_out=dim_out)
        super().__init__(
            data_train=(X, y),
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_split = train_split,
            shuffle_split = True,
            data_device=device,
        )