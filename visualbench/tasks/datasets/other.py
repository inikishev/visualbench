import csv
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import DatasetBenchmark


class WDBC(DatasetBenchmark):
    """You need wdbc.data https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.

    binary classification, 569 samples.

    input - ``(B, 30)``

    output - ``(B, 1)``
    """

    def __init__(
        self,
        wdbc_path: str,
        model,
        criterion=F.binary_cross_entropy_with_logits,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=400,
    ):
        # load the dataset
        with open(wdbc_path, 'r', encoding='utf8') as f:
            data = list(csv.reader(f))

        # it has 569 samples, each sample has 30 features and one binary target
        X = torch.tensor([list(map(float, row[2:])) for row in data], dtype=torch.float32)
        y = torch.tensor([0 if row[1] == 'B' else 1 for row in data], dtype=torch.float32)[:, None]

        # normalize X to have mean=0 and std=1
        X -= X.mean(0)
        X /= X.std(0)

        super().__init__(
            data_train=(X, y),
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_split=train_split,
        )