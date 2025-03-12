from collections.abc import Callable

from sklearn.datasets import fetch_california_housing, make_moons
from torch import nn
from torch.nn import functional as F

from .dataset import DatasetBenchmark
from .models import MLP, ModelClass


class CaliforniaHousing(DatasetBenchmark):
    def __init__(
        self,
        model: ModelClass = nn.Linear,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        test_split=0.8,
        normalize_x=True,
        normalize_y=True,
    ):
        x,y = fetch_california_housing(return_X_y=True)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model(8, 1),
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            test_split=test_split,
            shuffle_split=True,
            normalize=(normalize_x, normalize_y),
        )


class Moons(DatasetBenchmark):
    def __init__(
        self,
        model: ModelClass = MLP([8, 16, 24, 32]),
        criterion=F.binary_cross_entropy_with_logits,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        test_split=None,
        shuffle_split=True,
        normalize_x=True,
    ):
        x,y = make_moons(n_samples = 1024, noise = 0.2, random_state=0)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model(2, 1),
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            test_split=test_split,
            shuffle_split=shuffle_split,
            normalize=(normalize_x, False),
            decision_boundary=True,
            boundary_act=F.sigmoid,
        )
