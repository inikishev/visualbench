from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dataset import DatasetBenchmark


class CaliforniaHousing(DatasetBenchmark):
    """
    regression

    input - ``(B, 8)``

    output - ``(B, 1)``
    """
    def __init__(
        self,
        model,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
        normalize_y=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_x, normalize_y),
        )


class Moons(DatasetBenchmark):
    """
    binary classification

    input - ``(B, 2)``

    output - ``(B, 1)``
    """
    def __init__(
        self,
        model,
        criterion=F.binary_cross_entropy_with_logits,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        n_samples = 1024,
        noise = 0.2,
        train_split=None,
        shuffle_split=True,
        normalize_x=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.make_moons(n_samples = n_samples, noise = noise, random_state=0)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=shuffle_split,
            normalize=(normalize_x, False),
            decision_boundary=True,
            boundary_act=F.sigmoid,
        )

class OlivettiFaces(DatasetBenchmark):
    """
    classification (400 samples)

    input - ``(B, 4096)``

    output - ``(B, 40)``
    """
    def __init__(
        self,
        model,
        criterion=F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.75,
        normalize_x=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.fetch_olivetti_faces(return_X_y=True)
        super().__init__(
            data_train = (x, y), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            dtypes = (torch.float32, torch.int64),
            normalize=(normalize_x, False),
        )

class OlivettiFacesAutoencoding(DatasetBenchmark):
    """
    autoencoding (400 samples)

    input - ``(B, 4096)``

    output - ``(B, 4096)``
    """
    def __init__(
        self,
        model,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.75,
        normalize_x=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.fetch_olivetti_faces(return_X_y=True)
        super().__init__(
            data_train = (x, ), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_x,),
        )


class Digits(DatasetBenchmark):
    """
    classification (1,797 samples)

    input - ``(B, 64)``

    output - ``(B, 10)``
    """
    def __init__(
        self,
        model,
        criterion=F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.75,
        normalize_x=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.load_digits(return_X_y=True)
        super().__init__(
            data_train = (x, y), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            dtypes = (torch.float32, torch.int64),
            normalize=(normalize_x, False),
        )
class Covertype(DatasetBenchmark):
    """
    classification (581,012 samples)

    input - ``(B, 54)``

    output - ``(B, 7)``
    """
    def __init__(
        self,
        model,
        criterion=F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.fetch_covtype(return_X_y=True)
        super().__init__(
            data_train = (x, y-1), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            dtypes = (torch.float32, torch.int64),
            normalize=(normalize_x, False),
        )

class KDDCup1999(DatasetBenchmark):
    """
    multi-target regression (4,898,431 samples) but by default loads 10%.

    input - ``(B, 41)``

    output - ``(B, 23)``
    """
    def __init__(
        self,
        model,
        criterion=F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
        percent10: bool = True,
    ):
        import sklearn.datasets
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        x,y = sklearn.datasets.fetch_kddcup99(return_X_y=True, percent10=percent10, random_state=0)
        x = OrdinalEncoder().fit_transform(x) # pyright:ignore[reportArgumentType]
        y = LabelEncoder().fit_transform(y)
        super().__init__(
            data_train = (x, y), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            dtypes = (torch.float32, torch.int64),
            normalize=(normalize_x, False),
        )

class Friedman1(DatasetBenchmark):
    """
    regression (100 samples by default)

    input - ``(B, default=10)``

    output - ``(B, 1)``
    """
    def __init__(
        self,
        model,
        n_samples: int = 100,
        n_features: int = 10,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
        normalize_y=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.make_friedman1(n_samples=n_samples, n_features=n_features, random_state=0)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_x, normalize_y),
        )


class Friedman2(DatasetBenchmark):
    """
    regression (100 samples by default)

    input - ``(B, 4)``

    output - ``(B, 1)``
    """
    def __init__(
        self,
        model,
        n_samples: int = 100,
        noise: float = 0,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
        normalize_y=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.make_friedman2(n_samples=n_samples, noise=noise, random_state=0)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_x, normalize_y),
        )

class Friedman3(DatasetBenchmark):
    """
    regression (100 samples by default)

    input - ``(B, 4)``

    output - ``(B, 1)``
    """
    def __init__(
        self,
        model,
        n_samples: int = 100,
        noise: float = 0,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        normalize_x=True,
        normalize_y=True,
    ):
        import sklearn.datasets
        x,y = sklearn.datasets.make_friedman3(n_samples=n_samples, noise=noise, random_state=0)
        super().__init__(
            data_train = (x, y[...,None]), # type:ignore
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_x, normalize_y),
        )
