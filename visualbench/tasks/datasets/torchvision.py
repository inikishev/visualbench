from functools import partial
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision.transforms import v2
from .dataset import DatasetBenchmark
from ...utils import CUDA_IF_AVAILABLE

def _uncollate(dataset: Iterable, loader: Callable | None):
    uncollated = [[]]
    for sample in dataset:
        if loader is not None:
            sample = loader(sample)

        if isinstance(sample, dict):
            sample = tuple(sample.values())

        if isinstance(sample, (list, tuple)):
            for i, s in enumerate(sample):
                if i+1 > len(uncollated):
                    uncollated.append([])
                uncollated[i].append(s)

        else:
            uncollated[0].append(sample)

    return uncollated

class CustomDataset(DatasetBenchmark):
    def __init__(
        self,
        ds_train: Iterable,
        ds_test: Iterable | None,
        loader: Callable | None,
        model: torch.nn.Module,
        criterion: Callable,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split: int | float | None = None,
        shuffle_split=False,
        normalize: bool | Sequence[bool] = False,
        dtypes: torch.dtype | Sequence[torch.dtype] = torch.float32,
        data_device: torch.types.Device = 'cpu',
        decision_boundary = False,
        resolution = 192,
        boundary_act = None,
        truncate: int | None = None,
        seed = 0
    ):
        if truncate is not None:
            ds_train = [sample for i, sample in enumerate(ds_train) if i < truncate]

        super().__init__(
            data_train = _uncollate(ds_train, loader),
            data_test = _uncollate(ds_test, loader) if ds_test is not None else None,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=shuffle_split,
            normalize=normalize,
            dtypes=dtypes,
            data_device=data_device,
            decision_boundary=decision_boundary,
            resolution=resolution,
            boundary_act=boundary_act,
            seed=seed,
        )

class TorchvisionDataset(CustomDataset):
    def __init__(
        self,
        cls,
        root: str,
        model: torch.nn.Module,
        criterion: Callable,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split: int | float | None = None,
        shuffle_split=False,
        loader: Callable | None = None,
        transform: Callable | None = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
        normalize: bool | Sequence[bool] = False,
        dtypes: torch.dtype | Sequence[torch.dtype] = torch.float32,
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        decision_boundary = False,
        resolution = 192,
        boundary_act = None,
        download:bool=True,
        truncate: int | None = None,
        seed = 0,
    ):
        ds_train = cls(root=root, transform=transform, train=True, download=download)
        ds_test = cls(root=root, transform=transform, train=False, download=download)
        super().__init__(
            ds_train = ds_train,
            ds_test=ds_test,
            loader=loader,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            train_split=train_split,
            shuffle_split=shuffle_split,
            normalize=normalize,
            dtypes=dtypes,
            data_device=data_device,
            decision_boundary=decision_boundary,
            resolution=resolution,
            boundary_act=boundary_act,
            truncate=truncate,
            seed=seed,
        )

class MNIST(TorchvisionDataset):
    """
    classification

    input - ``(B, 1, 28, 28)``

    output - ``(B, 10)``
    """
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        criterion: Callable = F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,

        normalize: bool | Sequence[bool] = (True, False),
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        download:bool=True,
        seed = 0,
        truncate:int | None = None,
    ):
        super().__init__(
            cls=datasets.MNIST,
            root=root,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
            normalize=normalize,
            dtypes=(torch.float32, torch.int64),
            data_device=data_device,
            download=download,
            seed=seed,
            truncate=truncate,
        )

class CIFAR10(TorchvisionDataset):
    """
    classification

    input - ``(B, 3, 32, 32)``

    output - ``(B, 10)``
    """
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        criterion: Callable = F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,

        normalize: bool | Sequence[bool] = (True, False),
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        download:bool=True,
        seed = 0,
        truncate:int | None = None,
    ):
        super().__init__(
            cls=datasets.CIFAR10,
            root=root,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
            normalize=normalize,
            dtypes=(torch.float32, torch.int64),
            data_device=data_device,
            download=download,
            seed=seed,
            truncate=truncate,
        )

class CIFAR100(TorchvisionDataset):
    """
    classification

    input - ``(B, 3, 32, 32)``

    output - ``(B, 100)``
    """
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        criterion: Callable = F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,

        normalize: bool | Sequence[bool] = (True, False),
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        download:bool=True,
        seed = 0,
        truncate:int | None = None,
    ):
        super().__init__(
            cls=datasets.CIFAR100,
            root=root,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
            normalize=normalize,
            dtypes=(torch.float32, torch.int64),
            data_device=data_device,
            download=download,
            seed=seed,
            truncate=truncate,
        )

class FashionMNIST(TorchvisionDataset):
    """
    classification

    input - ``(B, 1, 28, 28)``

    output - ``(B, 10)``
    """
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        criterion: Callable = F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,

        normalize: bool | Sequence[bool] = (True, False),
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        download:bool=True,
        seed = 0,
        truncate:int | None = None,
    ):
        super().__init__(
            cls=datasets.FashionMNIST,
            root=root,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
            normalize=normalize,
            dtypes=(torch.float32, torch.int64),
            data_device=data_device,
            download=download,
            seed=seed,
            truncate=truncate,
        )

class FashionMNISTAutoencoding(TorchvisionDataset):
    """
    autoencoding

    input - ``(B, 1, 28, 28)``

    output - ``(B, 1, 28, 28)``
    """
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        criterion: Callable = F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,

        normalize: bool = True,
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        download:bool=True,
        seed = 0,
        truncate:int | None = None,
    ):
        super().__init__(
            cls=datasets.FashionMNIST,
            root=root,
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            loader = lambda x: x[0],
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
            normalize=normalize,
            dtypes=torch.float32,
            data_device=data_device,
            download=download,
            seed=seed,
            truncate=truncate,
        )

