from collections import abc

import torch
from light_dataloader import LightDataLoader, TensorDataLoader
from myai.data import DS
# I temporarily import it from myai. I need to move this into this lib later
from myai.python_tools import Composable
from myai.rng import RNG


def make_dataset(
    dataset: abc.Sequence,
    batch_size: int | None,
    test_batch_size : int | None = None,
    test_split: int | float | None = None,
    test_dataset: abc.Sequence | None = None,
    shuffle_split = True,
    preload = True,
    loader: Composable | None = None,
    is_dataloader: bool = False,
    seed: int | RNG | None = 0,
):
    """make benchmark from dataset.

    Args:
        dataset (Sequence): sequence of samples or tuples to collate, or a dataloader.
        batch_size (int | None): batch size, None if fullbatch or dataset is a dataloader already.
        test_batch_size (int | None): test batch size, None if fullbatch or dataset is a dataloader already.
        test_split (int | float | None): test split (0.2 for 20% test split).
            None if test dataset is provided or no test dataset needed.
        shuffle_split (bool, optional): whether to shuffle before splitting. Defaults to True.
        test_dataset (Sequence | None, optional): test dataset. Defaults to None.
        preload (bool, optional): preload dataset into memory. Defaults to True.
        loader (optional): loader for the dataset, callable or sequence of callables.
        is_dataloader (bool, optional): set to True if `dataset` and maybe `test_dataset` are dataloaders. Defaults to False.
        seed (int | RNG | None, optional): integer seed, RNG object or None for random seed. Defaults to 0.
    """
    rng = RNG(seed)

    if (test_split is not None) and (test_dataset is not None):
        raise ValueError("can't have test_p and test dataset at the same time")
    if (batch_size is not None) and is_dataloader:
        raise ValueError("can't have batch_size and is_dataloader at the same time")
    if (loader is not None) and is_dataloader:
        raise ValueError("loader and transform are incompatible with is_dataloader")
    if (batch_size is not None) and batch_size >= len(dataset):
        raise ValueError(f"{batch_size = } which is bigger than {len(dataset) = }")
    if (test_batch_size is not None) and test_batch_size >= len(dataset):
        raise ValueError(f"{test_batch_size = } which is bigger than {len(dataset) = }")
    if (batch_size is None) and (test_batch_size is not None):
        raise ValueError("if batch_size is None test_batch size must be None")

    # train/test split
    if test_split is not None:
        if not isinstance(dataset, DS):
            ds = DS()
            if preload: ds.add_samples_(dataset, loader = loader)
            else: ds.add_dataset_(dataset, loader = loader)
        else:
            ds = dataset
        ds_train, ds_test = ds.split(test_split, shuffle = shuffle_split, seed = rng)

    else:
        ds_train = dataset
        ds_test = test_dataset

    # make a dataloader if not fullbatch or is_batched
    if is_dataloader:
        train_data, test_data = ds_train, ds_test
    elif batch_size is not None:
        train_data = LightDataLoader(ds_train, batch_size, shuffle=True, seed = rng.seed)
    else: # batch_size is None, stack a fullbatch
        train_data = (next(iter(LightDataLoader(ds_train, len(ds_train)*2, shuffle=False))), )

    # make test dataloader
    if ds_test is not None:
        if test_batch_size is not None:
            test_data = LightDataLoader(ds_test, test_batch_size, shuffle=False)
        else:
            test_data = (next(iter(LightDataLoader(ds_test, len(ds_test)*2, shuffle=False))), )
    else: test_data = None

    return train_data, test_data


def make_dataset_from_tensor(
    dataset: torch.Tensor | abc.Sequence[torch.Tensor],
    batch_size: int | None,
    test_batch_size : int | None = None,
    test_split: int | float | None = None,
    test_dataset: torch.Tensor | abc.Sequence[torch.Tensor] | None = None,
    shuffle_split = True,
    memory_efficient = False,
    seed: int | RNG | None = 0,
):
    rng = RNG(seed)
    length = dataset.shape[0] if isinstance(dataset, torch.Tensor) else dataset[0].shape[0]

    if (test_split is not None) and (test_dataset is not None):
        raise ValueError("can't have test_p and test dataset at the same time")
    if (batch_size is not None) and batch_size >= length:
        raise ValueError(f"{batch_size = } which is bigger than {length = }")
    if (test_batch_size is not None) and test_batch_size >= length:
        raise ValueError(f"{test_batch_size = } which is bigger than {length = }")
    if (batch_size is None) and (test_batch_size is not None):
        raise ValueError("if batch_size is None test_batch size must be None")

    # train/test split
    if test_split is not None:
        if isinstance(test_split, float):
            test_split = round(test_split * length)
        if isinstance(dataset, torch.Tensor):
            if shuffle_split:
                perm = torch.randperm(length, device=dataset.device, generator = rng.torch(dataset.device))
                dataset = dataset[perm]
            train_data, test_data = dataset[:test_split], dataset[test_split:]
        else:
            if shuffle_split:
                perm = torch.randperm(length, device=dataset[0].device, generator = rng.torch(dataset[0].device))
                dataset = [i[perm] for i in dataset]
            train_data = [i[:test_split] for i in dataset]
            test_data = [i[test_split:] for i in dataset]
    else:
        train_data = dataset
        test_data = test_dataset

    # make a dataloader if not fullbatch or is_batched
    if batch_size is not None:
        train_data = TensorDataLoader(train_data, batch_size, shuffle=True, seed = rng.seed, memory_efficient=memory_efficient)
    # otherwise dataset is already collated
    else: train_data = (train_data, )

    # make test dataloader
    if test_data is not None:
        if test_batch_size is not None:
            test_data = TensorDataLoader(test_data, test_batch_size, shuffle=False, memory_efficient=memory_efficient)

        else: test_data = (test_data, )

    return train_data, test_data