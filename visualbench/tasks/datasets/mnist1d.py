"""
mnist1d from
https://github.com/greydanus/mnist1d
"""
from typing import TYPE_CHECKING
from importlib.util import find_spec
import pickle
from urllib.request import urlopen

import torch
from torch import nn
from torch.nn import functional as F

from ...utils import CUDA_IF_AVAILABLE
from .dataset import DatasetBenchmark

if TYPE_CHECKING or find_spec("mnist1d") is not None:
    from mnist1d.data import make_dataset
else:
    make_dataset = None

class ObjectView:
    """this is taken from mnist1d.utils (i added with)"""
    def __init__(self, d): self.__dict__ = d

def _load_frozen(path = None):
    """loads frozen mnist1d. if path is None downloads it."""
    if path is None:
        url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
        with urlopen(url) as f:
            data = pickle.load(f)

    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
    return data

def _process(data:dict, dtype, device):
    # data.keys()
    # >>> dict_keys(['x', 'x_test', 'y', 'y_test', 't', 'templates'])  # these are NumPy arrays
    # x is (4000, 40), x_test is (1000, 40)
    # y and y_test are (4000, ) and (1000, )
    x = torch.tensor(data['x'], dtype = dtype, device = device)
    x_test = torch.tensor(data['x_test'], dtype = dtype, device = device)
    y = torch.tensor(data['y'], dtype = torch.int64, device = device)
    y_test = torch.tensor(data['y_test'], dtype = torch.int64, device = device)
    return (x, y), (x_test, y_test)

def get_frozen_mnist1d(path = None, dtype = torch.float32, device = 'cuda'):
    data = _load_frozen(path)
    return _process(data, dtype, device)

def get_mnist1d( # pylint:disable = dangerous-default-value
    num_samples=5000,
    train_split=0.8,
    template_len=12,
    padding=[36, 60],
    scale_coeff=4,
    max_translation=48,
    corr_noise_scale=0.25,
    iid_noise_scale=2e-2,
    shear_scale=0.75,
    shuffle_seq=False,
    final_seq_length=40,
    seed=42,
    url="https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl",
    template=None,
    dtype=torch.float32,
    device: torch.types.Device = CUDA_IF_AVAILABLE,
):
    """loads frozen mnist1d. if path is None downloads it.

    returns `((x, y), (x_test, y_test))`.

    x and x_test is `(num_samples, final_seq_length)`; y and y_test are `(num_samples)`.

    By default x is `(4000, 40)`, y is `(4000)`, x_test is `(1000, 40)`, y_test is `(1000)`
    """
    kwargs = locals().copy()
    template = kwargs.pop('template')
    dtype = kwargs.pop('dtype')
    device = kwargs.pop('device')
    data = make_dataset(ObjectView(kwargs), template)

    return _process(data, dtype, device)


class Mnist1d(DatasetBenchmark):
    """
    Input - ``(B, 40)``

    output - logits ``(B, 10)``
    """
    def __init__(
        self,
        model,
        num_samples=5000,
        criterion=F.cross_entropy,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split: float = 0.8,
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
    ):
        (x,y), (x_test, y_test) = get_mnist1d(num_samples=num_samples, train_split=train_split)
        super().__init__(
            data_train = (x, y),
            data_test = (x_test, y_test),
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size =test_batch_size,
            dtypes = (torch.float32, torch.int64),
            data_device=data_device,
        )

class Mnist1dAutoencoding(DatasetBenchmark):
    """
    Input - (B, 40)

    output - (B, 40)
    """
    def __init__(
        self,
        model,
        num_samples=5000,
        criterion=F.mse_loss,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
    ):
        (x,_), (x_test, _) = get_mnist1d(num_samples=num_samples)
        super().__init__(
            data_train = (x, ),
            data_test = (x_test, ),
            model = model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size = test_batch_size,
            dtypes = (torch.float32,)
        )

