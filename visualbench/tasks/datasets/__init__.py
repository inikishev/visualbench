from importlib.util import find_spec
from typing import TYPE_CHECKING

from .mnist1d import Mnist1d, Mnist1dAutoencoding
from .other import WDBC
from .seg1d import SynthSeg1d
from .sklearn import (
    CaliforniaHousing,
    Covertype,
    Digits,
    Friedman1,
    Friedman2,
    Friedman3,
    KDDCup1999,
    Moons,
    OlivettiFaces,
    OlivettiFacesAutoencoding,
)
from .stress import EnglishWords

if TYPE_CHECKING or find_spec("torchvision") is not None:
    from .torchvision import (
        CIFAR10,
        CIFAR100,
        MNIST,
        CustomDataset,
        FashionMNIST,
        FashionMNISTAutoencoding,
        TorchvisionDataset,
    )
from .xor import XOR