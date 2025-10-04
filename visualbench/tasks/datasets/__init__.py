from importlib.util import find_spec
from typing import TYPE_CHECKING

from .sklearn import CaliforniaHousing, Moons, OlivettiFaces, OlivettiFacesAutoencoding, Covertype, KDDCup1999, Digits, Friedman1, Friedman2, Friedman3

from .mnist1d import Mnist1d, Mnist1dAutoencoding

from .seg1d import SynthSeg1d
from .torchvision import CustomDataset,TorchvisionDataset, MNIST,FashionMNIST,FashionMNISTAutoencoding, CIFAR10, CIFAR100
from .other import WDBC