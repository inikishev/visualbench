from . import data, losses, models
from .tasks import *
from .utils import tonumpy, totensor
from .tasks.linalg import linalg_utils
from .rng import RNG

from .benchmark import Benchmark

def all_benchmarks():
    names = []
    classes = []

    def subclasses(cls):
        sub = [c for c in cls.__subclasses__() if c not in classes]
        for s in sub.copy(): sub.extend(subclasses(s))
        names.extend([c.__name__ for c in sub if c.__name__ not in names])
        return sub

    subclasses(Benchmark)
    return ", ".join(sorted(names))
