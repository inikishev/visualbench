import functools
import operator
from collections import UserDict
from collections.abc import Callable, Iterable
from typing import Any, TypeVar


def round_significant(x: float, nsignificant: int):
    if x > 1: return round(x, nsignificant)
    if x == 0: return x

    v = 1
    for n in range(100):
        v /= 10
        if abs(x) > v: return round(x, nsignificant + n)

    return x # x is nan
    #raise RuntimeError(f"wtf {x = }")


def _flatten_no_check(iterable: Iterable) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. Note that if `iterable` is not Iterable, this will return `[iterable]`."""
    if isinstance(iterable, Iterable) and not isinstance(iterable, str):
        return [a for i in iterable for a in _flatten_no_check(i)]
    return [iterable]

def flatten(iterable: Iterable) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. If `iterable` is not iterable, raises a TypeError."""
    if isinstance(iterable, Iterable): return [a for i in iterable for a in _flatten_no_check(i)]
    raise TypeError(f'passed object is not an iterable, {type(iterable) = }')

X = TypeVar("X")
def reduce_dim(x:Iterable[Iterable[X]]) -> list[X]: # pylint:disable=E0602
    """Reduces one level of nesting. Takes an iterable of iterables of X, and returns an iterable of X."""
    return functools.reduce(operator.iconcat, x, [])
