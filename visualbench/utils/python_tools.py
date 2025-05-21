import functools
import operator
from collections import UserDict, UserList
from collections.abc import Callable, Iterable
from typing import Any, TypeVar
import math
from decimal import Decimal

def round_significant(x: Any, nsignificant: int):
    if x == 0: return 0.0
    if math.isnan(x) or math.isinf(x): return x
    if nsignificant <= 0: raise ValueError("nsignificant must be positive")

    x = Decimal(x) # otherwise there are rounding errors
    order = Decimal(10) ** math.floor(math.log10(abs(x)))
    v = round(x / order, nsignificant - 1) * order
    return float(v)



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


class SortedSet[T](UserList[T]):
    """not efficient"""
    def add(self, v: T):
        if v not in self: self.append(v)

    def intersection(self, other):
        return SortedSet(v for v in self if v in other)

    def union(self, other):
        return SortedSet(list(self) + [v for v in other if v not in self])