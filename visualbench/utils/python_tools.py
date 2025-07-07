import functools
import operator
from collections import UserDict, UserList
from collections.abc import Callable, Iterable
from typing import Any, TypeVar
import math
from decimal import Decimal, ROUND_HALF_UP

def format_number(number, n):
    """Rounds to n significant digits after the decimal point."""
    if number == 0: return 0
    if math.isnan(number) or math.isinf(number) or (not math.isfinite(number)): return number
    if n <= 0: raise ValueError("n must be positive")

    dec = Decimal(str(number))
    if dec.is_zero(): return 0
    if number > 10**n or dec % 1 == 0: return int(dec)

    if abs(dec) >= 1:
        places = n
    else:
        frac_str = format(abs(dec), 'f').split('.')[1]
        leading_zeros = len(frac_str) - len(frac_str.lstrip('0'))
        places = leading_zeros + n

    quantizer = Decimal('1e-' + str(places))
    rounded_dec = dec.quantize(quantizer, rounding=ROUND_HALF_UP)

    if rounded_dec % 1 == 0: return int(rounded_dec)
    return float(rounded_dec)


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


__invalid_fname_chars = frozenset("'\\?%*:|\"<>'/")

def to_valid_fname(string:str, fallback = '~', empty_fallback = 'empty', maxlen = 127, invalid_chars = __invalid_fname_chars) -> str:
    """Makes sure filename doesn't have forbidden characters and isn't empty or too long,
    this does not ensure a valid filename as there are a lot of other rules,
    but does a fine job most of the time.

    Args:
        string (str): _description_
        fallback (str, optional): _description_. Defaults to '-'.
        empty_fallback (str, optional): _description_. Defaults to 'empty'.
        maxlen (int, optional): _description_. Defaults to 127.

    Returns:
        _type_: _description_
    """
    assert fallback not in invalid_chars
    if len(string) == 0: return empty_fallback
    return ''.join([(c if c not in invalid_chars or c.isalnum() else fallback) for c in string[:maxlen]]).strip()
