# pylint: disable=undefined-variable
from collections.abc import Mapping, MutableMapping, Sequence


def normalize_string(s:str):
    return ''.join([c.lower() for c in s if c.isalnum()])

__all__ = [
    "RelaxedMultikeyDict",
]


class RelaxedMultikeyDict[VT](MutableMapping[str, VT]):
    def __init__(self):
        self._prim_to_values: dict[str, VT] = dict()
        """Mapping from primary keys to values"""
        self._relaxed_to_prim: dict[str, str] = dict()
        """Mapping from relaxed keys to primary keys"""
        self._relaxed_to_keys: dict[str, str] = dict()
        """Mapping from relaxed keys to original (non-relaxed) keys"""

    def __getitem__(self, key:str) -> VT:
        return self._prim_to_values[self._relaxed_to_prim[normalize_string(key)]]

    def __contains__(self, keys: str | Sequence[str]) -> bool: # type:ignore
        if isinstance(keys, str): return normalize_string(keys) in self._relaxed_to_prim
        return len(set(self._relaxed_to_prim.keys()).intersection([normalize_string(i) for i in keys])) > 0

    def __setitem__(self, keys:str | Sequence[str], value: VT):
        if isinstance(keys, str): keys = (keys, )
        relaxed_keys = [normalize_string(i) for i in keys]

        if len(set(self._relaxed_to_prim.keys()).intersection(relaxed_keys)) > 0:
            raise ValueError(f'Those keys already exist: {set(self._relaxed_to_prim.keys()).intersection(relaxed_keys)}')

        # 1st key is the primary key
        self._prim_to_values[keys[0]] = value

        for kr, ko in zip(relaxed_keys, keys):
            self._relaxed_to_prim[kr] = keys[0]
            self._relaxed_to_keys[kr] = ko

    def __delitem__(self, key:str) -> None:
        relaxed_key = normalize_string(key)
        primary_key = self._relaxed_to_prim[relaxed_key]

        del self._relaxed_to_keys[relaxed_key]
        del self._relaxed_to_prim[relaxed_key]
        del self._prim_to_values[primary_key]

    def __iter__(self):
        return iter(self._prim_to_values)

    def __len__(self):
        return len(self._prim_to_values)

    def keys(self):
        return self._prim_to_values.keys()

    def values(self):
        return self._prim_to_values.values()

    def items(self):
        return self._prim_to_values.items()

    def all_keys(self):
        # returns values which are un-relaxed keys
        return self._relaxed_to_keys.values()

    def all_relaxed_keys(self):
        return self._relaxed_to_keys.keys()

    def relaxed_to_orig(self, key:str):
        return self._relaxed_to_keys[normalize_string(key)]

    def update(self, other: "Mapping[str | Sequence[str], VT] | RelaxedMultikeyDict[VT]") -> None:
        if isinstance(other, RelaxedMultikeyDict):
            self._prim_to_values.update(other._prim_to_values)
            self._relaxed_to_prim.update(other._relaxed_to_prim)
            self._relaxed_to_keys.update(other._relaxed_to_keys)
        else:
            for k, v in other.items():
                self[k] = v
