import warnings
from collections import UserDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


class Logger(UserDict[str, dict[int, Any]]):
    def log(self, step: int, metric: str, value: Any):
        if metric not in self: self[metric] = {step: value}
        else: self[metric][step] = value

    def first(self, metric):
        return next(iter(self[metric].values()))

    def last(self, metric):
        return list(self[metric].values())[-1]

    def list(self, metric): return list(self[metric].values())
    def numpy(self, metric): return np.asarray(self.list(metric))
    def tensor(self, metric): return torch.from_numpy(self.numpy(metric).copy())
    def steps(self, metric): return list(self[metric].keys())

    def min(self, metric): return np.min(self.list(metric))
    def nanmin(self, metric): return np.nanmin(self.list(metric))
    def max(self, metric): return np.max(self.list(metric))
    def nanmax(self, metric): return np.nanmax(self.list(metric))
    def sum(self, metric): return np.sum(self.list(metric))

    def interp(self, metric: str) -> np.ndarray:
        """Returns a list of values for a given key, interpolating missing steps."""
        steps = range(max(len(v) for v in self.values()))
        existing = self[metric]
        return np.interp(steps, list(existing.keys()), list(existing.values()))

    def stepmin(self, metric:str) -> int:
        idx = np.nanargmin(self.list(metric)).item()
        return list(self[metric].keys())[idx]

    def stepmax(self, metric:str) -> int:
        idx = np.nanargmax(self.list(metric)).item()
        return list(self[metric].keys())[idx]

    def closest(self, metric: str, step: int):
        """same as logger[metric][step] but returns closest value if idx doesn't exist"""
        steps = np.asarray(self.steps(metric), dtype=np.int64)
        idx = np.abs(steps - step).argmin().item()
        return self[metric][steps[int(idx)]]


    def save(self, fname: str):
        """Save this logger to a compressed numpy array file (npz)."""
        arrays = {}

        for k in self.keys():
            try:
                arrays[f"__STEPS__.{k}"] = np.asarray(self.steps(k))
                arrays[f"__VALUES__.{k}"] = self.numpy(k)

            except Exception as e:
                warnings.warn(f"Failed to save `{k}`: {e}")

        np.savez_compressed(fname, **arrays)

    def load(self, fname: str, allow_pickle=False):
        """Load data from a compressed numpy array file (npz) to this logger."""
        arrays: Mapping[str, np.ndarray] = np.load(fname, allow_pickle=allow_pickle)
        for k, array in arrays.items():
            if k.startswith('__STEPS__.'):
                name = k.replace("__STEPS__.", "")
                values = arrays[f"__VALUES__.{name}"]
                self[name] = dict(zip(array, values))

    @classmethod
    def from_file(cls, fname: str):
        logger = cls()
        logger.load(fname)
        return logger