from collections import UserDict
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
        return self[metric][int(idx)]