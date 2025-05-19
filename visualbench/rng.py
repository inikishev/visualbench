import random
from typing import Any
import numpy as np
import torch


class RNG:
    def __init__(self, seed: "int | None | RNG"):
        if isinstance(seed, RNG):
            self.seed = seed.seed
            self.random = seed.random
            self.numpy = seed.numpy
            self._torch_generators = seed._torch_generators
        else:
            self.seed = seed
            self.random = random.Random(seed)
            self.numpy = np.random.default_rng(seed)

            self._torch_generators = {}
    def copy(self):
        return RNG(self.seed)

    def torch(self, device: Any = None) -> torch.Generator:
        if device is None: device = torch.get_default_device()
        elif not isinstance(device, torch.device): device = torch.device(device)
        key = (device.type, device.index)
        if key not in self._torch_generators:
            self._torch_generators[key] = torch.Generator(device).manual_seed(self.seed) if self.seed else None
        return self._torch_generators[key]

Seed = int | RNG | None