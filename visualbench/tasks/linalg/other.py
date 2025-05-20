from collections.abc import Callable
from typing import Literal, Any
import warnings
import torch

from ...benchmark import Benchmark
from ...utils import format, algebras
from . import linalg_utils

