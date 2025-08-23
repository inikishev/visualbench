from functools import partial
import itertools
from collections.abc import Iterable, Sequence, Callable
from typing import Any

import numpy as np
import torch

from ..utils.python_tools import format_number


def _tofloatlist(x) -> list[float]:
    if isinstance(x, (int,float)): return [x]
    if isinstance(x, np.ndarray) and x.size == 1: return [float(x.item())]
    if isinstance(x, torch.Tensor) and x.numel() == 1: return [float(x.item())]
    return [float(i) for i in x]

class MBS:
    """Univariate optimization via grid search followed by multi-binary search, supports multi-objective functions, good for plotting.

    Args:
        grid (Iterable[float], optional): values for initial grid search. Defaults to (2,1,0,-1,-2,-3,-4,-5).
        step (float, optional): expansion step size. Defaults to 1.
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
        rounding (int, optional): rounding is to significant digits, avoids evaluating points that are too close.
        log_scale (bool, optional):
            whether to minimize in log10 scale. If true, it is assumed that ``grid`` is given in log10 scale,
            and evaluated points are also stored in log10 scale.
    """

    def __init__(
        self,
        grid: Iterable[float],
        step: float,
        num_candidates: int = 3,
        num_binary: int = 20,
        num_expansions: int = 20,
        rounding: int| None = 2,
        log_scale: bool = False,
    ):
        self.objectives: dict[int, dict[float,float]] = {}
        """dictionary of objectives, each maps point (x) to value (v)"""

        self.evaluated: set[float] = set()
        """set of evaluated points (x)"""

        grid = tuple(grid)
        if len(grid) == 0: raise ValueError("At least one grid search point must be specified")
        self.grid = sorted(grid)

        self.step = step
        self.num_candidates = num_candidates
        self.num_binary = num_binary
        self.num_expansions = num_expansions
        self.rounding = rounding
        self.log_scale = log_scale

    def _get_best_x(self, n: int, objective: int):
        """n best points"""
        obj = self.objectives[objective]
        v_to_x = [(v,x) for x,v in obj.items()]
        v_to_x.sort(key = lambda vx: vx[0])
        xs = [x for v,x in v_to_x]
        return xs[:n]

    def _suggest_points_around(self, x: float, objective: int):
        """suggests points around x"""
        points = list(self.objectives[objective].keys())
        points.sort()
        if x not in points: raise RuntimeError(f"{x} not in {points}")

        expansions = []
        if x == points[0]:
            expansions.append((x-self.step, 'expansion'))

        if x == points[-1]:
            expansions.append((x+self.step, 'expansion'))

        if len(expansions) != 0: return expansions

        idx = points.index(x)
        xm = points[idx-1]
        xp = points[idx+1]

        x1 = (x - (x - xm)/2)
        x2 = (x + (xp - x)/2)

        return [(x1, 'binary'), (x2, 'binary')]

    def _evaluate(self, fn, x):
        """Evaluate a point, returns False if point is already in history"""
        if self.rounding is not None: x = format_number(x, self.rounding)
        if x in self.evaluated: return False
        self.evaluated.add(x)

        if self.log_scale: vals = _tofloatlist(fn(10 ** x))
        else: vals = _tofloatlist(fn(x))

        for idx, v in enumerate(vals):
            if idx not in self.objectives: self.objectives[idx] = {}
            self.objectives[idx][x] = v

        return True

    def run(self, fn):
        # step 1 - grid search
        for x in self.grid:
            self._evaluate(fn, x)

        # step 2 - binary search
        while True:
            if (self.num_candidates <= 0) or (self.num_expansions <= 0 and self.num_binary <= 0): break

            # suggest candidates
            candidates: list[tuple[float, str]] = []

            # sample around best points
            for objective in self.objectives:
                best_points = self._get_best_x(self.num_candidates, objective)
                for p in best_points:
                    candidates.extend(self._suggest_points_around(p, objective=objective))

            # filter
            if self.num_expansions <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'expansion']

            if self.num_candidates <= 0:
                candidates = [(x,t) for x,t in candidates if t != 'binary']

            # if expansion was suggested, discard anything else
            types = [t for x, t in candidates]
            if any(t == 'expansion' for t in types):
                candidates = [(x,t) for x,t in candidates if t == 'expansion']

            # evaluate candidates
            terminate = False
            at_least_one_evaluated = False
            for x, t in candidates:
                evaluated = self._evaluate(fn, x)
                if not evaluated: continue
                at_least_one_evaluated = True

                if t == 'expansion': self.num_expansions -= 1
                elif t == 'binary': self.num_binary -= 1

                if self.num_binary < 0:
                    terminate = True
                    break

            if terminate: break
            if not at_least_one_evaluated:
                if self.rounding is None: break
                self.rounding += 1
                if self.rounding == 10: break

        # return dict[float, tuple[float,...]]
        ret = {}
        for i, objective in enumerate(self.objectives.values()):
            for x, v in objective.items():
                if x not in ret: ret[x] = [None for _ in self.objectives]
                ret[x][i] = v

        for v in ret.values():
            assert len(v) == len(self.objectives), v
            assert all(i is not None for i in v), v

        return ret

def mbs_minimize(fn, grid: Iterable[float], step:float, num_candidates: int = 3, num_binary: int = 20, num_expansions: int = 20, rounding=2, log_scale=False):
    mbs = MBS(grid, step=step, num_candidates=num_candidates, num_binary=num_binary, num_expansions=num_expansions, rounding=rounding, log_scale=log_scale)
    return mbs.run(fn)

def _unpack(x):
    if isinstance(x, tuple): return x
    return x,x

def mbs_minimize_2d(
    fn: Callable[[float, float], Any],
    grid1: Iterable[float],
    grid2: Iterable[float],
    step: float | tuple[float, float],
    num_candidates: int | tuple[int, int] = 2,
    num_binary: int | tuple[int, int] = 4,
    num_expansions: int | tuple[int, int] = 10,
    rounding: int | None | tuple[int | None, int | None] = 2,
    log_scale: bool | tuple[bool, bool] = False,
):
    # unpack
    step1,step2 = _unpack(step)
    num_candidates1,num_candidates2 = _unpack(num_candidates)
    num_binary1,num_binary2 = _unpack(num_binary)
    num_expansions1,num_expansions2 = _unpack(num_expansions)
    rounding1,rounding2 = _unpack(rounding)
    log_scale1,log_scale2 = _unpack(log_scale)

    history = {}
    def cached_fn(x: float, y: float):
        if rounding1 is not None: x = format_number(x, rounding1)
        if rounding2 is not None: y = format_number(y, rounding2)
        if (x, y) in history: return history[x, y]

        x_true = x
        y_true = y
        if log_scale1: x_true=10**x
        if log_scale2: y_true=10**y

        f = fn(x_true, y_true)
        history[(x, y)] = f
        return f

    # 1,2
    def objective12(y: float):
        mbs1 = MBS(grid1, step=step1, num_candidates=num_candidates1, num_binary=num_binary1, num_expansions=num_expansions1, rounding=rounding1, log_scale=False)
        ret = mbs1.run(lambda x: cached_fn(x, y))
        return [min(v) for v in zip(*ret.values())]

    mbs2 = MBS(grid2, step=step2, num_candidates=num_candidates2, num_binary=num_binary2, num_expansions=num_expansions2, rounding=rounding2, log_scale=False)
    mbs2.run(objective12)

    # 2,1
    def objective21(x: float):
        mbs1 = MBS(grid2, step=step2, num_candidates=num_candidates2, num_binary=num_binary2, num_expansions=num_expansions2, rounding=rounding2, log_scale=False)
        ret = mbs1.run(lambda y: cached_fn(x, y))
        return [min(v) for v in zip(*ret.values())]

    mbs2 = MBS(grid1, step=step1, num_candidates=num_candidates1, num_binary=num_binary1, num_expansions=num_expansions1, rounding=rounding1, log_scale=False)
    mbs2.run(objective21)

    return history

def grid_search(fn, lb:float, ub:float, num:int, num_expansions: int = 20, log_scale=False):
    mbs = MBS(np.linspace(lb, ub, num), step=(ub-lb)/num, num_candidates=1, num_binary=0, num_expansions=num_expansions, rounding=None, log_scale=log_scale)
    return mbs.run(fn)
