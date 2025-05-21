import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from ..utils.python_tools import round_significant


def _tofloatlist(x) -> list[float]:
    if isinstance(x, (int,float)): return [x]
    if isinstance(x, np.ndarray) and x.size == 1: return [float(x.item())]
    if isinstance(x, torch.Tensor) and x.numel() == 1: return [float(x.item())]
    return [float(i) for i in x]

class MBS:
    """Univariate optimization via grid search followed by multi-binary search, supports multi-objective functions, good for plotting.

    Args:
        gs (Iterable[float], optional): values for initial grid search. Defaults to (2,1,0,-1,-2,-3,-4,-5).
        step (float, optional): expansion step size. Defaults to 1.
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
        rounding (int, optional): rounding is to significant digits, avoids evaluating points that are too close.
    """
    def __init__(self, grid: Iterable[float], step:float, num_candidates: int = 4, num_binary: int = 20, num_expansions: int = 20, rounding=2):
        self.objectives: dict[int, dict[float,float]] = {}
        """dictionary of objectives, each maps point (x) to value (v)"""

        self.evaluated: set[float] = set()
        """set of evaluated points (x)"""

        grid = tuple(grid)
        if len(grid) == 0: raise ValueError("At least one grid search point must be specified")
        self.grid = grid

        self.step = step
        self.num_candidates = num_candidates
        self.num_binary = num_binary
        self.num_expansions = num_expansions
        self.rounding = rounding

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
        key = round_significant(x, self.rounding)
        if key in self.evaluated: return False
        self.evaluated.add(key)

        vals = _tofloatlist(fn(x))
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
            # suggest candidates
            candidates: list[tuple[float, str]] = []

            # sample around best points
            for objective in self.objectives:
                best_points = self._get_best_x(self.num_candidates, objective)
                for p in best_points:
                    candidates.extend(self._suggest_points_around(p, objective=objective))

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

                if self.num_expansions < 0 or self.num_binary < 0:
                    terminate = True
                    break

            if terminate: break
            if not at_least_one_evaluated: self.rounding += 1

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

def minimize(fn, grid: Iterable[float], step:float, num_candidates: int = 4, num_binary: int = 20, num_expansions: int = 20, rounding=2):
    mbs = MBS(grid, step=step, num_candidates=num_candidates, num_binary=num_binary, num_expansions=num_expansions, rounding=rounding)
    return mbs.run(fn)
