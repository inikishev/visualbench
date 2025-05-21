from ..utils.python_tools import round_significant

class MBS:
    """Multi-binary search

    Args:
        gs (tuple, optional): values for initial grid search. Defaults to (2,1,0,-1,-2,-3,-4,-5).
        num_candidates (int, optional): number of best points to sample new points around on each iteration. Defaults to 2.
        num_binary (int, optional): maximum number of new points sampled via binary search. Defaults to 7.
        num_expansions (int, optional): maximum number of expansions (not counted towards binary search points). Defaults to 7.
    """
    def __init__(self, gs=(2,1,0,-1,-2,-3,-4,-5), num_candidates: int = 2, num_binary: int = 7, num_expansions: int = 7):
        self.evaluated = {}
        """maps point (x) to value (v)"""
        self.gs = gs
        self.num_candidates = num_candidates
        self.num_binary = num_binary
        self.num_expansions = num_expansions

    def _get_best_x(self, n: int):
        """n best points"""
        v_to_x = [(v,x) for x,v in self.evaluated.items()]
        v_to_x.sort(key = lambda vx: vx[0])
        xs = [x for v,x in v_to_x]
        return xs[:n]

    def _suggest_points(self, x: float):
        """suggests points around x"""
        points = list(self.evaluated.keys())
        points.sort(key = lambda x: x[0])
        if x not in points: raise RuntimeError(f"{x} not in {points}")

        if x == points[0]:
            return [(x-1, 'expansion'), ]

        if x == points[1]:
            return [(x+1, 'expansion'), ]

        idx = points.index(x)
        xm = points[idx-1]
        xp = points[idx+1]

        x1 = (x - (x - xm)/2)
        x2 = (x + (xp - x)/2)

        return [(x1, 'binary'), (x2, 'binary')]

    def _evaluate(self, fn, x):
        key = round_significant(x, 4)
        if key in self.evaluated: return
        self.evaluated[round_significant(x, 4)] = fn(x)


    def run(self, fn):
        # step 1 - grid search
        for x in self.gs:
            self._evaluate(fn, x)

        # step 2 - binary search
        while True:
            # suggest candidates
            candidates: list[tuple[float, str]] = []

            # sample around best points
            best_points = self._get_best_x(self.num_candidates)
            for p in best_points:
                candidates.extend(self._suggest_points(p))

            # if expansion was suggested, discard anything else
            types = [t for x, t in candidates]
            if any(t == 'expansion' for t in types):
                candidates = [(x,t) for x,t in candidates if t == 'expansion']


            # evaluate candidates
            terminate = False
            for x, t in candidates:
                self._evaluate(fn, x)

                if t == 'expansion': self.num_expansions -= 1
                elif t == 'binary': self.num_binary -= 1

                if self.num_expansions < 0 or self.num_binary < 0:
                    terminate = True
                    break

            if terminate: break

        return dict(sorted(self.evaluated.items(), key=lambda xv: xv[1]))