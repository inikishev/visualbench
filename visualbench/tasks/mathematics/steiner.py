import math
from itertools import combinations

import numpy as np
import torch
from torch import nn

from ...benchmark import Benchmark


class SteinerSystem(Benchmark):
    """Steiner system objective highly inspired by https://www.youtube.com/watch?v=kLBPZ6hro5c"""
    def __init__(self, n=31, lambda_=1.0):
        super().__init__()
        self.n = n
        self.lambda_ = lambda_
        self.frames = []

        elements = list(range(n))
        self.triples_list = list(combinations(elements, 3))
        self.pairs_list = list(combinations(elements, 2))

        self.num_triples = len(self.triples_list)
        self.num_pairs = len(self.pairs_list)

        self.triple_to_idx = {triple: i for i, triple in enumerate(self.triples_list)}
        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs_list)}

        self.c_abc = nn.Parameter(torch.randn(self.num_triples) * 0.1)

        # matrix A for L(S(x))
        A = torch.zeros((self.num_pairs, self.num_triples))
        for triple_idx, triple_coords in enumerate(self.triples_list):
            # triple_coords is (a,b,c)
            # Add 1 to relevant pair entries for this triple
            p1 = tuple(sorted((triple_coords[0], triple_coords[1])))
            p2 = tuple(sorted((triple_coords[0], triple_coords[2])))
            p3 = tuple(sorted((triple_coords[1], triple_coords[2])))

            A[self.pair_to_idx[p1], triple_idx] = 1 # type:ignore
            A[self.pair_to_idx[p2], triple_idx] = 1 # type:ignore
            A[self.pair_to_idx[p3], triple_idx] = 1 # type:ignore
        self.A = torch.nn.Buffer(A)

        # target vector O
        O_vec = torch.ones(self.num_pairs)
        self.O = torch.nn.Buffer(O_vec)

        # precomputed visualization
        self.pair_to_contributing_triples = [[] for _ in range(self.num_pairs)]
        for triple_idx, triple_coords in enumerate(self.triples_list):
            p1 = tuple(sorted((triple_coords[0], triple_coords[1])))
            p2 = tuple(sorted((triple_coords[0], triple_coords[2])))
            p3 = tuple(sorted((triple_coords[1], triple_coords[2])))
            self.pair_to_contributing_triples[self.pair_to_idx[p1]].append(triple_idx) # type:ignore
            self.pair_to_contributing_triples[self.pair_to_idx[p2]].append(triple_idx) # type:ignore
            self.pair_to_contributing_triples[self.pair_to_idx[p3]].append(triple_idx) # type:ignore


    def get_loss(self):
        r_abc = torch.sigmoid(self.c_abc)
        L_Sx = torch.matmul(self.A, r_abc)
        loss1 = torch.sum((L_Sx - self.O)**2)
        loss2 = torch.sum(r_abc * (1 - r_abc))
        total_loss = loss1 + self.lambda_ * loss2

        if self._make_images:
            frame = self._make_frame(L_Sx.detach(), r_abc.detach())
            self.log_image('image', frame, to_uint8=False)

        return total_loss

    @torch.no_grad
    def _make_frame(self, L_Sx, r_abc):
        image = np.zeros((self.n, self.n, 3), dtype=np.uint8)

        diff_L_O = (L_Sx - self.O).cpu()

        # deviation of +5/-5 (maybe tune this).
        blue_green = 255.0 / 5.0

        for pair_idx, pair_coords in enumerate(self.pairs_list):
            i, j = pair_coords
            value = diff_L_O[pair_idx].item()

            if value > 0:
                intensity = min(255, int(value * blue_green))
                image[i, j, 2] = intensity
                image[j, i, 2] = intensity
            elif value < 0:
                intensity = min(255, int(abs(value) * blue_green))
                image[i, j, 1] = intensity
                image[j, i, 1] = intensity

        red = 40.0

        r_abc_np = r_abc.cpu().numpy()

        for pair_idx, pair_coords in enumerate(self.pairs_list):
            i, j = pair_coords
            red_value_sum = 0
            for triple_idx in self.pair_to_contributing_triples[pair_idx]:
                r_val = r_abc_np[triple_idx]
                red_value_sum += r_val * (1 - r_val)

            intensity = min(255, int(red_value_sum * red))
            image[i, j, 0] = intensity
            image[j, i, 0] = intensity

        for i in range(self.n):
            image[i, i, :] = [50, 50, 50]

        return image

    def get_solution_triples(self, threshold=0.9):
        r_abc = torch.sigmoid(self.c_abc).detach().cpu().numpy()
        solution = []
        for i, r_val in enumerate(r_abc):
            if r_val > threshold:
                solution.append(self.triples_list[i])
        return solution
