import math
from itertools import combinations

import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark


class SteinerSystem(Benchmark):
    """inspired by https://www.youtube.com/watch?v=kLBPZ6hro5c"""
    def __init__(self, n=31, lambda_val=1.0):
        super().__init__(log_projections=True)
        self.n = n
        self.lambda_val = lambda_val
        self.frames = []

        # Generate lists of all possible triples and pairs
        elements = list(range(n))
        self.triples_list = list(combinations(elements, 3))
        self.pairs_list = list(combinations(elements, 2))

        self.num_triples = len(self.triples_list)
        self.num_pairs = len(self.pairs_list)

        # Create mappings for quick lookups
        self.triple_to_idx = {triple: i for i, triple in enumerate(self.triples_list)}
        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs_list)}

        # Trainable parameters c_abc (raw logits for sigmoid)
        # Initialize near 0 so sigmoid starts near 0.5, encouraging exploration
        self.c_abc = nn.Parameter(torch.randn(self.num_triples, device=self.device) * 0.1)

        # Construct matrix A for L(S(x))
        A = torch.zeros((self.num_pairs, self.num_triples), device=self.device)
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

        # Construct target vector O
        O_vec = torch.ones(self.num_pairs, device=self.device)
        self.O = torch.nn.Buffer(O_vec)

        # Precompute for red channel visualization: map pair_idx to list of triple_idx
        self.pair_to_contributing_triples = [[] for _ in range(self.num_pairs)]
        for triple_idx, triple_coords in enumerate(self.triples_list):
            p1 = tuple(sorted((triple_coords[0], triple_coords[1])))
            p2 = tuple(sorted((triple_coords[0], triple_coords[2])))
            p3 = tuple(sorted((triple_coords[1], triple_coords[2])))
            self.pair_to_contributing_triples[self.pair_to_idx[p1]].append(triple_idx) # type:ignore
            self.pair_to_contributing_triples[self.pair_to_idx[p2]].append(triple_idx) # type:ignore
            self.pair_to_contributing_triples[self.pair_to_idx[p3]].append(triple_idx) # type:ignore

        print(f"Initialized SteinerSystemObjective for n={n}")
        print(f"Number of triples (dim V): {self.num_triples}") # 31C3 = 4495
        print(f"Number of pairs (dim W): {self.num_pairs}")   # 31C2 = 465
        print(f"Matrix A shape: {self.A.shape}")


    def get_loss(self):
        # S(x) = sum sigmoid(c_abc) {a,b,c}
        # r_abc are the coefficients sigmoid(c_abc)
        r_abc = torch.sigmoid(self.c_abc)

        # L(S(x)) = A @ r_abc
        L_Sx = torch.matmul(self.A, r_abc)

        # Loss term 1: L(S(x)) = norm(A @ r_abc - O)^2
        loss1 = torch.sum((L_Sx - self.O)**2)

        # Loss term 2: M(S(x)) = sum r_abc * (1 - r_abc)
        # This penalizes values not close to 0 or 1
        loss2 = torch.sum(r_abc * (1 - r_abc))

        total_loss = loss1 + self.lambda_val * loss2

        # Generate frame for visualization (detach tensors from graph)
        if self._make_images:
            frame = self.generate_frame(L_Sx.detach().cpu(), r_abc.detach().cpu())
            self.log('image', frame, False, False)

        return total_loss

    def generate_frame(self, L_Sx_detached, r_abc_detached):
        image_data = np.zeros((self.n, self.n, 3), dtype=np.uint8)

        # L(S(x)) - O
        diff_L_O = L_Sx_detached - self.O

        # Max possible deviation for a pair is roughly n (e.g., if all triples involving it are 1, and O is 1).
        # Or if all are 0, deviation is -1.
        # Let's use a scale factor that maps a deviation of around +/- 5 to full intensity.
        # This is arbitrary and might need tuning.
        blue_green_scale_factor = 255.0 / 5.0

        for pair_idx, pair_coords in enumerate(self.pairs_list):
            i, j = pair_coords
            value = diff_L_O[pair_idx].item()

            if value > 0: # Positive: Blue
                intensity = min(255, int(value * blue_green_scale_factor))
                image_data[i, j, 2] = intensity
                image_data[j, i, 2] = intensity
            elif value < 0: # Negative: Green
                intensity = min(255, int(abs(value) * blue_green_scale_factor))
                image_data[i, j, 1] = intensity
                image_data[j, i, 1] = intensity

        # Red channel: sum_{c} r_{a,b,c}(1-r_{a,b,c})
        # Max value of r(1-r) is 0.25.
        # If a pair is in (n-2) triples, max sum is (n-2)*0.25. For n=31, (29)*0.25 = 7.25
        # So scale factor might be 255 / 7.25 approx 35. Let's use 40 for visibility.
        red_scale_factor = 40.0

        r_abc_np = r_abc_detached.cpu().numpy() # Move to CPU for NumPy operations

        for pair_idx, pair_coords in enumerate(self.pairs_list):
            i, j = pair_coords
            red_value_sum = 0
            for triple_idx in self.pair_to_contributing_triples[pair_idx]:
                r_val = r_abc_np[triple_idx]
                red_value_sum += r_val * (1 - r_val)

            intensity = min(255, int(red_value_sum * red_scale_factor))
            image_data[i, j, 0] = intensity
            image_data[j, i, 0] = intensity

        # Make diagonal black (or some other indicator)
        for i in range(self.n):
            image_data[i, i, :] = [50, 50, 50] # Dark grey

        return image_data

    def get_solution_triples(self, threshold=0.9):
        """Returns a list of triples whose r_abc value is above the threshold."""
        r_abc = torch.sigmoid(self.c_abc).detach().cpu().numpy()
        solution = []
        for i, r_val in enumerate(r_abc):
            if r_val > threshold:
                solution.append(self.triples_list[i])
        return solution
