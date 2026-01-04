import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...benchmark import Benchmark
from ...utils import to_CHW, to_square


class FarrellJonesAssembly(Benchmark):
    """
    Optimizing the Farrell-Jones Assembly Map.

    The task is to find a set of 'Local' topological signatures (from virtually
    cyclic subgroups) that assemble into a 'Global' topological signature.

    We model the Assembly Map as a sparse convolution-like operator where local
    periodic structures (cyclic subgroups) are summed to match a global pattern.
    """
    def __init__(self, target=None, num_subgroups: int = 8):
        super().__init__()
        self.num_subgroups = num_subgroups

        # 1. The Global Signature (Target)
        # In a real case, this is the K-theory element of the group ring RG.
        # We create a complex target representing 'topological information'.
        if target is None:
            target = torch.zeros((1, 128, 128))
            # Create some 'global' structure
            x = torch.linspace(-1, 1, 128)
            y = torch.linspace(-1, 1, 128)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            target[0] = torch.sin(10 * grid_x) * torch.cos(10 * grid_y)
        else:
            target = to_square(to_CHW(target))

        self.target = torch.nn.Buffer(target)
        size = self.size = self.target.size(-1)
        # 2. Local Signatures (Parameters to optimize)
        # These represent K-theory elements of virtually cyclic subgroups.
        # We model them as 1D 'strands' or small kernels that will be 'assembled'.
        self.local_elements = torch.nn.Parameter(
            torch.randn(num_subgroups, self.target.size(0), size // 4, size // 4)
        )

        # 3. Reference Image for visualization
        self.add_reference_image(name="Global Target", image=self.target, to_uint8=True)

    def assembly_map(self, locals_):
        """
        A simplified Assembly Map A: H(E_Vcyc) -> K(RG).
        Maps local cyclic data into the global space via 'lifts'.
        """
        # We simulate the assembly by placing local kernels at various
        # 'group orbits' across the global manifold.
        assembled = torch.zeros_like(self.target)

        # In this toy model, each subgroup acts on a specific 'patch' or direction
        for i in range(self.num_subgroups):
            # Scale and translate local data to simulate the assembly map
            # This is a proxy for the induction/push-forward maps in K-theory
            patch = F.interpolate(locals_[i:i+1], size=(self.size, self.size), mode='bilinear')
            assembled += patch[0]

        return assembled

    def get_loss(self):
        # Apply the Assembly Map
        prediction = self.assembly_map(self.local_elements)

        # The conjecture states the map is an isomorphism.
        # Therefore, there must exist local elements that perfectly reconstruct the global.
        reconstruction_loss = F.mse_loss(prediction, self.target)

        # Regularization: Sparse 'virtually cyclic' elements are preferred
        sparsity_loss = torch.mean(torch.abs(self.local_elements)) * 0.01

        total_loss = reconstruction_loss + sparsity_loss

        # Log metrics
        self.log("MSE Loss", reconstruction_loss)
        self.log("Sparsity", sparsity_loss)

        if self._make_images:
            # 1. The Assembled Result
            self.log_image(name='Assembled Global', image=prediction, to_uint8=True)

            # 2. The Error Map (Obstruction to FJC)
            error_map = torch.abs(prediction[0] - self.target[0])
            self.log_image(name='Obstruction Map', image=error_map, to_uint8=True)

            # 3. Local Elements Visual (The "Inducing" data)
            # We tile the small local kernels into one grid using torch
            local_grid = self._make_grid(self.local_elements)
            self.log_image(name='Local Signatures', image=local_grid, to_uint8=True)

        return total_loss

    def _make_grid(self, tensor):
        """Helper to tile local kernels into a single image without matplotlib."""
        n, c, h, w = tensor.shape
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid = torch.zeros((rows * h, cols * w))
        for i in range(n):
            r, c = i // cols, i % cols
            grid[r*h:(r+1)*h, c*w:(c+1)*w] = tensor[i, 0]
        return grid
