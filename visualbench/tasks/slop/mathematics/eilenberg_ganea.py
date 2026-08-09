# pylint:disable=not-callable,no-member
import cv2
import numpy as np
import torch

from ....benchmark import Benchmark


class EilenbergGanea(Benchmark):
    """Search for a nontrivial two-generator unitary matrix representation tied to the Eilenberg-Ganea conjecture.

    The Eilenberg-Ganea conjecture concerns the cohomological dimension of groups
    (specifically, that a group with a 2-dimensional Eilenberg-MacLane space actually
    admits a free action on a 2-complex). As a search/optimization task, this benchmark
    looks for an assignment of two unitary matrices (X, Y) that satisfy the group
    presentation given by the Akbulut-Kirby relations

        X Y X^-1 Y^-2 = I   and   Y X Y^-1 X^-2 = I

    which encode a braid-type group carrying highly non-trivial unitary representations.
    The parameters ``x_params``/``y_params`` are optimized so that the residual of the
    relations is driven toward zero, while a hinge-style repulsion term keeps X and Y
    far enough from the identity matrix so that the discovered representation is not the
    trivial one. The joint (X-I, Y-I) vs (R1-I, R2-I) grid is rendered as a heatmap
    ("internal_tension") to expose the braid structure.
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.x_params = torch.nn.Parameter(torch.randn(dim, dim, 2) * 0.5)
        self.y_params = torch.nn.Parameter(torch.randn(dim, dim, 2) * 0.5)

        self.eye = torch.nn.Buffer(torch.eye(dim, dtype=torch.complex64))

    def _get_unitary(self, params):
        m = torch.complex(params[..., 0], params[..., 1])
        a = m - m.adjoint()
        return torch.matrix_exp(a)

    def get_loss(self):
        X = self._get_unitary(self.x_params)
        Y = self._get_unitary(self.y_params)

        X_inv, Y_inv = X.adjoint(), Y.adjoint()

        # Akbulut-Kirby Relations
        R1 = X @ Y @ X_inv @ Y_inv @ Y_inv
        R2 = Y @ X @ Y_inv @ X_inv @ X_inv

        # Relation Errors (MSE)
        err1 = torch.linalg.matrix_norm(R1 - self.eye)
        err2 = torch.linalg.matrix_norm(R2 - self.eye)

        # NON-TRIVIALITY CONSTRAINT
        # We want dist(X, I) to be large.
        # We use a hinge-like loss to push the distance to be at least sqrt(dim)
        dist_x = torch.linalg.matrix_norm(X - self.eye)
        dist_y = torch.linalg.matrix_norm(Y - self.eye)

        # Push distance to stay around a target (e.g., 5.0)
        target_dist = 5.0
        repulsion = (dist_x - target_dist).pow(2) + (dist_y - target_dist).pow(2)

        self.log("R_identity_error", err1 + err2)
        self.log("Generator_Distance", dist_x + dist_y)

        if self.make_images:
            # VISUALIZATION IMPROVEMENT:
            # We visualize (X - I) and (Y - I).
            # This reveals the "braid" structure hidden inside the matrix.
            def prep(M):
                # Take deviation from identity and get magnitude
                diff = torch.abs(M - self.eye)
                # Normalize so the structure is always visible regardless of scale
                return diff / (diff.max() + 1e-8)

            top = torch.cat([prep(X), prep(Y)], dim=1)
            bot = torch.cat([prep(R1), prep(R2)], dim=1)
            grid = torch.cat([top, bot], dim=0)

            grid_np = (grid.detach().cpu().numpy() * 255).astype(np.uint8)
            # Use COLORMAP_JET to see high-frequency variations better
            heatmap = cv2.applyColorMap(grid_np, cv2.COLORMAP_JET)
            self.log_image(name="internal_tension", image=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), to_uint8=False)

        # We prioritize relations, but keep the repulsion strong enough to prevent collapse
        return (err1 + err2) + 0.05 * repulsion


class GroupRepresentation(Benchmark):
    """Discover a faithful non-trivial unitary representation of the trefoil knot group.

    The trefoil knot group is the group of the trefoil knot complement, and it has the two
    generator braid presentation

        <X, Y | X Y X = Y X Y>

    This benchmark optimizes two unitary matrices (X, Y) encoding a representation of that
    group. The primary loss is the residual of the braid relation

        X Y X (Y X Y)^-1  ->  I

    which must be driven to identity for the matrices to actually represent the group.
    A secondary repulsion term pushes the generators a fixed distance away from the identity
    matrix so the optimization finds a non-trivial (and high-dimensional) representation
    rather than collapsing all generators to the identity. The generators and relation error
    are visualized as a pair of MAGMA heatmaps ("representation").
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # Initialize with enough noise to find a non-trivial path
        self.x_params = torch.nn.Parameter(torch.randn(dim, dim, 2) * 0.2)
        self.y_params = torch.nn.Parameter(torch.randn(dim, dim, 2) * 0.2)
        self.eye = torch.nn.Buffer(torch.eye(dim, dtype=torch.complex64))

    def _get_unitary(self, params):
        m = torch.complex(params[..., 0], params[..., 1])
        a = m - m.adjoint() # Skew-hermitian
        return torch.matrix_exp(a)

    def get_loss(self):
        X = self._get_unitary(self.x_params)
        Y = self._get_unitary(self.y_params)

        # Trefoil Knot Relation: xyx = yxy  => xyx(yxy)^-1 = I
        # This group is known to have many non-trivial representations.
        R = X @ Y @ X @ (Y @ X @ Y).adjoint()

        # Distance from identity for the relation
        rel_loss = torch.linalg.matrix_norm(R - self.eye)

        # Distance from identity for generators (we want this to stay high)
        dist_x = torch.linalg.matrix_norm(X - self.eye)
        dist_y = torch.linalg.matrix_norm(Y - self.eye)

        # We want generators to be at least 'target_dist' away from Identity
        target_dist = 4.0
        repulsion = (dist_x - target_dist).pow(2) + (dist_y - target_dist).pow(2)

        self.log("Relation_Error", rel_loss)
        self.log("Gen_Dist", dist_x + dist_y)

        if self.make_images:
            def norm_img(M):
                diff = torch.abs(M - self.eye)
                return (diff / (diff.max() + 1e-8)).detach().cpu().numpy()

            # Top: Generators (should stay bright)
            # Bottom: Relation (should go dark/blue)
            top = np.hstack([norm_img(X), norm_img(Y)])
            bot = np.hstack([norm_img(R), np.zeros_like(norm_img(R))])
            grid = np.vstack([top, bot])

            heatmap = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            self.log_image(name="representation", image=cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), to_uint8=False)

        # Heavily weight the Relation_Error so it actually reaches ~0
        return 10.0 * rel_loss + 0.1 * repulsion
