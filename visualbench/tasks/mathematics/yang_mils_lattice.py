import torch
import torch.nn.functional as F

from ...benchmark import Benchmark
from ...utils import to_HW, to_square

class YangMillsLattice(Benchmark):
    """looks cool ish"""
    def __init__(self, source_mask=None):
        super().__init__()

        if source_mask is None:
            # Create a "Source": We force a specific Wilson Loop to be non-zero
            # This represents two static quarks at (16, 16) and (48, 16)
            source_mask = torch.zeros((64, 64))
            source_mask[16:48, 16] = 1.0 # Vertical line of energy

        else:
            source_mask = to_square(to_HW(source_mask))

        self.source_mask = torch.nn.Buffer(source_mask)

        self.grid_size = source_mask.size(-1)
        # [Dim, G, G, 4] links
        self.links = torch.nn.Parameter(torch.randn(2, self.grid_size, self.grid_size, 4) * 0.01)

        self.add_reference_image("vacuum state", self.source_mask, to_uint8=True)

    def get_su2_mul(self, q1, q2):
        """Quaternion multiplication representing SU(2) group multiplication."""
        a1, b1, c1, d1 = q1.unbind(-1)
        a2, b2, c2, d2 = q2.unbind(-1)
        return torch.stack([
            a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2
        ], dim=-1)

    def get_su2_inv(self, q):
        """Inverse of SU(2) (conjugate of quaternion)."""
        mask = torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
        return q * mask

    def get_loss(self):
        # 1. Normalize links to lie on the SU(2) manifold (unit sphere S3)
        U = F.normalize(self.links, p=2, dim=-1)

        # 2. Compute Plaquettes Up = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
        # For a 2D lattice, there is only one plane (μ=0, ν=1)
        u_x = U[0]
        u_y = U[1]

        u_x_plus_y = torch.roll(u_x, shifts=-1, dims=1)
        u_y_plus_x = torch.roll(u_y, shifts=-1, dims=0)

        # Plaquette calculation
        # p1 = U_x(x,y) * U_y(x+1, y)
        p1 = self.get_su2_mul(u_x, u_y_plus_x)
        # p2 = p1 * U_x_inv(x, y+1)
        p2 = self.get_su2_mul(p1, self.get_su2_inv(u_x_plus_y))
        # plaq = p2 * U_y_inv(x, y)
        plaq = self.get_su2_mul(p2, self.get_su2_inv(u_y))

        # The trace of an SU(2) matrix [a, b, c, d] is 2*a
        tr_plaq = 2 * plaq[..., 0]

        # Standard Wilson Action
        action_density = 1.0 - 0.5 * tr_plaq

        # SOURCE TERM: Force the field to have "Flux" in a specific region
        # This simulates the "Mass Gap" resisting the creation of a field
        source_loss = (action_density * self.source_mask).sum()

        # Total loss = Vacuum minimization + resisting the source
        total_loss = action_density.mean() + 0.1 * source_loss

        if self._make_images:
            # Fix: Use a constant normalization so we don't see microscopic noise
            # Field strength is roughly 0 to 1
            field_viz = plaq[..., 1:]
            self.log_image(name='field_strength', image=field_viz, to_uint8=True)

            # Action map will now show the "Flux Tube" between quarks
            # Clip and scale so it doesn't disappear when loss is low
            action_map = torch.clip(action_density * 10, 0, 1)
            self.log_image(name='action_map', image=action_map, to_uint8=True)

        return total_loss
