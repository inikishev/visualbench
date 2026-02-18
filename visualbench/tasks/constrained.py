"""Classical constrained engineering design optimization problems."""
import torch
from torch import nn

from ..benchmark import Benchmark


class PressureVesselDesign(Benchmark):
    """Pressure Vessel Design optimization problem.

    **Objective**: Minimize the total cost (material, forming and welding) of a cylindrical pressure vessel.

    **Design Variables** (4):
        - x1: Shell thickness (Ts) - discrete multiples of 0.0625
        - x2: Head thickness (Th) - discrete multiples of 0.0625
        - x3: Inner radius (R)
        - x4: Length of cylindrical section (L)

    **Constraints** (4):
        - g1: Shell thickness constraint (hoop stress): -x1 + 0.0193*x3 ≤ 0
        - g2: Head thickness constraint: -x2 + 0.00954*x3 ≤ 0
        - g3: Volume constraint (minimum 1,300,000 cubic inches)
        - g4: Length constraint (maximum 240 inches)

    **Bounds**:
        - 1 ≤ x1, x2 ≤ 99 (discrete, but optimized as continuous)
        - 10 ≤ x3 ≤ 200
        - 10 ≤ x4 ≤ 200

    **Known Global Optimum**: f* ≈ 6059.71 at x* ≈ (0.8125, 0.4375, 42.0984, 176.6389)

    **Visualization**:
        - Plots convergence of objective and constraint violations
        - Shows evolution of design variables

    Args:
        penalty_weight (float): Penalty weight for constraint violations. Default: 1e6

    Example:
        >>> bench = PressureVesselDesign()
        >>> opt = torch.optim.Adam(bench.parameters(), lr=1e-2)
        >>> bench.run(opt, max_steps=1000)
        >>> bench.plot()
    """

    def __init__(self, penalty_weight: float = 1e6):
        super().__init__(seed=0)

        # Initialize design variables with reasonable starting point
        # [Ts, Th, R, L]
        self.x = nn.Parameter(torch.tensor([1.0, 1.0, 50.0, 100.0]))

        # Bounds (use nn.Buffer for CUDA compatibility)
        self.register_buffer("lower_bounds", torch.tensor([1.0, 1.0, 10.0, 10.0]))
        self.register_buffer("upper_bounds", torch.tensor([99.0, 99.0, 200.0, 200.0]))

        self.penalty_weight = penalty_weight

        # Store optimal solution for reference
        self.optimal_value = 6059.71
        self.register_buffer("optimal_solution", torch.tensor([0.8125, 0.4375, 42.0984, 176.6389]))

    def _constraint_violations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations.

        Constraints (all should be ≤ 0):
            g1: -x1 + 0.0193*x3 ≤ 0  (shell thickness)
            g2: -x2 + 0.00954*x3 ≤ 0  (head thickness)
            g3: -π*x3²*x4 - (4/3)*π*x3³ + 1,300,000 ≤ 0  (volume)
            g4: x4 - 240 ≤ 0  (length)
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        g1 = -x1 + 0.0193 * x3
        g2 = -x2 + 0.00954 * x3
        g3 = -torch.pi * x3**2 * x4 - (4.0 / 3.0) * torch.pi * x3**3 + 1_300_000
        g4 = x4 - 240

        return torch.stack([g1, g2, g3, g4])

    def _penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for constraint violations."""
        violations = self._constraint_violations(x)
        # Only penalize positive violations (constraints are g ≤ 0)
        positive_violations = torch.relu(violations)
        return self.penalty_weight * positive_violations.sum()

    def _objective(self, x: torch.Tensor) -> torch.Tensor:
        """Compute objective function (total cost).

        f(x) = 0.6224*x1*x3*x4 + 1.7781*x2*x3² + 3.1661*x1²*x4 + 19.84*x1²*x3
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        term1 = 0.6224 * x1 * x3 * x4
        term2 = 1.7781 * x2 * x3**2
        term3 = 3.1661 * x1**2 * x4
        term4 = 19.84 * x1**2 * x3

        return term1 + term2 + term3 + term4

    def _bounds_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for bound violations (vectorized)."""
        lower_violation = torch.relu(self.lower_bounds - x).square().sum()
        upper_violation = torch.relu(x - self.upper_bounds).square().sum()
        return lower_violation + upper_violation

    def get_loss(self):
        # Compute objective
        obj = self._objective(self.x)

        # Compute constraint penalty
        constraint_penalty = self._penalty(self.x)

        # Compute bounds penalty
        bounds_penalty = self._bounds_penalty(self.x)

        # Log metrics
        self.log("cost", obj)
        self.log("constraint violation", constraint_penalty / self.penalty_weight)
        self.log("total loss", obj + constraint_penalty + bounds_penalty)

        # Log distance to optimal solution
        with torch.no_grad():
            dist_to_optimal = torch.norm(self.x - self.optimal_solution)
            self.log("distance to optimum", dist_to_optimal)

        return obj + constraint_penalty + bounds_penalty


class WeldedBeamDesign(Benchmark):
    """Welded Beam Design optimization problem.

    **Objective**: Minimize the fabrication cost of a welded beam.

    **Design Variables** (4):
        - x1: Weld thickness (h)
        - x2: Length of attached part of bar (l)
        - x3: Height of bar (t)
        - x4: Thickness of bar (b)

    **Constraints** (7):
        - g1: Shear stress constraint (τ ≤ 13600 psi)
        - g2: Bending stress constraint (σ ≤ 30000 psi)
        - g3: Geometric constraint (h ≤ b)
        - g4: Side constraint on design variables
        - g5: Deflection constraint (δ ≤ 0.25 inches)
        - g6: Buckling load constraint (P ≤ P_c)
        - g7: Minimum weld thickness (h ≥ 0.125)

    **Bounds**:
        - 0.125 ≤ x1 ≤ 2.0 (weld thickness, minimum enforced by constraint)
        - 0.1 ≤ x2, x3 ≤ 10.0
        - 0.1 ≤ x4 ≤ 2.0

    **Known Global Optimum**: f* ≈ 2.38 at x* ≈ (0.2057, 3.4705, 9.0366, 0.2057)

    **Visualization**:
        - Plots convergence of objective and constraint violations
        - Shows evolution of design variables

    Args:
        penalty_weight (float): Penalty weight for constraint violations. Default: 1e6

    Example:
        >>> bench = WeldedBeamDesign()
        >>> opt = torch.optim.Adam(bench.parameters(), lr=1e-2)
        >>> bench.run(opt, max_steps=1000)
        >>> bench.plot()
    """

    def __init__(self, penalty_weight: float = 1e6):
        super().__init__(seed=0)

        # Constants
        self.P = 6000.0  # Load (lb)
        self.L = 14.0  # Length of beam (inches)
        self.E = 30e6  # Young's modulus (psi)
        self.G = 12e6  # Shear modulus (psi)
        self.tau_max = 13600.0  # Max shear stress (psi)
        self.sigma_max = 30000.0  # Max bending stress (psi)
        self.delta_max = 0.25  # Max deflection (inches)

        # Initialize design variables [h, l, t, b]
        self.x = nn.Parameter(torch.tensor([0.5, 5.0, 5.0, 0.5]))

        # Bounds (use nn.Buffer for CUDA compatibility)
        # Note: x1 lower bound is 0.125 due to constraint g5 (h >= 0.125)
        self.register_buffer("lower_bounds", torch.tensor([0.125, 0.1, 0.1, 0.1]))
        self.register_buffer("upper_bounds", torch.tensor([2.0, 10.0, 10.0, 2.0]))

        self.penalty_weight = penalty_weight

        # Store optimal solution for reference
        self.optimal_value = 2.38
        self.register_buffer("optimal_solution", torch.tensor([0.2057, 3.4705, 9.0366, 0.2057]))

    def _constraint_violations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations.

        All constraints should be ≤ 0.
        """
        h, l, t, b = x[0], x[1], x[2], x[3]

        # Intermediate calculations
        M = self.P * (self.L + l / 2.0)  # Moment
        R = torch.sqrt(l**2 / 4.0 + (h + t)**2 / 4.0)  # Radius
        J = 2.0 * torch.sqrt(torch.tensor(2.0)) * h * l * (l**2 / 12.0 + (h + t)**2 / 4.0)  # Polar moment

        # Shear stress components
        tau1 = self.P / (torch.sqrt(torch.tensor(2.0)) * h * l)
        tau2 = M * R / J
        tau = torch.sqrt(tau1**2 + tau1 * tau2 * l / R + tau2**2)

        # Bending stress
        sigma = 6.0 * self.P * self.L / (t**2 * b)

        # Deflection
        delta = 4.0 * self.P * self.L**3 / (self.E * t**3 * b)

        # Buckling load
        P_c = 4.013 * self.E * torch.sqrt(t**2 * b**6 / 36.0) / self.L**2 * (
            1.0 - t / (2.0 * self.L) * torch.sqrt(torch.tensor(self.E / (4.0 * self.G)))
        )

        # Constraints (all should be ≤ 0)
        g1 = tau - self.tau_max  # Shear stress
        g2 = sigma - self.sigma_max  # Bending stress
        g3 = h - b  # Geometric: h ≤ b
        g4 = 0.10471 * h**2 + 0.04811 * t * b * (4 + l) - 5.0  # Side constraint
        g5 = 0.125 - h  # Minimum weld thickness
        g6 = delta - self.delta_max  # Deflection
        g7 = self.P - P_c  # Buckling load

        return torch.stack([g1, g2, g3, g4, g5, g6, g7])

    def _penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for constraint violations."""
        violations = self._constraint_violations(x)
        positive_violations = torch.relu(violations)
        return self.penalty_weight * positive_violations.sum()

    def _objective(self, x: torch.Tensor) -> torch.Tensor:
        """Compute objective function (fabrication cost).

        f(x) = 1.10471*h²*l + 0.04811*t*b*(4 + l)
        """
        h, l, t, b = x[0], x[1], x[2], x[3]
        return 1.10471 * h**2 * l + 0.04811 * t * b * (4 + l)

    def _bounds_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for bound violations (vectorized)."""
        lower_violation = torch.relu(self.lower_bounds - x).square().sum()
        upper_violation = torch.relu(x - self.upper_bounds).square().sum()
        return lower_violation + upper_violation

    def get_loss(self):
        # Compute objective
        obj = self._objective(self.x)

        # Compute constraint penalty
        constraint_penalty = self._penalty(self.x)

        # Compute bounds penalty
        bounds_penalty = self._bounds_penalty(self.x)

        # Log metrics
        self.log("cost", obj)
        self.log("constraint violation", constraint_penalty / self.penalty_weight)
        self.log("total loss", obj + constraint_penalty + bounds_penalty)

        # Log distance to optimal solution
        with torch.no_grad():
            dist_to_optimal = torch.norm(self.x - self.optimal_solution)
            self.log("distance to optimum", dist_to_optimal)

        return obj + constraint_penalty + bounds_penalty


class SpringDesign(Benchmark):
    """Tension/Compression Spring Design optimization problem.

    **Objective**: Minimize the weight of a tension/compression spring.

    **Design Variables** (3):
        - x1: Wire diameter (d)
        - x2: Mean coil diameter (D)
        - x3: Number of active coils (N)

    **Constraints** (5):
        - g1: Minimum deflection constraint
        - g2: Shear stress constraint
        - g3: Surge frequency constraint
        - g4: Outer diameter constraint (D + d ≤ 1.5)
        - g5: Geometric constraint (N ≥ 3)

    **Bounds**:
        - 0.05 ≤ x1 ≤ 2.0
        - 0.25 ≤ x2 ≤ 1.3
        - 2.0 ≤ x3 ≤ 15.0

    **Known Global Optimum**: f* ≈ 0.012665 at x* ≈ (0.0517, 0.3567, 11.2889)

    **Visualization**:
        - Plots convergence of objective and constraint violations
        - Shows evolution of design variables

    Args:
        penalty_weight (float): Penalty weight for constraint violations. Default: 1e6

    Example:
        >>> bench = SpringDesign()
        >>> opt = torch.optim.Adam(bench.parameters(), lr=1e-2)
        >>> bench.run(opt, max_steps=1000)
        >>> bench.plot()
    """

    def __init__(self, penalty_weight: float = 1e6):
        super().__init__(seed=0)

        # Initialize design variables [d, D, N]
        self.x = nn.Parameter(torch.tensor([0.1, 0.5, 5.0]))

        # Bounds (use nn.Buffer for CUDA compatibility)
        self.register_buffer("lower_bounds", torch.tensor([0.05, 0.25, 2.0]))
        self.register_buffer("upper_bounds", torch.tensor([2.0, 1.3, 15.0]))

        self.penalty_weight = penalty_weight

        # Store optimal solution for reference
        self.optimal_value = 0.012665
        self.register_buffer("optimal_solution", torch.tensor([0.0517, 0.3567, 11.2889]))

    def _constraint_violations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations.

        All constraints should be ≤ 0.

        Standard spring design constraints from literature:
            g1: 1 - (D^3 * N) / (71785 * d^4) ≤ 0  (minimum deflection)
            g2: (4*D^2 - d*D)/(12566*D*d^3 - d^4) + 1/(5108*d^2) - 1 ≤ 0  (shear stress)
            g3: 1 - (140.45 * d) / (D^2 * N) ≤ 0  (surge frequency)
            g4: (d + D) / 1.5 - 1 ≤ 0  (outer diameter)
            g5: N - 15 ≤ 0  (max coils, handled by bounds)
        """
        d, D, N = x[0], x[1], x[2]

        # Constraints (all should be ≤ 0)
        # g1: Minimum deflection constraint
        g1 = 1.0 - (D**3 * N) / (71785.0 * d**4)

        # g2: Shear stress constraint
        numerator = 4.0 * D**2 - d * D
        denominator = 12566.0 * D * d**3 - d**4
        g2 = numerator / denominator + 1.0 / (5108.0 * d**2) - 1.0

        # g3: Surge frequency constraint
        g3 = 1.0 - (140.45 * d) / (D**2 * N)

        # g4: Outer diameter constraint
        g4 = (d + D) / 1.5 - 1.0

        return torch.stack([g1, g2, g3, g4])

    def _penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for constraint violations."""
        violations = self._constraint_violations(x)
        positive_violations = torch.relu(violations)
        return self.penalty_weight * positive_violations.sum()

    def _objective(self, x: torch.Tensor) -> torch.Tensor:
        """Compute objective function (spring weight).

        f(x) = (N + 2) * D * d²
        """
        d, D, N = x[0], x[1], x[2]
        return (N + 2) * D * d**2

    def _bounds_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute penalty for bound violations (vectorized)."""
        lower_violation = torch.relu(self.lower_bounds - x).square().sum()
        upper_violation = torch.relu(x - self.upper_bounds).square().sum()
        return lower_violation + upper_violation

    def get_loss(self):
        # Compute objective
        obj = self._objective(self.x)

        # Compute constraint penalty
        constraint_penalty = self._penalty(self.x)

        # Compute bounds penalty
        bounds_penalty = self._bounds_penalty(self.x)

        # Log metrics
        self.log("weight", obj)
        self.log("constraint violation", constraint_penalty / self.penalty_weight)
        self.log("total loss", obj + constraint_penalty + bounds_penalty)

        # Log distance to optimal solution
        with torch.no_grad():
            dist_to_optimal = torch.norm(self.x - self.optimal_solution)
            self.log("distance to optimum", dist_to_optimal)

        return obj + constraint_penalty + bounds_penalty
