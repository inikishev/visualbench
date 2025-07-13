import torch
import torch.nn as nn
import imageio
from ..benchmark import Benchmark
import math
from ..utils.torch_tools import to_jet_cmap, to_overflow_cmap

def _create_gaussian_blob(size, center_x, center_y, sigma, magnitude, device):
    """Helper function to create a 2D Gaussian blob."""
    x = torch.linspace(0, 1, size, device=device)
    y = torch.linspace(0, 1, size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    dist_sq = (xx - center_x)**2 + (yy - center_y)**2
    gaussian = magnitude * torch.exp(-dist_sq / (2 * sigma**2))
    return gaussian



class PoissonProblem(Benchmark):
    """
    solves the 2D Poisson equation (∇²u = f) with all kinds of boundary conditions and internal sources/sinks.

    This creates a more challenging and visually interesting optimization problem.

    Args:
        size (int): The height and width of the square grid.
        frames (List[torch.Tensor]): An external list to which visualization
                                     frames (uint8 tensors) will be appended.
        device (str): The torch device to run the computation on.
    """
    def __init__(self, size: int = 128):
        super().__init__()

        if size < 20:
            raise ValueError("Size should be larger for interesting patterns.")

        self.size = size
        self.frames = frames
        self.device = device

        # The optimizable parameters are the grid's interior points.
        # Initialized to zero.
        self.interior = nn.Parameter(torch.zeros(size - 2, size - 2, device=device))

        # --- Define the fixed source term `f` for the Poisson equation ∇²u = f ---
        # This term is not optimized. It defines the problem itself.
        # We create a "hot spot" (source) and a "cold spot" (sink).
        source_term = torch.zeros(size, size, device=device)

        # Add a "hot spot" (positive Gaussian)
        hot_spot = _create_gaussian_blob(
            size, center_x=0.25, center_y=0.3, sigma=0.05, magnitude=80.0, device=device
        )
        source_term += hot_spot

        # Add a "cold spot" (negative Gaussian)
        cold_spot = _create_gaussian_blob(
            size, center_x=0.7, center_y=0.6, sigma=0.07, magnitude=-100.0, device=device
        )
        source_term += cold_spot

        # We only care about the source term in the interior of the domain.
        # Register it as a buffer so it's moved to the correct device with .to()
        self.register_buffer('f_interior', source_term[1:-1, 1:-1])

        # --- Define fixed boundary conditions (BCs) ---
        # We create them once and store them.
        self.u_boundary = torch.zeros(size, size, device=device)
        # Top boundary: Sinusoidal wave
        self.u_boundary[0, :] = torch.sin(torch.linspace(0, 3 * math.pi, size, device=device))
        # Bottom boundary: Another sinusoidal wave
        self.u_boundary[-1, :] = torch.cos(torch.linspace(0, 2 * math.pi, size, device=device)) * 0.8
        # Left and Right boundaries are fixed at 0

        # Define a fixed range for visualization to avoid colormap flickering
        self.vis_min = -1.5
        self.vis_max = 1.5

    def forward(self) -> torch.Tensor:
        """
        Computes the loss and generates a visualization frame.
        """
        # --- 1. Assemble the full grid `u` ---
        # Start with the boundary condition template
        u = self.u_boundary.clone()
        # Place the optimizable interior into the grid
        u[1:-1, 1:-1] = self.interior

        # --- 2. Calculate the loss (Mean Squared Residual of Poisson's Eq) ---
        # Calculate the Laplacian ∇²u for the interior points
        laplacian = (
            u[2:, 1:-1] +      # u(i+1, j)
            u[:-2, 1:-1] +     # u(i-1, j)
            u[1:-1, 2:] +      # u(i, j+1)
            u[1:-1, :-2] -     # u(i, j-1)
            4 * self.interior  # -4 * u(i, j)
        )

        # The residual is the difference between the computed Laplacian and the target source term `f`.
        residual = laplacian - self.f_interior
        loss = torch.mean(residual**2)

        # --- 3. Generate and append visualization frame ---
        if self.frames is not None:
            with torch.no_grad():
                # Normalize the full grid for visualization using a fixed range
                normalized_u = (u - self.vis_min) / (self.vis_max - self.vis_min)

                # Apply the custom colormap
                rgb_float = to_jet_cmap(normalized_u)

                # Convert to uint8 image format [0, 255]
                frame = (rgb_float * 255).to(torch.uint8)
                self.frames.append(frame)

        return loss

# --- Main execution block to demonstrate the objective ---
if __name__ == "__main__":
    # --- Configuration ---
    GRID_SIZE = 128
    LEARNING_RATE = 0.2
    OPTIMIZATION_STEPS = 800 # More steps for a more complex problem
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_FILENAME = "poisson_problem_optimization.gif"

    print(f"Using device: {DEVICE}")

    # --- Setup ---
    visualization_frames = []
    model = PoissonProblem(size=GRID_SIZE, frames=visualization_frames, device=DEVICE)

    # Adam is an excellent choice for this kind of problem.
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
    # Scheduler can help fine-tune the solution in later stages
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # --- Optimization Loop ---
    print("Starting optimization...")
    for i in tqdm(range(OPTIMIZATION_STEPS)):
        def closure():
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            return loss
        loss = optimizer.step(closure)

        if (i + 1) % 100 == 0:
            tqdm.write(
              f"Step {i+1}/{OPTIMIZATION_STEPS}, "
              f"Loss: {loss.item():.6f}, "
            #   f"LR: {scheduler.get_last_lr()[0]:.4f}"
            )

    print("Optimization finished.")

    # --- Save the visualization ---
    if visualization_frames:
        print(f"Saving {len(visualization_frames)} frames to {OUTPUT_FILENAME}...")
        # Convert torch tensors to numpy arrays for imageio
        numpy_frames = [frame.cpu().numpy() for frame in visualization_frames]
        imageio.mimsave(OUTPUT_FILENAME, numpy_frames, fps=60, codec='libx264', quality=8)
        # Using a video codec can be more efficient for many frames
        # For a standard GIF, use: imageio.mimsave(OUTPUT_FILENAME, numpy_frames, fps=60)
        print("Done.")