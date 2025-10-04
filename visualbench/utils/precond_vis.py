"""Sigma slow"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.linalg import sqrtm
from tqdm import tqdm

from ..tasks.function_descent.test_functions import TEST_FUNCTIONS, TestFunction
from .format import totensor


# A "bouncy" easing function for smooth but distinct transitions
def ease_in_out(t):
    """Quintic ease-in-out function."""
    if t < 0.5:
        return 16 * t**5
    t_ = (t * 2 - 2) / 2
    return 1 + 16 * t_**5

@torch.no_grad
def visualize_preconditioning(
    func,
    points,
    preconditioners,
    minima_loc,
    view_size=5.0,
    interp_frames=30,
    interval=20,
    repeat:bool=True,
    repeat_delay: int = 200,
):
    """
    Animates the optimization process with a side-by-side view of the original
    and preconditioned landscape.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.5))

    # Ensure consistent dtype for all tensor operations
    dtype = points[0].dtype

    # Calculate the coordinate transformation matrices M_t = P_t^(1/2)
    transform_matrices = [
        torch.from_numpy(sqrtm(P.cpu().numpy(force=True)).real).to(dtype) for P in preconditioners
    ]

    # --- FIX 2 (REVISED): Calculate a SINGLE fixed radius that is good on average ---
    # The scale of the right plot is related to the inverse of the transform matrix M.
    # The area scaling is det(inv(M)) = det(P)^(-1/2).
    # The linear scaling is roughly the square root of that, so det(P)^(-1/4).
    # We compute an average scaling factor to adjust the radius of our reference circle.

    # Add a small epsilon to avoid log(0) for singular matrices
    dets = torch.stack([torch.det(P) for P in preconditioners]) + 1e-20

    # Calculate the scaling factor for each step.
    # A small determinant means a large zoom-out on the right, so we need a larger circle.
    scaling_factors = dets.pow(-0.25)

    # Use the geometric mean to find a good average scaling factor.
    log_geomean_scaling = torch.mean(torch.log(scaling_factors))
    avg_scaling_factor = torch.exp(log_geomean_scaling)

    # Set the fixed radius for the entire animation.
    base_radius = view_size / 10
    radius = base_radius * avg_scaling_factor
    # --- End of FIX 2 ---

    num_steps = len(points) - 1
    total_frames = num_steps * interp_frames

    def update(frame_idx):
        # Determine current step and interpolation progress
        step_idx = frame_idx // interp_frames
        interp_progress = (frame_idx % interp_frames) / (interp_frames - 1)
        eased_progress = ease_in_out(interp_progress)

        # Interpolate key values
        pos_start, pos_end = points[step_idx], points[step_idx + 1]
        M_start, M_end = transform_matrices[step_idx], transform_matrices[step_idx + 1]

        current_pos = pos_start + (pos_end - pos_start) * eased_progress
        current_M = M_start + (M_end - M_start) * eased_progress
        inv_current_M = torch.inverse(current_M)

        # --- Clear and setup axes for this frame ---
        for ax, title in zip([ax_left, ax_right], ["Original Space", "Transformed (Preconditioned) Space"]):
            ax.clear()
            ax.set_title(title, fontsize=14)
            ax.set_aspect('equal', 'box')
            ax.grid(True, linestyle='--', alpha=0.5)

        # =======================================================
        # Left Plot: Original Space (centered on `current_pos`)
        # =======================================================

        # 1. Define the viewing window in the original space
        x_min, x_max = current_pos[0] - view_size / 2, current_pos[0] + view_size / 2
        y_min, y_max = current_pos[1] - view_size / 2, current_pos[1] + view_size / 2
        x_lin_orig = np.linspace(x_min.cpu().numpy(force=True), x_max.cpu().numpy(force=True), 100)
        y_lin_orig = np.linspace(y_min.cpu().numpy(force=True), y_max.cpu().numpy(force=True), 100)
        X_grid_orig, Y_grid_orig = np.meshgrid(x_lin_orig, y_lin_orig)

        # 2. Evaluate function directly on this grid
        Z_orig = func(torch.from_numpy(X_grid_orig).to(dtype), torch.from_numpy(Y_grid_orig).to(dtype))

        # 3. Plot contours, trajectory, and markers
        ax_left.contour(X_grid_orig, Y_grid_orig, Z_orig, levels=np.logspace(0, 3.5, 15), cmap='viridis')

        # --- FIX 3: Corrected Trajectory Plotting ---
        line_points = torch.stack(points[:step_idx + 2]).cpu().numpy(force=True)
        ax_left.plot(line_points[:, 0], line_points[:, 1], '-', color='red', alpha=0.6, label='Trajectory')
        marker_points = torch.stack(points[:step_idx + 1]).cpu().numpy(force=True)
        if marker_points.shape[0] > 0:
            ax_left.plot(marker_points[:, 0], marker_points[:, 1], 'o', color='red', alpha=0.6)
        # --- End of FIX 3 ---

        ax_left.plot(*minima_loc.cpu().numpy(force=True), 'g*', markersize=15, label='Minimum')
        ax_left.plot(*current_pos.cpu().numpy(force=True), 'ro', markersize=8, label='Current Position')

        # 4. Set final camera limits for the left plot
        ax_left.set_xlim(x_min.cpu().numpy(force=True), x_max.cpu().numpy(force=True))
        ax_left.set_ylim(y_min.cpu().numpy(force=True), y_max.cpu().numpy(force=True))


        # ======================================================================
        # Right Plot: Transformed Space
        # ======================================================================

        # 1. Transform key points into the new coordinate system
        current_pos_transformed = inv_current_M @ current_pos
        minima_transformed = inv_current_M @ minima_loc

        # 2. Transform the corners of the left plot's view to determine the view for the right plot.
        corners_orig = torch.tensor([
            [x_min, x_max, x_max, x_min],
            [y_min, y_min, y_max, y_max]
        ], dtype=dtype)
        corners_transformed = inv_current_M @ corners_orig
        u_min, u_max = corners_transformed[0].min(), corners_transformed[0].max()
        v_min, v_max = corners_transformed[1].min(), corners_transformed[1].max()

        # --- FIX 1: Make Right Frame Square ---
        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        u_range = u_max - u_min
        v_range = v_max - v_min
        max_range = max(u_range.item(), v_range.item())

        u_min = u_center - max_range / 2
        u_max = u_center + max_range / 2
        v_min = v_center - max_range / 2
        v_max = v_center + max_range / 2
        # --- End of FIX 1 ---

        # 3. Create a NEW grid in the transformed space
        u_lin_trans = np.linspace(u_min.cpu().numpy(force=True), u_max.cpu().numpy(force=True), 100)
        v_lin_trans = np.linspace(v_min.cpu().numpy(force=True), v_max.cpu().numpy(force=True), 100)
        U_grid_trans, V_grid_trans = np.meshgrid(u_lin_trans, v_lin_trans)

        # 4. Map this grid BACK to the original space to evaluate the function
        grid_coords_trans = torch.stack([
            torch.from_numpy(U_grid_trans.flatten()),
            torch.from_numpy(V_grid_trans.flatten())
        ]).to(dtype)
        grid_coords_orig_for_eval = current_M @ grid_coords_trans
        X_eval = grid_coords_orig_for_eval[0].reshape(U_grid_trans.shape)
        Y_eval = grid_coords_orig_for_eval[1].reshape(U_grid_trans.shape)
        Z_trans = func(X_eval, Y_eval)

        # 5. Plot contours, trajectory, and markers
        ax_right.contour(U_grid_trans, V_grid_trans, Z_trans, levels=np.logspace(0, 3.5, 15), cmap='viridis')

        hist_points_orig = torch.stack(points[:step_idx + 2])
        hist_points_transformed_line = (inv_current_M @ hist_points_orig.T).T.cpu().numpy(force=True)
        ax_right.plot(hist_points_transformed_line[:, 0], hist_points_transformed_line[:, 1], '-', color='red', alpha=0.6)

        hist_points_markers_orig = torch.stack(points[:step_idx + 1])
        if hist_points_markers_orig.shape[0] > 0:
            hist_points_transformed_markers = (inv_current_M @ hist_points_markers_orig.T).T.cpu().numpy(force=True)
            ax_right.plot(hist_points_transformed_markers[:, 0], hist_points_transformed_markers[:, 1], 'o', color='red', alpha=0.6)

        ax_right.plot(*minima_transformed.cpu().numpy(force=True), 'g*', markersize=15)
        ax_right.plot(*current_pos_transformed.cpu().numpy(force=True), 'ro', markersize=8)

        # ======================================================================
        # Plot Reference Shapes (Ellipse on Left, Circle on Right)
        # ======================================================================

        # 1. Define a circle in the transformed coordinate system (relative to origin) using the pre-calculated fixed radius
        theta = torch.linspace(0, 2 * np.pi, 100, dtype=dtype)
        circ_points_relative_trans = torch.stack([
            torch.cos(theta) * radius,
            torch.sin(theta) * radius
        ])

        # 2. Center it at the current position and plot on the RIGHT (preconditioned space)
        circ_points_absolute_trans = current_pos_transformed.view(2, 1) + circ_points_relative_trans
        ax_right.plot(
            circ_points_absolute_trans[0, :].cpu().numpy(force=True),
            circ_points_absolute_trans[1, :].cpu().numpy(force=True),
            color='blue', lw=2, linestyle='--'
        )

        # 3. Transform the circle's points back to original space to get an ellipse and plot it on the LEFT.
        ellipse_points_orig = (current_M @ circ_points_absolute_trans).cpu().numpy(force=True)
        ax_left.plot(
            ellipse_points_orig[0, :],
            ellipse_points_orig[1, :],
            color='blue', lw=2, linestyle='--', label='Preconditioner Shape'
        )
        ax_left.legend(loc="upper left")

        # 6. Set final camera limits for the right plot
        ax_right.set_xlim(u_min.cpu().numpy(force=True), u_max.cpu().numpy(force=True))
        ax_right.set_ylim(v_min.cpu().numpy(force=True), v_max.cpu().numpy(force=True))

        fig.suptitle(f"Optimization Step {step_idx + 1}/{num_steps+1} (Frame {frame_idx+1}/{total_frames})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Create and save the animation
    ani = FuncAnimation(fig, update, frames=total_frames, interval=interval, repeat=repeat, repeat_delay=repeat_delay)
    plt.close(fig) # Prevent duplicate plot display in some environments
    return ani

class Preconditioner(ABC):

    @abstractmethod
    def step(self, x: torch.Tensor, g: torch.Tensor, H:torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """returns new x and current linear transformation matrix"""


class Adagrad(Preconditioner):
    def __init__(self, lr=1e-3, eps=1e-12):
        self.lr = lr
        self.eps = eps
        self.accumulator = torch.zeros(2)

    def step(self, x, g, H):
        self.accumulator.addcmul_(g, g)

        P = self.accumulator.sqrt().add(self.eps).reciprocal()
        dir = g * P

        return x.sub(dir, alpha=self.lr), P.diag_embed()


class Adam(Preconditioner):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-12):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.exp_avg = torch.zeros(2)
        self.exp_avg_sq = torch.zeros(2)
        self.current_step = 1

    def step(self, x, g, H):
        self.exp_avg.lerp_(g, 1-self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(g, g, value=1-self.beta2)

        exp_avg = self.exp_avg / (1 - self.beta1**self.current_step)
        exp_avg_sq = self.exp_avg_sq / (1 - self.beta2**self.current_step)

        P = exp_avg_sq.sqrt().add(self.eps).reciprocal()
        dir = exp_avg * P

        return x.sub(dir, alpha=self.lr), P.diag_embed()

class FullMatrixAdam(Preconditioner):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-12):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.exp_avg = torch.zeros(2)
        self.covariance = torch.eye(2) * (1 - beta2)
        self.current_step = 1

    def step(self, x, g, H):
        self.exp_avg.lerp_(g, 1-self.beta1)
        self.covariance.lerp_(g.outer(g), 1-self.beta2)

        exp_avg = self.exp_avg / (1 - self.beta1**self.current_step)
        covariance = self.covariance / (1 - self.beta2**self.current_step)

        L, Q = torch.linalg.eigh(covariance) # pylint:disable=not-callable

        P = Q @ L.sqrt().add(self.eps).reciprocal().diag_embed() @ Q.T
        dir = P @ exp_avg

        return x.sub(dir, alpha=self.lr), P

class AdamInFullMatrixAdam(Preconditioner):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.95, matrix_beta=0.95, eps=1e-12):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.matrix_beta = matrix_beta
        self.eps = eps
        self.exp_avg = torch.zeros(2)
        self.exp_avg_sq = torch.zeros(2)
        self.covariance = torch.eye(2) * (1 - matrix_beta)
        self.current_step = 1

    def step(self, x, g, H):
        self.covariance.lerp_(g.outer(g), 1-self.beta2)
        covariance = self.covariance / (1 - self.beta2**self.current_step)
        L, Q = torch.linalg.eigh(covariance) # pylint:disable=not-callable

        g_proj = Q.T @ g

        self.exp_avg.lerp_(g_proj, 1-self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(g_proj, g_proj, value=1-self.beta1)

        exp_avg = self.exp_avg / (1 - self.beta1**self.current_step)
        exp_avg_sq = self.exp_avg_sq / (1 - self.beta2**self.current_step)

        P_proj = exp_avg_sq.sqrt().add(self.eps).reciprocal()
        dir_proj = exp_avg * P_proj

        dir = Q @ dir_proj

        return x.sub(dir, alpha=self.lr), Q @ P_proj

class Newton(Preconditioner):
    def __init__(self, lr=1e-3, beta=0, sfn:bool=False):
        self.lr = lr
        self.beta = beta
        self.H = torch.zeros(0,0)
        self.sfn = sfn

    def step(self, x, g, H):
        assert H is not None
        self.H.lerp_(H, 1-self.beta)

        if self.sfn:
            L, Q = torch.linalg.eigh(self.H) # pylint:disable=not-callable
            P = Q @ L.abs().reciprocal() @ Q.T

        else:
            P = torch.linalg.inv(H) # pylint:disable=not-callable

        dir = P @ g

        return x.sub(dir, alpha=self.lr), P


class PreconditionerVisualizer:
    def __init__(self, func, x0=None, dtype=torch.float32, minima=None):
        if isinstance(func, str): f = TEST_FUNCTIONS[func].to(device = 'cpu', dtype = dtype)
        else: f = func

        if isinstance(f, TestFunction):
            if x0 is None: x0 = f.x0()
            if minima is None: minima = f.minima()

        else:
            assert x0 is not None

        self.func = f
        self.x0 = totensor(x0, dtype=dtype)
        self.minima = minima

        self.points = [self.x0]
        self.matrices = [torch.eye(2)]

    @torch.no_grad
    def run(self, opt: Preconditioner, steps: int, hessian:bool=False):
        x = self.x0.requires_grad_(True)
        fx = self.func(*x)

        for step in range(steps):
            with torch.enable_grad():
                fx = self.func(*x)
                Hx = None
                if hessian:
                    gx = torch.autograd.grad(fx, x, create_graph=True)[0]
                    Hx = torch.autograd.grad(gx, x, torch.eye(2), is_grads_batched=True)[0]
                else:
                    gx = torch.autograd.grad(fx, x)[0]

            x_new, P = opt.step(x, gx, Hx)
            self.points.append(x_new.detach().clone())
            self.matrices.append(P.detach().clone())

            x.copy_(x_new)

        return fx

    def make_animation(
        self,
        fov,
        fps=20,
        interp_frames=5,
        repeat:bool=True,
        repeat_delay: int = 200,
    ):
        return visualize_preconditioning(
            func=self.func,
            points=self.points,
            preconditioners=self.matrices,
            minima_loc=totensor(self.minima).float() if self.minima is not None else totensor((0,0)).float(),
            view_size=fov,
            interp_frames=interp_frames,
            interval=int(1000/fps),
            repeat=repeat,
            repeat_delay=repeat_delay,
        )


# if __name__ == "__main__":
#     from tqdm import tqdm
#     from visualbench.utils.precond_vis import PreconditionerVisualizer, Adagrad

#     adagrad = Adagrad(2)
#     func = PreconditionerVisualizer("stretched")
#     func.run(adagrad, 200)
#     anim = func.make_animation(fov=50)

#     print("SAVING")
#     pbar = tqdm()
#     anim.save('adagrad s.mp4', fps=30, progress_callback=lambda _i, _n: pbar.update(1))