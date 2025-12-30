# pylint:disable=no-member # makes it shut up about OpenCV
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...benchmark import Benchmark


# function taken from https://github.com/damek/alpha_evolve_problem_B1
def _compute_autoconvolution_values(heights: torch.Tensor, delta_x: float, P: int) -> torch.Tensor:
    """
    Computes the values of the autoconvolution (f*f)(t) at the knot points.
    (f*f)(t) is piecewise linear. Max value occurs at one of these knots.
    Knots are t_m = 2*x_min + m*delta_x for m = 0, ..., 2P.
    Values are [0, delta_x * (H*H)_0, ..., delta_x * (H*H)_{2P-2}, 0].
    (H*H) is the discrete convolution of the height sequence H.
    """
    # Ensure heights is 1D
    if heights.ndim != 1 or heights.shape[0] != P:
        raise ValueError(f"heights tensor must be 1D with length P={P}. Got shape {heights.shape}")

    # Reshape heights for conv1d: (batch_size, C_in, L_in)
    # batch_size=1, C_in=1
    h_signal = heights.view(1, 1, P)

    # The kernel for conv1d to compute (H*H) should be H flipped.
    # weight for conv1d: (C_out, C_in/groups, L_kernel)
    # C_out=1, C_in/groups=1
    h_kernel_flipped = torch.flip(heights, dims=[0]).view(1, 1, P)

    # Compute H*H using conv1d. Padding P-1 results in output length 2P-1.
    # These are (H*H)_0, ..., (H*H)_{2P-2}
    conv_result = F.conv1d(h_signal, h_kernel_flipped, padding=P-1).squeeze() # pylint:disable=not-callable

    # Scale by delta_x
    conv_scaled = delta_x * conv_result

    # Add zeros for (f*f)(t_0) and (f*f)(t_{2P})
    zero = torch.tensor([0.0], device=heights.device, dtype=heights.dtype)
    autoconvolution_knot_values = torch.cat([zero, conv_scaled, zero])

    return autoconvolution_knot_values

# function taken from https://github.com/damek/alpha_evolve_problem_B1
def _projection_simplex_pytorch(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    n_features = v.shape[0]
    if n_features == 0:
        return torch.empty_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv_minus_z = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, n_features + 1, device=v.device)
    cond = u - cssv_minus_z / ind > 0
    true_indices = torch.where(cond)[0]
    rho_idx = true_indices[-1]
    rho = ind[rho_idx]
    theta = cssv_minus_z[rho_idx] / rho
    w = torch.clamp(v - theta, min=0.0)
    return w


# --- Drawing Constants and Helpers ---
BG_COLOR = (20, 20, 20)
AXIS_COLOR = (150, 150, 150)
TEXT_COLOR = (240, 240, 240)
GRID_COLOR = (60, 60, 60)
FUNC_F_COLOR = (100, 150, 255)       # Blue
FUNC_G_COLOR = (255, 100, 150)       # Red
AUTOCONV_COLOR = (100, 255, 150)     # Green
MAX_POINT_COLOR = (80, 220, 255)     # Yellow/Cyan

def _draw_plot_axes(frame, x_range, y_range, plot_rect, title):
    """Helper to draw axes, grid, and title for a plot."""
    px, py, pw, ph = plot_rect
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Draw Title
    cv2.putText(frame, title, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    # Map function to convert data coordinates to pixel coordinates
    def to_pixel(x, y):
        try:
            x_p = int(px + (x - x_min) / (x_max - x_min) * pw)
            y_p = int(py + ph - (y - y_min) / (y_max - y_min) * ph)
            return x_p, y_p
        except ValueError:
            return (px, py)

    # Draw Axes
    x_axis_y = to_pixel(0, 0)[1] if y_min < 0 < y_max else to_pixel(0, y_min)[1]
    y_axis_x = to_pixel(0, 0)[0] if x_min < 0 < x_max else to_pixel(0, x_min)[0]
    cv2.line(frame, (px, x_axis_y), (px + pw, x_axis_y), AXIS_COLOR, 1)
    cv2.line(frame, (y_axis_x, py), (y_axis_x, py + ph), AXIS_COLOR, 1)

    # Draw labels for axis limits
    cv2.putText(frame, f"{x_min:.2f}", (px, x_axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{x_max:.2f}", (px + pw - 40, x_axis_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{y_max:.2f}", (y_axis_x + 5, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{y_min:.2f}", (y_axis_x + 5, py + ph - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return to_pixel

def _make_frame(frame_size, heights, autoconv_values, max_val, max_idx, knots_t, knots_x, f_x_min, f_x_max) -> np.ndarray:
    W, H = frame_size
    frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

    # --- Unpack visualization data ---
    t_at_max = knots_t[max_idx]

    # --- TOP PANE: f(x) and f(t_max - x) ---
    margin = 50
    px, py, pw, ph = (margin, margin, W - 2 * margin, H // 2 - margin - 10)

    # Define plot ranges
    h_min, h_max = min(0, heights.min()), max(0.1, heights.max())
    y_margin = (h_max - h_min) * 0.1
    data_x_range = (f_x_min * 2.5, f_x_max * 2.5)
    data_y_range = (h_min - y_margin, h_max + y_margin)

    # --- Vectorized drawing of the two functions ---
    # Create an array of all horizontal pixel coordinates in the plot area
    px_coords = np.arange(px, px + pw)
    # Map these pixel coordinates back to the function's data coordinates
    x_data = np.linspace(data_x_range[0], data_x_range[1], pw)

    # Function 1: f(x)
    # For each x_data, find the index of the step it belongs to
    indices = np.searchsorted(knots_x, x_data, side='right') - 1
    indices = np.clip(indices, 0, len(heights) - 1)
    y_data_f = heights[indices]

    # Function 2: g(x) = f(t_max - x)
    # Evaluate where we need to sample f
    x_for_g = t_at_max - x_data
    indices_g = np.searchsorted(knots_x, x_for_g, side='right') - 1
    # Handle points outside the original domain of f (where height is 0)
    valid_g = (x_for_g >= f_x_min) & (x_for_g < f_x_max)
    y_data_g = np.zeros_like(x_data)
    y_data_g[valid_g] = heights[np.clip(indices_g[valid_g], 0, len(heights) - 1)]

    # Convert both functions' data y-values to pixel y-coordinates
    y_scale = ph / (data_y_range[1] - data_y_range[0])
    py_f = (py + ph - (y_data_f - data_y_range[0]) * y_scale).astype(np.int32)
    py_g = (py + ph - (y_data_g - data_y_range[0]) * y_scale).astype(np.int32)
    py_f = np.clip(py_f, py, py + ph - 1)
    py_g = np.clip(py_g, py, py + ph - 1)

    # Draw the functions using direct array indexing (this is the fast part)
    frame[py_f, px_coords] = FUNC_F_COLOR
    frame[py_g, px_coords] = FUNC_G_COLOR

    # Draw axes using array slicing (fast)
    x_axis_y = int(py + ph - (-data_y_range[0]) * y_scale)
    y_axis_x = int(px + (-data_x_range[0]) / (data_x_range[1] - data_x_range[0]) * pw)
    if py <= x_axis_y < py + ph: frame[x_axis_y, px:px+pw] = AXIS_COLOR
    if px <= y_axis_x < px + pw: frame[py:py+ph, y_axis_x] = AXIS_COLOR

    # --- BOTTOM PANE: Autoconvolution Plot (f*f)(t) ---
    px2, py2, pw2, ph2 = (margin, H // 2 + 10, W - 2 * margin, H // 2 - margin - 10)

    # Define plot ranges
    data_x_range2 = (knots_t[0], knots_t[-1])
    data_y_range2 = (0, max_val * 1.1 if max_val > 0 else 1.0)

    # Use optimized cv2.polylines
    x_scale2 = pw2 / (data_x_range2[1] - data_x_range2[0])
    y_scale2 = ph2 / (data_y_range2[1] - data_y_range2[0])

    px_t = (px2 + (knots_t - data_x_range2[0]) * x_scale2).astype(np.int32)
    py_v = (py2 + ph2 - (autoconv_values - data_y_range2[0]) * y_scale2).astype(np.int32)
    points = np.vstack((px_t, py_v)).T
    cv2.polylines(frame, [points], isClosed=False, color=AUTOCONV_COLOR, thickness=1) # thickness=1 is fastest

    # Draw axes (fast)
    x_axis_y2 = py2 + ph2
    y_axis_x2 = int(px2 + (-data_x_range2[0]) * x_scale2)
    frame[x_axis_y2, px2:px2+pw2] = AXIS_COLOR
    if px2 <= y_axis_x2 < px2 + pw2: frame[py2:py2+ph2, y_axis_x2] = AXIS_COLOR

    # Draw max point marker with slicing (fast)
    max_px = points[max_idx, 0]
    max_py = points[max_idx, 1]
    # Draw a 5x5 square for the marker
    marker_size = 3
    frame[max_py-marker_size : max_py+marker_size, max_px-marker_size : max_px+marker_size] = MAX_POINT_COLOR

    # Add text (low overhead as it's only called once)
    text = f"Max Val: {max_val:.4f}"
    cv2.putText(frame, text, (W - margin - 200, H - margin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA)

    return frame



class AlphaEvolveB1(Benchmark):
    """Alpha Evolve's B1 problem.

    The objective is to minimize maximum value of autoconvolution of a vector with positive values and of a fixed norm.
    The lowest known value for P=600 is something like 1.502.

    Renders:
        The vector at the top and the autoconvolution at the bottom.
        The vector also shows another vector shifted by the largest index, although a bit janky.

    Code is based on https://github.com/damek/alpha_evolve_problem_B1."""
    def __init__(self, P=600, frame_size=(960, 540)):
        super().__init__()

        self.P = P
        self.heights = nn.Parameter(torch.rand(P, generator=self.rng.torch()))

        f_interval = (-0.25, 0.25) # interval for f(x)
        self.f_x_min, self.f_x_max = f_interval
        self.knots_x = np.linspace(*f_interval, P + 1)
        self.knots_t = np.linspace(2 * self.f_x_min, 2 * self.f_x_max, 2 * P + 1)
        self.delta_x = (self.f_x_max - self.f_x_min) / P
        self.frame_size = frame_size
        self._show_titles_on_video = False

    def get_loss(self):
        heights = self.heights

        autoconv_values = _compute_autoconvolution_values(2*self.P*F.softmax(heights, dim=0), self.delta_x, self.P)

        if self._make_images:
            max_val, max_idx = torch.max(autoconv_values, dim=0)
            loss = max_val.amax()

            frame = _make_frame(
                frame_size=self.frame_size,
                heights=heights.detach().cpu().numpy(),# pylint:disable=not-callable
                autoconv_values=autoconv_values.detach().cpu().numpy(),
                max_val=max_val.item(),
                max_idx=max_idx.item(),
                knots_t=self.knots_t,
                knots_x=self.knots_x,
                f_x_min=self.f_x_min,
                f_x_max=self.f_x_max,
            )
            self.log_image("solution", frame, to_uint8=False)


        else:
            loss = autoconv_values.amax()

        return loss
