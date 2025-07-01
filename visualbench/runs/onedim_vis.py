
# pylint: disable=no-member
import warnings
import math

import cv2
import numpy as np

from ..benchmark import Benchmark
from ..utils.renderer import OpenCVRenderer


class UnivariateVisualizer:
    """
    Visualizer for univariate function optimization. useful for HPO and line searches.

    Example:
    .. code:: py

        def objective(x):
            return (x - 2) ** 2

        vis = UnivariateVisualizer(objective)

        scipy.optimize.minimize_scalar(vis)

        vis.render("minimize_scalar.mp4", fps=5)

    ```
    """
    def __init__(
        self,
        fn,
        width=800,
        height=600,
        padding=0.1,
        num_grid=100,
        margin=60,
        log_scale:bool=False,
    ):
        """_summary_

        Args:
            fn (function): function takes in scalar and returns scalar
            width (int, optional): frame width. Defaults to 800.
            height (int, optional): frame height. Defaults to 600.
            padding (float, optional): expand frame around evaluated points. Defaults to 0.1.
            num_grid (int, optional): number of points on a grid to evaluate and show. Defaults to 100.
            margin (int, optional): idk. Defaults to 60.
        """
        self.fn = fn
        self.history = []
        self.width = width
        self.height = height
        self.padding = padding
        self.num_grid = num_grid
        self.margin = margin
        self.log_scale = log_scale

        # Define some colors (BGR format for OpenCV)
        self.COLOR_BACKGROUND = (255, 255, 255) # White
        self.COLOR_AXES = (50, 50, 50)         # Dark Gray
        self.COLOR_OBJECTIVE_PLOT = (200, 200, 200) # Light Gray
        self.COLOR_HISTORY_POINTS = (0, 0, 255)    # Red
        self.COLOR_VERTICAL_LINE = (255, 100, 100) # Light Blue
        self.COLOR_PIECEWISE_MODEL = (0, 180, 0) # Green
        self.COLOR_TEXT = (0, 0, 0)            # Black
        self.POINT_RADIUS = 5
        self.LINE_THICKNESS = 2

    def evaluate(self, x: float) -> float:
        y = self.fn(x)
        if self.log_scale: y = math.log10(y)
        self.history.append((float(x), float(y)))
        return y

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def _transform_point(self, x, y, x_min_data, x_range_data, y_min_data, y_range_data):
        draw_width = self.width - 2 * self.margin
        draw_height = self.height - 2 * self.margin

        px = self.margin + ((x - x_min_data) / x_range_data) * draw_width
        py = self.margin + ((y_min_data + y_range_data - y) / y_range_data) * draw_height # Y is inverted

        return int(round(px)), int(round(py))

    def render(self, fname: str, fps=2):
        if not self.history:
            raise ValueError("History is empty")

        # 1. Determine overall plotting range based on history and dense evaluation
        hist_x = np.array([p[0] for p in self.history])
        hist_y = np.array([p[1] for p in self.history])

        min_x_hist, max_x_hist = np.min(hist_x), np.max(hist_x)
        x_range_hist = max_x_hist - min_x_hist
        if x_range_hist == 0: x_range_hist = abs(min_x_hist) * 0.2 if min_x_hist != 0 else 1.0 # Handle single point case

        padding_x = x_range_hist * self.padding
        plot_min_x = min_x_hist - padding_x
        plot_max_x = max_x_hist + padding_x
        plot_range_x = plot_max_x - plot_min_x
        if plot_range_x == 0: # Ensure range is not zero
            plot_min_x -= 0.5
            plot_max_x += 0.5
            plot_range_x = 1.0


        # 2. Dense objective evaluation for the background plot
        grid_x_coords = np.linspace(plot_min_x, plot_max_x, self.num_grid)
        try:
            grid_y_coords = np.array([self.fn(x) for x in grid_x_coords])
        except Exception as e:
            warnings.warn(f"Warning: Error evaluating objective for dense plot: {e}")
            # Fallback if objective fails on dense points (e.g. discontinuous, problematic outside history)
            grid_y_coords = np.interp(grid_x_coords, hist_x, hist_y) # simple interp based on history

        if self.log_scale: grid_y_coords = np.log10(grid_y_coords)

        all_y_coords = np.concatenate([hist_y, grid_y_coords])
        min_y_data, max_y_data = np.min(all_y_coords), np.max(all_y_coords)
        y_range_data_overall = max_y_data - min_y_data
        if y_range_data_overall == 0: y_range_data_overall = abs(min_y_data) * 0.2 if min_y_data != 0 else 1.0

        padding_y = y_range_data_overall * self.padding
        plot_min_y = min_y_data - padding_y
        plot_max_y = max_y_data + padding_y
        plot_range_y = plot_max_y - plot_min_y
        if plot_range_y == 0: # Ensure range is not zero
            plot_min_y -= 0.5
            plot_max_y += 0.5
            plot_range_y = 1.0

        # 3. Setup Video Writer
        with OpenCVRenderer(fname, fps) as writer:

            # 4. Animation Loop
            for frame_idx in range(len(self.history)):
                frame = np.full((self.height, self.width, 3), self.COLOR_BACKGROUND, dtype=np.uint8)

                # Draw Axis labels (min/max values)
                cv2.putText(frame, f"x_min: {plot_min_x:.2f}", (self.margin - 20, self.height - self.margin + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
                cv2.putText(frame, f"x_max: {plot_max_x:.2f}", (self.width - self.margin - 70, self.height - self.margin + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
                cv2.putText(frame, f"y_min: {plot_min_y:.2f}", (self.margin - 50, self.height - self.margin + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
                cv2.putText(frame, f"y_max: {plot_max_y:.2f}", (self.margin - 50, self.margin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)

                # Draw "Axes" lines (borders of the plot area)
                # Bottom X-axis like line
                cv2.line(frame, (self.margin, self.height - self.margin), (self.width - self.margin, self.height - self.margin), self.COLOR_AXES, 1)
                # Left Y-axis like line
                cv2.line(frame, (self.margin, self.margin), (self.margin, self.height - self.margin), self.COLOR_AXES, 1)

                # Draw Dense Objective Plot
                if len(grid_x_coords) > 1:
                    for i in range(len(grid_x_coords) - 1):
                        p1_data = (grid_x_coords[i], grid_y_coords[i])
                        p2_data = (grid_x_coords[i+1], grid_y_coords[i+1])

                        # Clip points to be within the plot_min_y and plot_max_y for drawing
                        # This prevents extreme values from making the transform behave badly if they are way out
                        y1_clipped = np.clip(p1_data[1], plot_min_y, plot_max_y)
                        y2_clipped = np.clip(p2_data[1], plot_min_y, plot_max_y)

                        pt1_px = self._transform_point(p1_data[0], y1_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                        pt2_px = self._transform_point(p2_data[0], y2_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                        cv2.line(frame, pt1_px, pt2_px, self.COLOR_OBJECTIVE_PLOT, self.LINE_THICKNESS -1)

                current_eval_points = self.history[:frame_idx + 1]

                # Draw History Points (all up to current frame)
                for i, (hx, hy) in enumerate(current_eval_points):
                    hy_clipped = np.clip(hy, plot_min_y, plot_max_y)
                    px, py = self._transform_point(hx, hy_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                    color = self.COLOR_HISTORY_POINTS
                    if i == frame_idx: # Current (latest) point
                        color = (0, 165, 255) # Orange for current point
                    cv2.circle(frame, (px, py), self.POINT_RADIUS, color, -1)
                    cv2.circle(frame, (px, py), self.POINT_RADIUS, self.COLOR_TEXT, 1) # Outline

                # Draw Vertical Line for the newest point in this frame
                latest_x, latest_y = current_eval_points[-1]
                latest_y_clipped = np.clip(latest_y, plot_min_y, plot_max_y)

                # top_px = self._transform_point(latest_x, plot_max_y, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                # bottom_px = self._transform_point(latest_x, plot_min_y, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                # Ensure x-coordinate is the same for a perfect vertical line
                vline_x = self._transform_point(latest_x, latest_y_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)[0]
                cv2.line(frame, (vline_x, self.margin), (vline_x, self.height - self.margin), self.COLOR_VERTICAL_LINE, 1)

                # Draw Piecewise Model (if >= 2 points)
                if len(current_eval_points) >= 2:
                    # Sort points by x-value for correct piecewise connection
                    sorted_current_history = sorted(current_eval_points, key=lambda p: p[0])
                    for i in range(len(sorted_current_history) - 1):
                        p1_data = sorted_current_history[i]
                        p2_data = sorted_current_history[i+1]

                        y1_clipped = np.clip(p1_data[1], plot_min_y, plot_max_y)
                        y2_clipped = np.clip(p2_data[1], plot_min_y, plot_max_y)

                        pt1_px = self._transform_point(p1_data[0], y1_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                        pt2_px = self._transform_point(p2_data[0], y2_clipped, plot_min_x, plot_range_x, plot_min_y, plot_range_y)
                        cv2.line(frame, pt1_px, pt2_px, self.COLOR_PIECEWISE_MODEL, self.LINE_THICKNESS)

                # Add frame number and current point info
                cv2.putText(frame, f"Step: {frame_idx + 1}/{len(self.history)}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 1)
                cv2.putText(frame, f"Current Eval: ({latest_x:.2f}, {latest_y:.2f})", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 1)

                writer.write(frame)

