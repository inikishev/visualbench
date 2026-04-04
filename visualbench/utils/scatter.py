import cv2
import numpy as np


class TrainingVisualizer:
    def __init__(self, x: np.ndarray, y: np.ndarray, x_vis: np.ndarray, width=800, height=600, margin=50):
        self.width = width
        self.height = height
        self.margin = margin

        # Determine data boundaries for scaling
        self.x_min, self.x_max = x_vis.min(), x_vis.max()
        self.y_min, self.y_max = y.min(), y.max()

        # Pre-compute scaling factors
        # We subtract margin*2 from the drawable area
        self.scale_x = (width - 2 * margin) / (self.x_max - self.x_min)
        self.scale_y = (height - 2 * margin) / (self.y_max - self.y_min)

        # Pre-render the background (static dataset)
        self.background = np.full((height, width, 3), 255, dtype=np.uint8)

        # Convert static points to pixel coordinates
        px = self._to_pixels_x(x)
        py = self._to_pixels_y(y)

        # Draw dataset points (Blue)
        for i in range(len(px)):
            cv2.circle(self.background, (px[i], py[i]), 1, (255, 100, 0), -1, cv2.LINE_AA) # pylint:disable=no-member

    def _to_pixels_x(self, x_vals):
        return (self.margin + (x_vals - self.x_min) * self.scale_x).astype(np.int32)

    def _to_pixels_y(self, y_vals):
        # Y is inverted in image coordinates (0 is top)
        return (self.height - self.margin - (y_vals - self.y_min) * self.scale_y).astype(np.int32)

    def render_frame(self, x_pred, y_pred):
        """
        Generates a frame with the model prediction line.
        x_pred/y_pred: numpy arrays
        """
        # Copy the pre-rendered background
        frame = self.background.copy()

        # Convert prediction to pixel coordinates
        px = self._to_pixels_x(x_pred)
        py = self._to_pixels_y(y_pred)

        # Stack into (N, 1, 2) array for cv2.polylines
        pts = np.stack([px, py], axis=1).reshape((-1, 1, 2))

        # Draw the prediction line (Red)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255),# pylint:disable=no-member
                      thickness=1, lineType=cv2.LINE_AA)# pylint:disable=no-member

        return frame