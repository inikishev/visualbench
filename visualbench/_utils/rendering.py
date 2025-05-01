import cv2
import numpy as np
import matplotlib.pyplot as plt

def array_to_image_cv2(
    array: np.ndarray,
    cmap_name: str = 'coolwarm',
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_position: str = 'right', # 'right' or 'bottom'
    colorbar_thickness_fraction: float = 0.05, # Fraction of image width/height
    colorbar_padding: int = 10, # Pixels between image and colorbar
    num_ticks: int = 5,
    label_precision: int = 2,
    font_scale_factor: float = 0.5 # Adjust for text size relative to thickness
) -> np.ndarray:
    """
    Imshow but way faster.

    Args:
        array: The 2D input NumPy array.
        cmap_name: Name of the Matplotlib colormap (e.g., 'viridis', 'jet').
        vmin: Minimum value for colormap scaling. If None, uses array.min().
        vmax: Maximum value for colormap scaling. If None, uses array.max().
        colorbar_position: Position of the colorbar ('right' or 'bottom').
        colorbar_thickness_fraction: Fraction of the main image dimension
                                     (width for 'right', height for 'bottom')
                                     used for the colorbar's thickness.
        colorbar_padding: Space in pixels between the main image and the colorbar.
        num_ticks: Approximate number of ticks on the colorbar.
        label_precision: Number of decimal places for tick labels.
        font_scale_factor: Multiplier to adjust font size relative to colorbar
                           thickness. Smaller values make text smaller.

    Returns:
        A NumPy array representing the BGR image with the colorbar,
        or None if input is invalid. Returns uint8 array [0-255].
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Error: Input 'array' must be a 2D NumPy array.")

    if colorbar_position not in ['right', 'bottom']:
        raise ValueError("Error: 'colorbar_position' must be 'right' or 'bottom'.")

    # --- 1. Data Scaling and Colormapping ---
    h, w = array.shape
    _vmin = vmin if vmin is not None else np.min(array)
    _vmax = vmax if vmax is not None else np.max(array)

    # Handle case where vmin == vmax
    if _vmin == _vmax:
        _vmax += 1e-6 # Add a tiny epsilon to avoid division by zero

    # Normalize array to [0, 1]
    normalized_array = (array - _vmin) / (_vmax - _vmin)
    normalized_array = np.clip(normalized_array, 0, 1)

    # Get the colormap
    cmap = plt.colormaps[cmap_name]

    # Apply colormap (returns RGBA float 0-1)
    colored_array_rgba = cmap(normalized_array)

    # Convert to RGB uint8 [0-255]
    colored_array_rgb_uint8 = (colored_array_rgba[:, :, :3] * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    main_image_bgr = cv2.cvtColor(colored_array_rgb_uint8, cv2.COLOR_RGB2BGR)

    # --- 2. Create Colorbar ---
    if colorbar_position == 'right':
        cb_h = h
        cb_w = max(1, int(w * colorbar_thickness_fraction))
        # Gradient goes from top (vmax) to bottom (vmin) -> [1, 0]
        gradient = np.linspace(1, 0, cb_h)
        cb_normalized_array = np.tile(gradient[:, np.newaxis], (1, cb_w))
    else: # bottom
        cb_h = max(1, int(h * colorbar_thickness_fraction))
        cb_w = w
        # Gradient goes from left (vmin) to right (vmax) -> [0, 1]
        gradient = np.linspace(0, 1, cb_w)
        cb_normalized_array = np.tile(gradient[np.newaxis, :], (cb_h, 1))

    # Apply colormap to gradient
    cb_rgba = cmap(cb_normalized_array)
    cb_rgb_uint8 = (cb_rgba[:, :, :3] * 255).astype(np.uint8)
    colorbar_bgr = cv2.cvtColor(cb_rgb_uint8, cv2.COLOR_RGB2BGR)

    # --- 3. Add Ticks and Labels to Colorbar ---
    # Create a slightly larger canvas for the colorbar to draw ticks/text outside gradient
    tick_length = 5
    text_color = (255, 255, 255) # White text
    line_color = (255, 255, 255) # White lines
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    # Adjust font scale based on thickness, ensuring it's not too small/large
    font_scale = max(0.2, min(cb_w, cb_h) * 0.05 * font_scale_factor)
    font_thickness = 1

    # Estimate maximum label size to calculate padding needed
    max_label = f"{max(abs(_vmin), abs(_vmax)):.{label_precision}f}"
    (label_w, label_h), _ = cv2.getTextSize(max_label, font_face, font_scale, font_thickness)

    if colorbar_position == 'right':
        cb_canvas_w = cb_w + tick_length + label_w + 5 # 5px extra padding
        cb_canvas_h = cb_h
        cb_canvas = np.zeros((cb_canvas_h, cb_canvas_w, 3), dtype=np.uint8)
        # Place gradient on the left of the canvas
        cb_canvas[:, 0:cb_w] = colorbar_bgr
        gradient_origin_x = 0
        gradient_origin_y = 0
        tick_start_x = cb_w
        tick_end_x = cb_w + tick_length
        text_start_x = tick_end_x + 3 # 3px padding
    else: # bottom
        cb_canvas_w = cb_w
        cb_canvas_h = cb_h + tick_length + label_h + 5 # 5px extra padding
        cb_canvas = np.zeros((cb_canvas_h, cb_canvas_w, 3), dtype=np.uint8)
        # Place gradient at the top of the canvas
        cb_canvas[0:cb_h, :] = colorbar_bgr
        gradient_origin_x = 0
        gradient_origin_y = 0
        tick_start_y = cb_h
        tick_end_y = cb_h + tick_length
        text_start_y = tick_end_y + label_h + 2 # 2px padding


    # Add ticks and labels
    tick_values = np.linspace(_vmin, _vmax, num_ticks)
    tick_positions_norm = np.linspace(0, 1, num_ticks) # Position along the gradient [0,1]

    for val, pos_norm in zip(tick_values, tick_positions_norm):
        label = f"{val:.{label_precision}f}"
        (tw, th), _ = cv2.getTextSize(label, font_face, font_scale, font_thickness)

        if colorbar_position == 'right':
            # Y position needs to be inverted because gradient goes 1->0 top->bottom
            y_pos = int(gradient_origin_y + (1 - pos_norm) * (cb_h - 1))
            cv2.line(cb_canvas, (tick_start_x, y_pos), (tick_end_x, y_pos), line_color, font_thickness)
            cv2.putText(cb_canvas, label, (text_start_x, y_pos + th // 2), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        else: # bottom
            x_pos = int(gradient_origin_x + pos_norm * (cb_w - 1))
            cv2.line(cb_canvas, (x_pos, tick_start_y), (x_pos, tick_end_y), line_color, font_thickness)
            # Center text below tick
            cv2.putText(cb_canvas, label, (x_pos - tw // 2, text_start_y), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # --- 4. Combine Image and Colorbar Canvas ---
    # Create the final canvas
    if colorbar_position == 'right':
        final_w = w + colorbar_padding + cb_canvas_w
        final_h = h
        final_image = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        # Place main image
        final_image[0:h, 0:w] = main_image_bgr
        # Place colorbar canvas (needs potential vertical alignment if heights differ slightly)
        offset_y = (h - cb_canvas_h) // 2
        final_image[offset_y:offset_y+cb_canvas_h, w + colorbar_padding:final_w] = cb_canvas

    else: # bottom
        final_w = w
        final_h = h + colorbar_padding + cb_canvas_h
        final_image = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        # Place main image
        final_image[0:h, 0:w] = main_image_bgr
         # Place colorbar canvas (needs potential horizontal alignment if widths differ slightly)
        offset_x = (w - cb_canvas_w) // 2
        final_image[h + colorbar_padding:final_h, offset_x:offset_x+cb_canvas_w] = cb_canvas


    return final_image
