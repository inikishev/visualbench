"""ngl gemini is better than davinci resolve"""
import glob
import os
import sys  # For progress bar
import time

import cv2
import numpy as np

# # --- Default Configuration (can be overridden by function arguments) ---
# DEFAULT_FADE_DURATION_SEC = 1.0
# DEFAULT_TITLE_HEIGHT = 50
# DEFAULT_TITLE_FONT = cv2.FONT_HERSHEY_SIMPLEX
# DEFAULT_TITLE_FONT_SCALE = 0.8
# DEFAULT_TITLE_FONT_COLOR = (255, 255, 255) # White
# DEFAULT_TITLE_FONT_THICKNESS = 2
# DEFAULT_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
# DEFAULT_FOURCC = 'mp4v' # Codec for mp4 output
# # --- End Default Configuration ---

def get_video_files(folder_path, extensions):
    """Finds video files in the specified folder and sorts them."""
    files = []
    for ext in extensions:
        # Case-insensitive search
        files.extend(glob.glob(os.path.join(folder_path, f'*[{ext[1:].lower()}{ext[1:].upper()}]')))
    files.sort() # Sort alphabetically for consistent order
    if not files:
        print(f"Warning: No video files found in '{folder_path}' with extensions: {extensions}")
    return files

def add_title_to_frame(frame, text, title_height, font, scale, color, thickness):
    """Adds a title bar with text above the given frame."""
    if frame is None:
        return None
    original_height, original_width = frame.shape[:2]
    new_height = original_height + title_height

    # Create new taller frame (default black background for title area)
    new_frame = np.zeros((new_height, original_width, 3), dtype=np.uint8)

    # Copy original frame content to the bottom part
    new_frame[title_height:new_height, 0:original_width] = frame

    if text: # Only add text if provided
        # Add text to the top title bar
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        # Center text horizontally, vertically within the title bar
        text_x = max(0, (original_width - text_size[0]) // 2) # Ensure >= 0
        text_y = max(0,(title_height + text_size[1]) // 2)    # Ensure >= 0
        cv2.putText(new_frame, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    return new_frame

def create_blank_frame_with_title_bar(width, height, title_height):
    """Creates a frame with a black title bar area and black content area."""
    new_height = height + title_height
    blank_frame = np.zeros((new_height, width, 3), dtype=np.uint8)
    return blank_frame


def concatenate_videos_with_fade(
    input_folder,
    output_filename,
    fade_duration_sec=1.0,
    title_height=50,
    title_font=cv2.FONT_HERSHEY_SIMPLEX,
    title_font_scale=0.8,
    title_font_color=(255, 255, 255),
    title_font_thickness=2,
    video_extensions=('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'),
    output_fourcc='mp4v'
    ):
    """
    Concatenates videos from a folder with fade transitions and titles above.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_filename (str): Path for the output concatenated video file.
        fade_duration_sec (float): Duration of the fade transition in seconds.
        title_height (int): Height of the title bar above the video in pixels.
        title_font: OpenCV font type for the title.
        title_font_scale (float): Font scale for the title.
        title_font_color (tuple): Font color (B, G, R) for the title.
        title_font_thickness (int): Font thickness for the title.
        video_extensions (list): List of video file extensions to look for.
        output_fourcc (str): FourCC code for the output video codec (e.g., 'mp4v', 'XVID').
    """

    print("--- Starting Video Concatenation with Fade ---")
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return False

    video_files = get_video_files(input_folder, video_extensions)
    if not video_files:
        print("No video files found to process.")
        return False

    print(f"Found {len(video_files)} video files:")
    for f in video_files:
        print(f" - {os.path.basename(f)}")

    # --- Get properties from the first video ---
    first_video_path = video_files[0]
    cap_first = cv2.VideoCapture(first_video_path)
    if not cap_first.isOpened():
        print(f"Error: Cannot open the first video file: {first_video_path}")
        return False

    fps = cap_first.get(cv2.CAP_PROP_FPS)
    original_width = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame_ret, _ = cap_first.read() # Check if we can read at least one frame
    cap_first.release()

    if not first_frame_ret or fps <= 0 or original_width <= 0 or original_height <= 0:
        print(f"Error: Invalid video properties detected in {first_video_path}.")
        print(f"  FPS: {fps}, Width: {original_width}, Height: {original_height}")
        print("  Ensure the first video is valid and readable.")
        # Attempt fallback for FPS only
        if fps <= 0 :
             print("Warning: FPS read as 0 or less. Defaulting to 30 FPS for processing.")
             fps = 30.0 # Use a default if FPS is invalid
        if original_width <= 0 or original_height <= 0:
            print("Error: Cannot proceed with invalid frame dimensions.")
            return False # Cannot proceed without valid dimensions

    # Calculate new dimensions and fade frames
    output_height = original_height + title_height
    output_width = original_width # Width remains the same
    fade_frame_count = max(1, int(fps * fade_duration_sec)) # Ensure at least 1 frame for fade

    print(f"\nOutput Settings:")
    print(f" - Resolution (WxH): {output_width}x{output_height} (Content: {original_width}x{original_height}, Title Bar: {title_height}px)")
    print(f" - Frame Rate (FPS): {fps:.2f}")
    print(f" - Fade Duration: {fade_duration_sec}s (~{fade_frame_count} frames)")
    print(f" - Output File: {output_filename}")
    print(f" - Output Codec: {output_fourcc}")

    # --- Setup Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*output_fourcc)
    out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_filename}.")
        print(f"  Check if codec '{output_fourcc}' is supported and you have write permissions.")
        return False

    start_time = time.time()
    total_frames_written = 0
    last_video_last_frame_content = None # To store the content part of the last frame for fading

    # --- Process Videos ---
    for i, video_path in enumerate(video_files):
        filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing video {i+1}/{len(video_files)}: {filename_no_ext}...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Warning: Could not open video {video_path}. Skipping.")
            continue

        # --- Fade Transition (if not the first video) ---
        if i > 0 and last_video_last_frame_content is not None:
            print(f"  - Creating fade transition from previous video...")

            # Get the first frame of the CURRENT video
            ret_next, next_video_first_frame_raw = cap.read()
            if not ret_next:
                print(f"  Warning: Could not read first frame of {filename_no_ext} for transition. Skipping fade.")
                # Reset cap to beginning if possible, or just continue (might miss first frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_video_first_frame_content = None
            else:
                # Resize if necessary
                h_next, w_next = next_video_first_frame_raw.shape[:2]
                if w_next != original_width or h_next != original_height:
                     next_video_first_frame_content = cv2.resize(next_video_first_frame_raw, (original_width, original_height), interpolation=cv2.INTER_AREA)
                else:
                    next_video_first_frame_content = next_video_first_frame_raw

                # Rewind the video capture to the beginning to process it normally later
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # Generate fade frames
                if next_video_first_frame_content is not None:
                    for k in range(fade_frame_count):
                        alpha = k / (fade_frame_count -1) # Blend factor from 0.0 to 1.0
                        beta = 1.0 - alpha

                        # Blend the CONTENT parts only
                        try:
                            faded_content = cv2.addWeighted(last_video_last_frame_content, beta, next_video_first_frame_content, alpha, 0.0)
                        except cv2.error as e:
                             print(f"\nError during cv2.addWeighted (fade): {e}")
                             print(f"  last_video_last_frame_content shape: {last_video_last_frame_content.shape}, dtype: {last_video_last_frame_content.dtype}")
                             print(f"  next_video_first_frame_content shape: {next_video_first_frame_content.shape}, dtype: {next_video_first_frame_content.dtype}")
                             print(f"  beta={beta}, alpha={alpha}")
                             print("  Skipping remainder of fade for this transition.")
                             break # Skip rest of fade if blending fails


                        # Add a blank title bar area
                        faded_frame_with_blank_title = create_blank_frame_with_title_bar(output_width, original_height, title_height)
                        faded_frame_with_blank_title[title_height:, :] = faded_content # Put faded content below blank title

                        out.write(faded_frame_with_blank_title)
                        total_frames_written += 1
                    print(f"  - Fade transition completed.")
                else:
                     print("   - Could not generate fade frames as next video's first frame was unavailable.")
            # --- End Fade Transition ---

        # --- Process Current Video Frames ---
        frame_count_current_video = 0
        current_frame_index = 0
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # For progress bar

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of this video

            # Store the raw frame content of the last successfully read frame for the *next* transition
            # Resize if necessary BEFORE storing and adding title
            current_h, current_w = frame.shape[:2]
            if current_w != original_width or current_h != original_height:
                frame_content = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_AREA)
            else:
                frame_content = frame

            last_video_last_frame_content = frame_content # Update for potential next fade

            # Add the title bar above the frame content
            frame_with_title = add_title_to_frame(
                frame_content, filename_no_ext, title_height,
                title_font, title_font_scale, title_font_color, title_font_thickness
            )

            # Write the modified frame
            out.write(frame_with_title)
            frame_count_current_video += 1
            total_frames_written += 1
            current_frame_index += 1

            # Simple progress bar
            if total_frames_in_video > 0 and frame_count_current_video % 50 == 0: # Update every 50 frames
                 progress = int(50 * current_frame_index / total_frames_in_video)
                 sys.stdout.write(f"\r    Progress: [{'=' * progress}{' ' * (50 - progress)}] {current_frame_index}/{total_frames_in_video} frames")
                 sys.stdout.flush()


        if total_frames_in_video > 0 : # Final progress update for the video
            sys.stdout.write(f"\r    Progress: [{'=' * 50}] {frame_count_current_video}/{total_frames_in_video} frames\n")
            sys.stdout.flush()
        else: # If total frames couldn't be read
            print(f"\r    Processed {frame_count_current_video} frames (total frame count unknown).")

        cap.release()
        print(f"  - Finished processing {filename_no_ext}.")


    # --- Cleanup ---
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"\n------------------------------------")
    print(f"Finished concatenating videos!")
    print(f"Total frames written: {total_frames_written}")
    print(f"Output video saved as: {output_filename}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"------------------------------------")
    return True

