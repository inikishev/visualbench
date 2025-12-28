import os
from os import PathLike
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image

from .format import tonumpy


def make_hw3(x: np.ndarray, allow_4_channels = False) -> np.ndarray:
    """Forces input tensor to be (H, W, 3) format.
    If tensor has 2 channels, the third will be filled with zeroes.
    Takes the central slice of tensors with more than 3 dimensions."""

    x = x.squeeze()
    if x.ndim > 3: raise ValueError(f"array has a shape of {x.shape}, needs to be 2D or 3D")

    # create channel dimension if it doesn't exist
    if x.ndim == 2: x = x[:,:,None]

    # channel first to channel last
    if x.shape[0] < x.shape[-1]: x = np.moveaxis(x, 0, -1)

    maxv = 4 if allow_4_channels else 3

    # (H, W, 1), repeat a single channel 3 times
    if x.shape[-1] == 1: x = np.concatenate([x,x,x], axis=-1)

    # (H, W, 2), add the third channel with zeroes
    elif x.shape[-1] == 2: x = np.concatenate([x, np.zeros_like(x[:,:,0,None])], axis=-1)

    # (H, W, 4+), remove extra channels
    elif x.shape[-1] > maxv: x = x[:,:,:maxv]
    return x

class _GIFWriter:
    """mimics OpenCV writer"""
    def __init__(self, outfile, codec, fps, shape):
        self.outfile = outfile
        self.fps = fps
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        frames = [Image.fromarray(frame) for frame in self.frames]
        frames[0].save(self.outfile, save_all=True, append_images=frames[1:], duration=1000/self.fps, loop=0)

class OpenCVRenderer:
    """A frame by frame video renderer using OpenCV.

    Args:
        outfile (str): path to the file to write the video to. The file will be created when first frame is added
        fps (int, optional): frames per second. Defaults to 60.
        codec (str, optional): codec. Defaults to "mp4v".
        scale (int, optional): rescale frames. Scale needs to be integer, 1/scale needs to be an integer. Defaults to 1.

    Example:
    ```
    # create a renderer
    with OpenCVRenderer('out.mp4', fps = 60) as renderer:

        # add 1000 640x320 frames with random values
        for i in range(1000):
            frame = np.random.uniform(0, 255, size = (640, 320, 3))
            renderer.add_frame(frame)

    # if you haven't used a context, you have to release the renderer, which makes the video file openable
    renderer = OpenCVRenderer('out.mp4', fps = 60)
    for i in range(1000): renderer.add_frame(torch.randn(100,100,3))
    renderer.release()
    ```
    """

    def __init__(
        self,
        outfile: PathLike | str,
        fps = 60,
        codec="avc1",
        scale: int | float = 1,
    ):
        outfile = str(outfile)

        if fps < 1: raise ValueError(f"FPS must be at least 1, got {fps}")
        if not outfile.lower().endswith((".mp4", ".gif")):
            outfile += ".mp4"

        self.outfile = outfile
        self.is_gif = self.outfile.lower().endswith(".gif")
        self.fps = fps
        self.codec = codec
        self.scale = scale

        self.writer = None

    def write(self, frame: np.ndarray | torch.Tensor):
        """Write the next frame to the video file.
        All frames must of the same shape, have np.uint8 or torch.uint8 data type.

        Args:
            frame (np.ndarray | torch.Tensor): frame in np.uint8 data type
        """

        # make sure it is hw3 and scale it
        frame = make_hw3(tonumpy(frame))

        if not self.is_gif:
            frame = frame[:,:,::-1]

        if self.scale > 1:
            frame = np.repeat(np.repeat(frame, int(self.scale), 0), int(self.scale), 1)

        elif self.scale < 1:
            skip = round(1/self.scale)
            frame = frame[::skip,::skip]

        # on first frame create writer and use frame shape as video size
        if self.writer is None:
            self.shape = frame.shape

            if str(self.outfile).lower().endswith(".gif"): Writer = _GIFWriter
            else: Writer = cv2.VideoWriter # pylint:disable = no-member
            self.writer = Writer(
                self.outfile,
                cv2.VideoWriter_fourcc(*self.codec), # pyright:ignore[reportAttributeAccessIssue] # pylint:disable = no-member
                self.fps,
                (self.shape[1], self.shape[0])
            )

        # check frame shape (opencv doesn't have that check, it just skips the frame)
        if frame.ndim != 3: raise ValueError(f"Frame must have 3 dimensions: (H, W, 3), got frame of shape {frame.shape}")
        if frame.shape[2] != 3: raise ValueError(f"The last frame dimension must be 3 (RGB), got frame of shape {frame.shape}")
        if frame.shape != self.shape: raise ValueError(f"Frame size {frame.shape} is different from previous frame size {self.shape}")
        if frame.dtype != np.uint8: raise ValueError(f"Frame must be of type np.uint8, got {frame.dtype}")

        # write new frame to file
        self.writer.write(frame)

    def release(self):
        """Close the writer, releasing access to the video file."""
        if self.writer is None: raise ValueError("No frames have been added to this renderer.")
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

def render_frames(file: PathLike | str, frames, fps = 60, norm: Literal['none', 'each', 'all'] = 'none', scale=1):
    dirname = os.path.dirname(file)
    if len(dirname) > 0 and not os.path.exists(dirname):
        raise FileNotFoundError(f"directory `{dirname}` doesn't exist!")

    if norm == 'all':
        frames = [tonumpy(f).astype(np.float32) for f in frames]
        min_v = min(f.min() for f in frames)
        frames = [(f - min_v) for f in frames]
        max_v = max(f.max() for f in frames)
        if max_v == 0: max_v = 1
        frames = [(f / max_v) * 255 for f in frames]
        frames = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]

    if norm == 'each':
        frames = [tonumpy(f).astype(np.float32) for f in frames]
        frames = [(f - f.min()) for f in frames]
        frames = [(f / (f.max() if f.max() != 0 else 1)) * 255 for f in frames]
        frames = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]


    with OpenCVRenderer(file, fps, scale=scale) as r:
        for frame in frames:
            r.write(frame)

