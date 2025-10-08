import warnings
from os import PathLike

import numpy as np
import torch


def _imread_skimage(path:str|PathLike) -> np.ndarray:
    import skimage
    return skimage.io.imread(path)

def _imread_plt(path:str|PathLike) -> np.ndarray:
    import matplotlib.pyplot as plt
    return plt.imread(str(path))

def _imread_cv2(path: str|PathLike) -> np.ndarray:
    import cv2
    image = cv2.imread(str(path)) # pylint:disable=no-member
    assert image is not None
    if image.ndim == 3: image = image[:, :, ::-1] # BRG -> RGB
    return image

def _imread_imageio(path: str|PathLike):
    from imageio import v3
    return v3.imread(str(path))

def _imread_pil(path: str|PathLike) -> np.ndarray:
    import PIL.Image
    return np.array(PIL.Image.open(path))

def _imread_torchvision(path: str|PathLike, dtype=None, device=None) -> torch.Tensor:
    import torchvision
    return torchvision.io.read_image(str(path)).to(dtype=dtype, device=device, copy=
                                              False)

def _imread(path: str|PathLike) -> torch.Tensor:
    try: return _imread_torchvision(path)
    except Exception:
        img = None
        exceptions = []
        for fn in (_imread_plt, _imread_pil, _imread_cv2, _imread_imageio, _imread_skimage):
            try: img = fn(path)
            except Exception as e: exceptions.append(e)

    if img is None: raise exceptions[0] from None
    return torch.from_numpy(img.copy())

