import warnings
import numpy as np
import torch

def _imread_skimage(path:str) -> np.ndarray:
    import skimage
    return skimage.io.imread(path)

def _imread_plt(path:str) -> np.ndarray:
    import matplotlib.pyplot as plt
    return plt.imread(path)

def _imread_cv2(path):
    import cv2
    image = cv2.imread(path) # pylint:disable=no-member
    if image.ndim == 3: image = image[:, :, ::-1] # BRG -> RGB
    return image

def _imread_imageio(path):
    from imageio import v3
    return v3.imread(path)

def _imread_pil(path:str) -> np.ndarray:
    import PIL.Image
    return np.array(PIL.Image.open(path))

def _imread_torchvision(path:str, dtype=None, device=None) -> torch.Tensor:
    import torchvision
    return torchvision.io.read_image(path).to(dtype=dtype, device=device, copy=
                                              False)

def _imread(path: str) -> torch.Tensor:
    try: return _imread_torchvision(path)
    except Exception:
        img = None
        exceptions = []
        for fn in (_imread_plt, _imread_pil, _imread_cv2, _imread_imageio, _imread_skimage):
            try: img = fn(path)
            except Exception as e: exceptions.append(e)

    if img is None: raise exceptions[0] from None
    return torch.from_numpy(img.copy())

