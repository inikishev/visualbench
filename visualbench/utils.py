import numpy as np
from myai.loaders.image import imreadtensor
from myai.transforms import force_hw3, force_hwc
import torch

def to_float_hw3_tensor(x):
    if isinstance(x, str): return force_hw3(imreadtensor(x).float())
    if isinstance(x, torch.Tensor): return force_hw3(x.float())
    if isinstance(x, np.ndarray): return force_hw3(torch.from_numpy(x).float())
    return force_hw3(torch.from_numpy(np.asarray(x)).float())


def to_float_hwc_tensor(x):
    if isinstance(x, str): return force_hwc(imreadtensor(x).float())
    if isinstance(x, torch.Tensor): return force_hwc(x.float())
    if isinstance(x, np.ndarray): return force_hwc(torch.from_numpy(x).float())
    return force_hwc(torch.from_numpy(np.asarray(x)).float())


CUDA_IF_AVAILABLE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')