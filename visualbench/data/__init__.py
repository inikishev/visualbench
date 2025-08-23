import os
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from ..utils import normalize, to_3HW
from ..utils.image import _imread

_path = os.path.dirname(__file__)

QRCODE96 = os.path.join(_path, 'qr-96.jpg')
ATTNGRAD96 = os.path.join(_path, 'attngrad-96.png')
SANIC96 = os.path.join(_path, 'sanic-96.jpg')
FROG96 = os.path.join(_path, 'frog-96.png')
WEEVIL96 = os.path.join(_path, 'weevil-96.png')
TEST96 = os.path.join(_path, 'test-96.jpg')
MAZE96 = os.path.join(_path, 'maze-96.png')
TEXT96 = os.path.join(_path, 'text-96.png')
GEOM96 = os.path.join(_path, 'geometry-96.png')
RUBIC96 = os.path.join(_path, 'rubic-96.png')
SPIRAL96 = os.path.join(_path, 'spiral-96.png')
BIANG96 = os.path.join(_path, 'biang-96.png')
EMOJIS96 = os.path.join(_path, 'emojis-96.png')
GRID96 = os.path.join(_path, 'grid-96.png')

def get_qrcode():
    qrcode = to_3HW(_imread(QRCODE96).float()).mean(0)
    return torch.where(qrcode > 128, 1, 0).float().contiguous()

def get_maze():
    qrcode = to_3HW(_imread(MAZE96).float()).mean(0)
    return torch.where(qrcode > 128, 1, 0).float().contiguous()

def get_grid():
    grid = to_3HW(_imread(GRID96).float()).mean(0)
    return torch.where(grid > 128, 1, 0).float().contiguous()

def get_text():
    qrcode = to_3HW(_imread(TEXT96).float()).mean(0)
    return normalize(qrcode.float().contiguous(), 0, 1)

def get_biang():
    biang = to_3HW(_imread(BIANG96).float()).mean(0)
    return normalize(biang.float().contiguous(), 0, 1)

def get_randn(size:int = 64):
    return torch.randn(size, size, generator = torch.Generator('cpu').manual_seed(0))

def get_circulant(size: int = 64):
    import scipy.linalg
    generator = np.random.default_rng(0)
    c = generator.uniform(-1, 1, (3, size))
    return torch.from_numpy(scipy.linalg.circulant(c).copy()).float().contiguous()

def get_dft(size: int = 96):
    import scipy.linalg
    dft = np.stack([scipy.linalg.dft(size).real, scipy.linalg.dft(size).imag], 0)
    return torch.from_numpy(dft).float().contiguous()

def get_fielder(size: int = 64):
    import scipy.linalg
    generator = np.random.default_rng(0)
    c = generator.uniform(-1, 1, (3, size))
    return torch.from_numpy(scipy.linalg.fiedler(c).copy()).float().contiguous()

def get_hadamard(size: int = 64):
    import scipy.linalg
    return torch.from_numpy(scipy.linalg.hadamard(size, float).copy()).float().contiguous() # pyright:ignore[reportArgumentType]

def get_helmert(size: int = 64):
    import scipy.linalg
    return torch.from_numpy(scipy.linalg.helmert(size).copy()).float().contiguous() # pyright:ignore[reportArgumentType]

def get_hilbert(size: int = 64):
    import scipy.linalg
    return torch.from_numpy(scipy.linalg.hilbert(size).copy()).float().contiguous() # pyright:ignore[reportArgumentType]

def get_invhilbert(size: int = 64):
    import scipy.linalg
    return torch.from_numpy(scipy.linalg.invhilbert(size).copy()).float().contiguous() # pyright:ignore[reportArgumentType]

def get_3d_structured48():
    qr = get_qrcode() # (96x96)
    attn = to_3HW(ATTNGRAD96) # (3x96x96)
    sanic = to_3HW(SANIC96)
    test = to_3HW(TEST96)

    qr = qr.unfold(0, 48, 48).unfold(1, 48, 48).flatten(0,1) # 4,48,48
    qr = torch.cat([qr, qr.flip(0), qr.flip(1)]) # 12,48,48
    attn = attn.unfold(1, 48, 48).unfold(2, 48, 48).flatten(0,2) # 12,48,48
    sanic = attn.unfold(1, 48, 48).unfold(2, 48, 48).flatten(0,2) # 12,48,48
    test = attn.unfold(1, 48, 48).unfold(2, 48, 48).flatten(0,2) # 12,48,48

    stacked = torch.cat([qr,attn,sanic,test]) # 48,48,48
    # make dims varied
    stacked[:12] = attn
    stacked = stacked.transpose(0, 1)
    stacked[:12] = test
    stacked = stacked.transpose(0,2)
    stacked[:12] = qr

    return stacked



def get_lowrank(size: Sequence[int], rank:int, seed=0):
    from ..tasks.linalg.linalg_utils import make_low_rank_tensor
    return make_low_rank_tensor(size, rank, seed=seed)

def get_ill_conditioned(size: int | tuple[int,int], cond:float=1e17):
    """cond can't be above around 1e17 because of precision"""
    if isinstance(size, int): size = (size, size)

    # precision is better in numpy
    *b, rows, cols = size
    k = min(rows, cols)
    singular_values = np.linspace(1, 1/cond, k, dtype=np.float128)

    Sigma = np.zeros((rows, cols), dtype=np.float64) # linalg doesnt support float128
    np.fill_diagonal(Sigma, singular_values)
    U = np.linalg.qr(np.random.rand(rows, rows))[0]
    V = np.linalg.qr(np.random.rand(cols, cols))[0]
    A = U @ Sigma @ V.T
    return torch.from_numpy(A.copy()).float().contiguous()