import os
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..utils import normalize, to_3HW
from ..utils.image import _imread

# _path = os.path.dirname(__file__)

def _get_path(fname:str):
    return Path(str(resources.files("visualbench").joinpath("data", fname)))

QRCODE96 = _get_path('qr-96.jpg')
"""QR code that links to my github account"""

ATTNGRAD96 = _get_path('attngrad-96.png')
"""Piece of gradient of some model from transformers library except I don't remember which one"""

SANIC96 = _get_path('sanic-96.jpg')
"""is an 8 year old image from my images folder and I think it is a screenshot from one of the sanic games"""

FROG96 = _get_path('frog-96.png')
"""frame from https://www.youtube.com/@NinjaFrog777/videos"""

WEEVIL96 = _get_path('weevil-96.png')
"""is from http://growingorganic.com/ipm-guide/weevils/"""

TEST96 = _get_path('test-96.jpg')
"""this was generated in like 2012 ago by google doodle generator and its still my favourite image and it is called test"""

MAZE96 = _get_path('maze-96.png')
"""a generic maze"""

TEXT96 = _get_path('text-96.png')
"""lorem ipsum from lorem ipsum text"""

GEOM96 = _get_path('geometry-96.png')
"""CC0 image from wikicommons, SORRY I CAN'T FIND THE LINK ANYMORE!!!"""

RUBIC96 = _get_path('rubic-96.png')
"""is from https://speedsolving.fandom.com/wiki/Rubik%27s_Cube?file=Rubik%27s_Cube_transparency.png"""

SPIRAL96 = _get_path('spiral-96.png')
"""A colorful spiral"""

BIANG96 = _get_path('biang-96.png')
"""apparently its the hardest hieroglyph and it is from https://commons.wikimedia.org/wiki/File:Bi%C3%A1ng_%28regular_script%29.svg"""

EMOJIS96 = _get_path('emojis-96.png')
"""some random emojis"""

GRID96 = _get_path('grid-96.png')
"""Grid of black and white cells"""

def get_qrcode():
    """QR code that links to my github account, this returns single channel binary image (in float32)"""
    qrcode = to_3HW(_imread(QRCODE96).float()).mean(0)
    return torch.where(qrcode > 128, 1, 0).float().contiguous()

def get_maze():
    """a generic maze, this returns single channel binary image (in float32)"""
    qrcode = to_3HW(_imread(MAZE96).float()).mean(0)
    return torch.where(qrcode > 128, 1, 0).float().contiguous()

def get_grid():
    """Grid of black and white cells, this returns single channel binary image (in float32)"""
    grid = to_3HW(_imread(GRID96).float()).mean(0)
    return torch.where(grid > 128, 1, 0).float().contiguous()

def get_text():
    """lorem ipsum from lorem ipsum text, this returns it in (0, 1) range."""
    qrcode = to_3HW(_imread(TEXT96).float()).mean(0)
    return normalize(qrcode.float().contiguous(), 0, 1)

def get_biang():
    """apparently its the hardest hieroglyph and it is from https://commons.wikimedia.org/wiki/File:Bi%C3%A1ng_%28regular_script%29.svg, this returns it in (0, 1) range."""
    biang = to_3HW(_imread(BIANG96).float()).mean(0)
    return normalize(biang.float().contiguous(), 0, 1)

def get_randn(size:int = 64):
    """randn but seed is 0"""
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

def get_orthonormal(size: int = 64):
    Q, R = torch.linalg.qr(torch.randn((size, size)), mode='reduced')  # pylint:disable=not-callable
    return Q

def get_3d_structured48():
    """A mix of images that becomes 48x48x348 tensor"""
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

    sigma = np.zeros((rows, cols), dtype=np.float64) # linalg doesnt support float128
    np.fill_diagonal(sigma, singular_values)
    U = np.linalg.qr(np.random.rand(rows, rows))[0]
    V = np.linalg.qr(np.random.rand(cols, cols))[0]
    A = U @ sigma @ V.T
    return torch.from_numpy(A.copy()).float().contiguous()


def get_font_dict(dtype=torch.bool, device=None):
    """returns a dictionary which maps letters, numbers and +-*/|. to 3x3 binary images."""
    path = _get_path('3x3 font.jpeg')
    image = to_3HW(_imread(path).float()).mean(0)
    image = torch.where(image > 128, 1, 0).contiguous().to(dtype=dtype, device=device)

    # the font is 6x6 grid of letters with 1 px padding outside and between letters
    image = image[1:-1,1:-1]
    rows = [t.clone() for t in torch.chunk(image, 6)]
    letters = [t.clone() for row in rows for t in torch.chunk(row, 6)]

    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    font = dict(zip(chars, letters))

    # add some other stuff
    add = torch.zeros_like(font["a"])
    add[1] = 1; add[:, 1] = 1
    font["+"] = add

    sub = torch.zeros_like(font["a"])
    sub[1] = 1
    font["-"] = sub

    mul = torch.zeros_like(font["a"])
    # mul[0,0] = 1; mul[1, 1] = 1; mul[2, 2] = 1; mul[2, 0] = 1; mul[0, 2] = 1
    mul[1,1] = 1
    font["*"] = mul

    div = torch.zeros_like(font["a"])
    div[0,0] = 1; div[1, 1] = 1; div[2, 2] = 1
    font["/"] = div

    period = torch.zeros_like(font["a"])
    period[-1,-1] = 1
    font["."] = period

    line = torch.zeros_like(font["a"])
    line[:, 1] = 1
    font["|"] = line

    return font


