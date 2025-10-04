from collections.abc import Sequence

import torch
from torch import nn

from ..benchmark import Benchmark


def _sections_mask(init: torch.Tensor, n_nodes, withds):
    mask = torch.ones_like(init, dtype=torch.bool)
    h, w = mask.shape
    if n_nodes == 1: return mask
    n_sections = n_nodes * 2 - 1
    section_width = w // n_sections
    cur = 0
    middle = h//2
    for bridge_w in list(withds) + [None]:
        end = min(cur+section_width*2, w)
        if end == w: return mask
        size = end - cur
        mask[:, end-size//2:end] = False
        mask[middle:middle+bridge_w] = True
        cur += section_width*2

    raise RuntimeError('widths not enough of them')


def _snake_mask(init: torch.Tensor, width):
    h,w = init.shape
    wall = torch.zeros((h, width), dtype=torch.bool, device=init.device)
    path = torch.ones((h, width), dtype=torch.bool, device=init.device)

    section = torch.cat([wall, path], 1)
    mask = section.repeat(1, int((w/2) // width))

    cur = 0
    cur_up = True
    while True:
        if cur_up:
            mask[:width, cur:cur+width] = True

        else:
            mask[-width:, cur:cur+width] = True

        cur_up = not cur_up
        cur += width * 2
        if cur + width >= w: break

    return mask

def _better_snake_mask(init: torch.Tensor, width):
    h,w = init.shape
    wall = torch.zeros((h, width), dtype=torch.bool, device=init.device)
    path = torch.ones((h, width), dtype=torch.bool, device=init.device)

    section = torch.cat([wall, path], 1)
    num_sections = int((w/2) // width)
    mask = section.repeat(1, num_sections)

    cur = 0
    cur_up = True
    vertical_widths = torch.linspace(h**0.5, 1, num_sections).pow(2).int()
    for vert_w in vertical_widths:
        if cur_up:
            mask[:vert_w, cur:cur+width] = True

        else:
            mask[-vert_w:, cur:cur+width] = True

        cur_up = not cur_up
        cur += width * 2
        if cur + width >= w: break

    return mask


class Colorization(Benchmark):
    """The goal is to minimize difference between each pixel and it's neighbours, while one pixel is fixed at value of one.

    Renders:
        pixels.

    Tip: for some presets use this: ``bench = vb.Colorization.snake()`` or ``bench = vb.Colorization.small()`` or
    ``bench = vb.Colorization.tiny()``.

    Inspired by https://distill.pub/2017/momentum/"""
    def __init__(self, init: torch.Tensor | None = None, mask: torch.Tensor | None = None, white_idxs: Sequence[Sequence[int]] | None = None, order: int = 1, power: float = 2):
        super().__init__(bounds=(0,1))

        if init is None: init = torch.zeros(96, 256)
        if mask is None: mask =_better_snake_mask(init, 4)
        if white_idxs is None: white_idxs = ((0, 0), )

        mask = mask.float()
        white_mask = torch.zeros_like(mask)
        image = init * mask
        for idx in white_idxs:
            image[*idx] = 1
            white_mask[*idx] = 1

        self.image = nn.Parameter(image)
        self.mask = nn.Buffer(mask)
        self.white_mask = nn.Buffer(white_mask)
        self.white_mask_inverse = nn.Buffer(1-white_mask)


        self.white_idxs = white_idxs
        self.order = order
        self.power = power
        self._show_titles_on_video = False

    @classmethod
    def snake(cls, order: int = 1, power: float = 2):
        init = torch.zeros(96, 256)
        return cls(init = init, mask =_better_snake_mask(init, 16), white_idxs = ((0, 0),), order = order, power=power)

    @classmethod
    def small(cls, order: int = 1, power: float = 2):
        init = torch.zeros(32, 128)
        return cls(init = init, mask =_better_snake_mask(init, 4), white_idxs = ((0, 0),), order = order, power=power)

    @classmethod
    def tiny(cls, order: int = 1, power: float = 2):
        init = torch.zeros(8, 32)
        return cls(init = init, mask =_better_snake_mask(init, 4), white_idxs = ((0, 0),), order = order, power=power)

    def get_loss(self):
        w = (self.image * self.mask * self.white_mask_inverse) + self.white_mask

        # colorizer = 0
        # for idx in self.white_idxs:
        #     colorizer = colorizer + (1 - w[*idx])**2

        diff_ver = torch.diff(w, self.order, 0) * self.mask[self.order:] * self.mask[:-self.order]
        diff_hor = torch.diff(w, self.order, 1) * self.mask[:, self.order:] * self.mask[:, :-self.order]

        if self.power % 2 != 0 or self.power < 0:
            diff_ver = diff_ver.abs()
            diff_hor = diff_hor.abs()

        spreader = torch.sum(diff_ver**self.power) + torch.sum(diff_hor**self.power)

        if self._make_images:
            with torch.no_grad():
                frame = (w + (1-self.mask)*0.1)[:,:,None].repeat_interleave(3, 2)
                red_overflow = (frame - 1).clip(min=0)
                red_overflow[:,:,1:] *= 2
                blue_overflow = - frame.clip(max=0)
                blue_overflow[:,:,2] *= 2
                frame = ((frame - red_overflow + blue_overflow) * 255).clip(0,255).to(torch.uint8).detach().cpu()
                self.log_image('image', frame, to_uint8=False, show_best=True, log_difference=True)

        return spreader
        #return 0.5*colorizer + 0.5*spreader