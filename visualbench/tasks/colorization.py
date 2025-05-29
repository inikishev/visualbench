import torch
from torch import nn
from ..benchmark import Benchmark

def _get_mask(init: torch.Tensor, n_nodes, withds):
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

_INIT = torch.zeros(64, 256)

class Colorization(Benchmark):
    """inspired by https://distill.pub/2017/momentum/"""
    def __init__(self, init: torch.Tensor = _INIT, mask: torch.Tensor=_get_mask(_INIT, 4, (33, 10, 2)), pull_idxs = ((32, 0),), order: int = 1,):
        super().__init__(bounds=(0,1))
        image = init * mask
        for idx in pull_idxs:
            image[*idx] = 1

        self.image = nn.Parameter(image)
        self.mask = nn.Buffer(mask.float())
        self.pull_idxs = pull_idxs
        self.order = order

    def get_loss(self):
        w = self.image * self.mask

        colorizer = 0
        for idx in self.pull_idxs:
            colorizer = colorizer + (1 - w[*idx])**2

        diff_ver = torch.diff(w, self.order, 0) * self.mask[self.order:] * self.mask[:-self.order]
        diff_hor = torch.diff(w, self.order, 1) * self.mask[:, self.order:] * self.mask[:, :-self.order]

        # diff_ver = (w[1:, :] - w[:-1, :]) * self.mask[1:] * self.mask[:-1]
        # diff_hor = (w[:, 1:] - w[:, :-1]) * self.mask[:, 1:] * self.mask[:, :-1]
        spreader = torch.sum(diff_ver**2) + torch.sum(diff_hor**2)

        if self._make_images:
            with torch.no_grad():
                frame = (w + (1-self.mask)*0.1)[:,:,None].repeat_interleave(3, 2)
                # frame[0,0]=1; frame[-1,-1]=0 # lazy way to do minimal norm
                red_overflow = (frame - 1).clip(min=0)
                red_overflow[:,:,1:] *= 2
                blue_overflow = - frame.clip(max=0)
                blue_overflow[:,:,2] *= 2
                frame = ((frame - red_overflow + blue_overflow) * 255).clip(0,255).to(torch.uint8).detach().cpu()
                self.log_image('image', frame, to_uint8=False)

        return 0.5*colorizer + 0.5*spreader