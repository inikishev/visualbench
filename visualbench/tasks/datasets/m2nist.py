import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ... import utils
from .dataset import DatasetBenchmark


class PaddingWrapperForSegResNetDS(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        shape = x.shape
        padding = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 4, device=x.device, dtype=x.dtype, layout=x.layout)
        x = torch.cat([x, padding], dim=-1)
        out = self.net(x)
        return out[..., :shape[-1]]


class M2NIST(DatasetBenchmark):
    """
    5000 samples - segmentation.

    input - ``(B, 1, 64, 84)``

    output - ``(B, 11, 64, 84)``

    a good criterion is ``monai.losses.DiceFocalLoss(softmax=True)``

    ```py
    net = SegResNetDS(2, init_filters=32, in_channels=1, out_channels=11, resolution=(64, 88)).cuda()
    model = PaddedSegResNetDS(net)
    ```
    """
    PaddingWrapperForSegResNetDS = PaddingWrapperForSegResNetDS
    def __init__(
        self,
        combined: str | os.PathLike | Any,
        segmented: str | os.PathLike | Any,
        model: torch.nn.Module,
        criterion, # = DiceFocalLoss(softmax=True,),
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        train_split=0.8,
        data_device = utils.CUDA_IF_AVAILABLE,
    ):
        if isinstance(combined, (str, os.PathLike)): combined = np.load(combined)
        if isinstance(segmented, (str, os.PathLike)): segmented = np.load(segmented)
        x = utils.totensor(combined).float().unsqueeze(1) # 5000, 1, 64, 84
        y = utils.totensor(segmented).float().movedim(-1, 1) # 5000, 11, 64, 84

        # normalize X
        x = x - x.mean()
        x = x / x.std()

        super().__init__(
            data_train = (x, y),
            model = model,
            criterion = criterion,
            batch_size = batch_size,
            test_batch_size = test_batch_size,
            train_split = train_split,
            dtypes = (torch.float32, torch.float32),
            data_device = data_device,
        )

        assert self._dltest is not None
        test_x, test_y = next(iter(self._dltest))

        self.test_x: torch.Tensor = test_x[0].unsqueeze(0)

        try:
            from skimage.color import label2rgb
            self.add_reference_image("target", label2rgb(test_y[0].argmax(0).numpy(force=True)), to_uint8=True)
        except (ImportError, ModuleNotFoundError):
            pass

    @torch.no_grad()
    def after_get_loss(self, x: Tensor, y: Tensor, y_hat: Tensor):
        if not (self.training and self.make_images): return

        mode = self.training
        self.eval()

        y_hat = self.model(self.test_x).squeeze(0).softmax(0) # 11, 64, 84
        for i, img in enumerate((y_hat[:3], y_hat[3:6], y_hat[6:9], y_hat[9:11]), start=1):
            self.log_image(f"pred-{i}", img, to_uint8=True)

        self.train(mode)
