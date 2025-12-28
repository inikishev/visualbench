import os

import cv2
import numpy as np
import torch
from torch import nn

from ..benchmark import Benchmark


class Hadamard(Benchmark):
    """
    """
    def __init__(self, n_half: int, binarization_weight: float = 0.1):
        super().__init__()
        if not isinstance(n_half, int) or n_half <= 0:
            raise ValueError("n_half must be a positive integer.")

        self.n_half = n_half
        self.N = 2 * n_half
        self.binarization_weight = binarization_weight

        self.raw_a_first_row = nn.Parameter(torch.randn(self.n_half) * 0.5)
        self.raw_b_first_row = nn.Parameter(torch.randn(self.n_half) * 0.5)

    def _construct_circulant_matrix(self, first_row: torch.Tensor) -> torch.Tensor:
        dim = first_row.shape[0]
        rows = [torch.roll(first_row, shifts=i) for i in range(dim)]
        return torch.stack(rows, dim=0)

    def get_loss(self) -> torch.Tensor:
        a_first_row = torch.tanh(self.raw_a_first_row)
        b_first_row = torch.tanh(self.raw_b_first_row)

        A = self._construct_circulant_matrix(a_first_row)
        B = self._construct_circulant_matrix(b_first_row)

        # H = [[A,  B],
        #      [B, -A]]
        H_top_row = torch.cat((A, B), dim=1)
        H_bottom_row = torch.cat((B, -A), dim=1)
        H = torch.cat((H_top_row, H_bottom_row), dim=0)

        # 1. Hadamard Loss: H @ H.T should be N * I
        identity_N = torch.eye(self.N, device=H.device, dtype=H.dtype)
        target_HHT = self.N * identity_N

        HHT = torch.matmul(H, H.T)
        loss_hadamard = torch.mean((HHT - target_HHT)**2)

        # 2. Binarization Loss: Entries of H should be close to +1 or -1
        loss_binarization = torch.mean((H**2 - 1)**2)

        total_loss = loss_hadamard + self.binarization_weight * loss_binarization

        # vis
        if self._make_images:
            frame = self._make_frame(H)
            self.log_image('image', frame, to_uint8=False)

        return total_loss

    @torch.no_grad
    def _make_frame(self, matrix_H: torch.Tensor, frame_size: tuple = (256, 256)):
        H_np = matrix_H.detach().cpu().numpy()

        img_normalized = (H_np + 1.0) / 2.0
        img_uint8 = (img_normalized * 255.0).astype(np.uint8)

        if img_uint8.shape[0] < frame_size[0] or img_uint8.shape[1] < frame_size[1]:
            img_resized = cv2.resize(img_uint8, frame_size, interpolation=cv2.INTER_NEAREST) # pylint:disable=no-member
        else:
            img_resized = img_uint8

        return img_resized

    def get_final_H(self, threshold: float = 0.0) -> torch.Tensor:
        with torch.no_grad():
            a_first_row = torch.tanh(self.raw_a_first_row)
            b_first_row = torch.tanh(self.raw_b_first_row)
            A = self._construct_circulant_matrix(a_first_row)
            B = self._construct_circulant_matrix(b_first_row)
            H_top_row = torch.cat((A, B), dim=1)
            H_bottom_row = torch.cat((B, -A), dim=1)
            H_continuous = torch.cat((H_top_row, H_bottom_row), dim=0)

            H_binary = torch.ones_like(H_continuous)
            H_binary[H_continuous <= threshold] = -1.0
            return H_binary.detach()
