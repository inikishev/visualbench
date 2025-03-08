from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from light_dataloader import TensorDataLoader
from myai.transforms import normalize, totensor
from torch import nn

from ..._utils import CUDA_IF_AVAILABLE
from ...benchmark import Benchmark


class DatasetBenchmark(Benchmark):
    def __init__(
        self,
        x,
        y,
        model,
        criterion,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        x_test=None,
        y_test=None,
        test_split: int | float | None = None,
        shuffle_split=False,
        normalize_x = False,
        normalize_y = False,
        x_dtype = torch.float32,
        y_dtype = torch.float32,
        data_device=CUDA_IF_AVAILABLE,
        decision_boundary = False,
        resolution = 192,
        boundary_act = None,
        seed = 0
    ):
        """_summary_

        Args:
            x (_type_): inputs
            y (_type_): targets
            model (_type_): model
            criterion (_type_): loss MUST HAVE "REDUCTION" PARAMETER
            batch_size (int | None, optional): _description_. Defaults to None.
            x_test (_type_, optional): _description_. Defaults to None.
            y_test (_type_, optional): _description_. Defaults to None.
            test_split (int | float | None, optional): _description_. Defaults to None.
            shuffle_split (bool, optional): _description_. Defaults to False.
            dtype (_type_, optional): _description_. Defaults to torch.float32.
            data_device (_type_, optional): _description_. Defaults to CUDA_IF_AVAILABLE.
            seed (int, optional): _description_. Defaults to 0.
        """
        if batch_size is None and test_batch_size is not None: raise NotImplementedError('batch_size is None, but test_batch size is not None')
        x = totensor(x, dtype=x_dtype, device=data_device)
        y = totensor(y, dtype=y_dtype, device=data_device)

        if x_test is not None:
            x_test = totensor(x_test, dtype=x_dtype, device=data_device)
            y_test = totensor(y_test, dtype=y_dtype, device=data_device)

        # split x y into train and test based on test_split
        elif test_split is not None:
            if isinstance(test_split, float):
                test_split = int(test_split * len(x))

            if shuffle_split:
                indices = torch.randperm(len(x), generator=torch.Generator(x.device).manual_seed(seed), device=x.device)
                x = torch.index_select(x, 0, indices)
                y = torch.index_select(y, 0, indices)

            # clone and delete so that we dont store unnecessary full data
            x_test = x[test_split:].clone(); xx = x[:test_split].clone(); del x; x = xx
            y_test = y[test_split:].clone(); yy = y[:test_split].clone(); del y; y = yy

        # ensure shapes
        if x.ndim > 2: raise ValueError(x.shape)
        if y.ndim > 2: raise ValueError(y.shape)
        # if y.ndim == 1: y = y.unsqueeze(1)

        if x_test is not None:
            assert y_test is not None
            if x_test.ndim > 2: raise ValueError(x_test.shape)
            if y_test.ndim > 2: raise ValueError(y_test.shape)
            # if y_test.ndim == 1: y_test = y_test.unsqueeze(1)

        # normalize
        if normalize_x:
            mean = x.mean(0, keepdim=True); std = x.std(0, keepdim=True)
            x = (x - mean) / std
            if x_test is not None: x_test = (x_test - mean) / std
        if normalize_y:
            mean = y.mean(0, keepdim=True); std = y.std(0, keepdim=True)
            y = (y - mean) / std
            if y_test is not None: y_test = (y_test - mean) / std


        # if batch size is None we pass train and test data at once for ultra speed
        if batch_size is None:
            # stack train and test data
            if x_test is None:
                X = x.to(data_device)
                Y = y.to(data_device)
                self.test_start_idx = None
            else:
                assert y_test is not None
                X = torch.cat([x, x_test]).to(data_device)
                Y = torch.cat([y, y_test]).to(data_device)
                self.test_start_idx = len(x)

            dltrain = ((X, Y), )
            dltest = None

        # otherwise make dataloaders
        else:
            self.test_start_idx = None
            x = x.to(data_device); y = y.to(data_device)
            dltrain = TensorDataLoader((x, y), batch_size=batch_size, shuffle=True, seed = seed)
            if x_test is not None:
                assert y_test is not None
                x_test = x_test.to(data_device); y_test = y_test.to(data_device)
                if test_batch_size is None:
                    dltest = [(x_test, y_test)]
                else:
                    dltest = TensorDataLoader((x_test, y_test), batch_size=batch_size, shuffle=False, seed = seed)
            else:
                dltest = None

        super().__init__(dltrain=dltrain, dltest=dltest, log_projections=True, seed = seed)

        self.model = model
        self.criterion = criterion

        self._make_images = decision_boundary
        if decision_boundary:
            assert len(x[0]) == 2, x.shape
            assert y.ndim == 1 or len(y[0]) in (1,2), y.shape
            if y.ndim == 1: y = y.unsqueeze(1)
            if len(y[0]) == 2: y = y.argmax(1)

            domain = totensor([
                [x[:, 0].min().detach().cpu().item() - 1, x[:, 1].min().detach().cpu().item() - 1],
                [x[:, 0].max().detach().cpu().item() + 1, x[:, 1].max().detach().cpu().item() + 1]], device=data_device,dtype=x_dtype)

            # grid of all possible samples
            x_lin = torch.linspace(domain[0][0], domain[1][0], resolution, device=data_device)
            y_lin = torch.linspace(domain[0][1], domain[1][1], resolution, device=data_device)
            xx, yy = torch.meshgrid(x_lin, y_lin, indexing="xy")
            grid_points = torch.stack((xx.ravel(), yy.ravel()), dim=1).float()

            # X_train in grid_points coords
            x_step = x_lin[1] - x_lin[0]
            y_step = y_lin[1] - y_lin[0]
            X_train_grid = ((x - domain[0]) / torch.stack((x_step, y_step))).round().int()

            # mask to quickly display dataset points on the image
            mask = torch.zeros((resolution, resolution), dtype = torch.bool, device = data_device)
            data = torch.zeros((resolution, resolution), dtype = torch.float32, device = data_device)
            data[*X_train_grid.T.flip(0)] = normalize(y.squeeze(), 0, 255)
            mask[*X_train_grid.T.flip(0)] = True

            self.grid_points = nn.Buffer(grid_points)
            self.mask = nn.Buffer(mask)
            self.data = nn.Buffer(data)

            self.resolution = resolution
            self.set_display_best('image')
            self.boundary_act = boundary_act



    def penalty(self, preds):
        return 0

    def get_loss(self):
        self.model.train()
        x, y = self.batch
        device = self.device
        x = x.to(device); y = y.to(device)
        y_hat = self.model(x)
        penalty = self.penalty(y_hat)

        if self.test_start_idx is not None:
            loss = self.criterion(y_hat, y, reduction='none')

            train_loss = loss[:self.test_start_idx].mean() + penalty
            test_loss = loss[self.test_start_idx:].mean() + penalty

            self.log('test loss', test_loss, log_test=False)

        else:
            train_loss = self.criterion(y_hat, y) + penalty

        # decision boundary
        if self._make_images:
            self.model.eval()
            with torch.inference_mode():
                out: torch.Tensor = self.model(self.grid_points)
                if self.boundary_act is not None: out = self.boundary_act(out)
                Z: torch.Tensor = (out * 255).reshape(self.resolution, self.resolution).unsqueeze(0).repeat((3,1,1))
                Z[0] = 255 - Z[0] # 0 is red, 1 is green
                Z[2] = 0
                Z = torch.where(self.mask, self.data, Z)
                Z = Z.clamp_(0, 255)
                self.log("image", Z.to(torch.uint8), log_test=False, to_uint8=False)
                self.log_difference("image update", Z, to_uint8=True)

        return train_loss
