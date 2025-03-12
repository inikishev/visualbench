from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from light_dataloader import TensorDataLoader
from myai.transforms import normalize as _normalize, totensor
from torch import nn

from ..._utils import CUDA_IF_AVAILABLE
from ...benchmark import Benchmark


class DatasetBenchmark(Benchmark):
    def __init__(
        self,
        data_train,
        # x,
        # y,
        model: torch.nn.Module,
        criterion: Callable,
        batch_size: int | None = None,
        test_batch_size: int | None = None,
        data_test=None,
        test_split: int | float | None = None,
        shuffle_split=False,
        normalize: bool | Sequence[bool] = False,
        dtypes: torch.dtype | Sequence[torch.dtype] = torch.float32,
        data_device: torch.types.Device = CUDA_IF_AVAILABLE,
        decision_boundary = False,
        resolution = 192,
        boundary_act = None,
        seed = 0
    ):
        """dataset benchmark

        Args:
            data_train (Any): sequence of things to collate, e.g. (X, y)
            model (torch.nn.Module): model.
            criterion (Callable): loss function.
            batch_size (int | None, optional): batch size, None for fullbatch. Defaults to None.
            test_batch_size (int | None, optional): test batch size, None for fullbatch. Defaults to None.
            data_test (_type_, optional): sequence of things to collate, e.g. (X, y)
            test_split (int | float | None, optional):
                splits data_train into train and test, can be int or float. Defaults to None.
            shuffle_split (bool, optional):
                whether to shuffle before splitting data_train into train and test sets. Defaults to False.
            normalize (bool | Sequence[bool], optional):
                whether to normalize each element of data_train and data_test along first dim. Defaults to False.
            dtypes (torch.dtype | Sequence[torch.dtype], optional):
                dtypes for each element of data_train and data_test. Defaults to torch.float32.
            data_device (torch.types.Device, optional): device for data_train and data_test. Defaults to CUDA_IF_AVAILABLE.
            decision_boundary (bool, optional):
                if True, dataset needs to be 2d, will make decision boundary animation. Defaults to False.
            resolution (int, optional):
                resolution for decision boundary. Defaults to 192.
            boundary_act (_type_, optional):
                activation for decision boundary for example when you use BCE with logits you can put torch.sigmoid. Defaults to None.
            seed (int, optional): seed. Defaults to 0.
        """
        if batch_size is None and test_batch_size is not None: raise NotImplementedError('batch_size is None, but test_batch size is not None')

        data_train = [totensor(i) for i in data_train]

        if len(set(len(i) for i in data_train)) != 1:
            raise ValueError(f"Got different number of elements in data_train: {[len(i) for i in data_train]}")

        n_samples = len(data_train[0])

        # create test dataset from data_test
        if data_test is not None:
            data_test = [totensor(i) for i in data_test]
            if len(set(len(i) for i in data_test)) != 1:
                raise ValueError(f"Got different number of elements in data_test: {[len(i) for i in data_test]}")

        # if data_test is None and test_split is not None, split data into train and test based on test_split
        elif test_split is not None:
            if isinstance(test_split, float):
                test_split = int(test_split * n_samples)

            # shuffle data before splitting to train/test
            if shuffle_split:
                indices = torch.randperm(n_samples, generator=torch.Generator(data_train[0].device).manual_seed(seed), device=data_train[0].device)
                data_train = [torch.index_select(i, 0, indices) for i in data_train]

            # split
            data_test = [i[test_split:] for i in data_train] # needs to be 1st
            data_train = [i[:test_split] for i in data_train]

        # normalize along 1st dim
        if isinstance(normalize, bool): normalize = [normalize] * len(data_train)

        def _norm(x: torch.Tensor, normalize):
            if not normalize: return x
            mean = x.mean(0, keepdim=True); std = x.std(0, keepdim=True)
            x = (x - mean) / std
            return x

        data_train = [_norm(i, n) for i,n in zip(data_train, normalize)]
        if data_test is not None: data_test = [_norm(i, n) for i,n in zip(data_test, normalize)]

        if not isinstance(dtypes, Sequence): dtypes = [dtypes] * len(data_train)

        # if batch size is None we pass train and test data at once for ultra speed
        if batch_size is None:

            # stack train and test data
            if data_test is None:
                data_train = [i.to(device = data_device, dtype = dt) for i, dt in zip(data_train, dtypes)]
                self.test_start_idx = None
            else:
                self.test_start_idx = len(data_train[0])
                data_train = [torch.cat([tr,te]).to(data_device, dtype = dt) for tr,te, dt in zip(data_train, data_test, dtypes)]

            dltrain = (data_train, )
            dltest = None

        # otherwise make dataloaders
        else:
            self.test_start_idx = None
            data_train = [i.to(device = data_device, dtype = dt) for i, dt in zip(data_train, dtypes)]
            dltrain = TensorDataLoader(data_train, batch_size=batch_size, shuffle=True, seed = seed)

            if data_test is not None:
                data_test = [i.to(device = data_device, dtype = dt) for i, dt in zip(data_test, dtypes)]
                if test_batch_size is None:
                    dltest = (data_test, )
                else:
                    dltest = TensorDataLoader(data_test, batch_size=test_batch_size, shuffle=False, seed = seed)
            else:
                dltest = None

        super().__init__(dltrain=dltrain, dltest=dltest, log_projections=True, seed = seed)

        self.model = model
        self.criterion = criterion

        self._make_images = decision_boundary
        if decision_boundary:
            if len(data_train) != 2: raise ValueError(f'{len(data_train) = }, needs to be 2 for decision boundary')
            x,y = data_train
            x_dtype, y_dtype = dtypes

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
            data[*X_train_grid.T.flip(0)] = _normalize(y.squeeze(), 0, 255)
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
        device = self.device
        self.model.train()

        if len(self.batch) == 1: x = y = self.batch[0].to(device)
        else: x, y = [i.to(device) for i in self.batch]

        y_hat = self.model(x)
        if isinstance(y_hat, tuple): y_hat, penalty = y_hat
        else: penalty = 0
        penalty = penalty + self.penalty(y_hat)

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
