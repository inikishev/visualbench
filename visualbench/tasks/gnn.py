from collections.abc import Callable
from typing import Any
import copy
import torch
from myai import nn as mynn
from myai.python_tools import reduce_dim
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, global_mean_pool

from ..benchmark import Benchmark
from .._utils import CUDA_IF_AVAILABLE


def _reduce_dim_filter_none(x):
    return [i for i in reduce_dim(x) if i is not None]

class Seq(torch.nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.seq = mynn.ModuleList(args)

    def forward(self,x, edge_index):
        for m in self.seq:
            if "conv" in m.__class__.__name__.lower(): x = m(x, edge_index)
            else: x = m(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden = (16, ), act: Any = F.relu, dropout = 0.0, conv_cls = GCNConv, global_pool=False):
        super().__init__()

        channels = [num_features] + list(hidden)
        self.convs = Seq(
            *_reduce_dim_filter_none(
                [[conv_cls(i,o), act, torch.nn.Dropout(dropout) if dropout>0 else None] for i, o in zip(channels[:-1], channels[1:])])
        )

        self.head = GCNConv(channels[-1], num_classes)
        self.global_pool = global_pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.convs(x, edge_index)
        x = self.head(x, edge_index)

        if self.global_pool: x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)



class GraphNN(Benchmark):
    def __init__(self, model_cls: Callable = GCN, dataset: Any = Planetoid(root='/var/mnt/hdd/Datasets/PyTorch Geometric', name='Cora'), loss_fn = F.nll_loss, device=CUDA_IF_AVAILABLE):
        super().__init__()

        self.model = model_cls(dataset.num_features, dataset.num_classes)
        self.model.train()
        self.model_backup = copy.deepcopy(self.model.cpu())
        try:
            self.data = dataset.to(device)
        except ValueError:
            dataset._data_list = None
            self.data = dataset.to(device)
        self.loss_fn = loss_fn

        if hasattr(self.data, "test_mask"):
            self.y_train = self.data.y[self.data.train_mask].to(device)
            self.y_test = self.data.y[self.data.test_mask].to(device)
        else:
            self.y_train = self.data.y.to(device)
            self.y_test = None

        self.to(device)

    def get_loss(self):
        out = self.model(self.data)
        if self.y_test is None: return self.loss_fn(out, self.y_train)

        train_loss = self.loss_fn(out[self.data.train_mask], self.y_train)
        test_loss = self.loss_fn(out[self.data.test_mask], self.y_test)

        self.log('test loss', test_loss, False)
        return train_loss


    def reset(self):
        super().reset()
        self.model = copy.deepcopy(self.model_backup).to(self.device)
        self.model.train()

# bench = TorchGeometricDataset()
# opt = torch.optim.Adam(bench.parameters(), 1e-2)
# bench.run(opt, 1000)
# bench.plot_loss()