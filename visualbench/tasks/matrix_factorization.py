import urllib.request
import zipfile
from itertools import cycle
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..benchmark import Benchmark
from ..utils import maybe_per_sample_loss
from ..utils.light_dataloader import TensorDataLoader
from ..utils._benchmark_utils import _should_run_test_epoch

class _MatrixFactorization(nn.Module):
    def __init__(self, n1, n2, n_factors=20):
        super().__init__()
        self.factors1 = nn.Embedding(n1, n_factors)
        self.factors2 = nn.Embedding(n2, n_factors)
        nn.init.normal_(self.factors1.weight, std=0.01)
        nn.init.normal_(self.factors2.weight, std=0.01)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):
        return (self.factors1(x1) * self.factors2(x2)).sum(1)

class MFMovieLens(Benchmark):
    """download https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset?resource=download, unpack and specify path to that folder"""
    def __init__(self, path, n_factors: int = 32, batch_size: int | None = 1024, criterion=F.mse_loss, device=None):
        super().__init__()
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.data_dir = Path(path)

        ds_train, ds_test, n_users, n_items = self._load_movie_lens()
        self.model = _MatrixFactorization(n_users, n_items, self.n_factors)

        # so we need 2 dataloaders so the test_every can't be used
        if batch_size is None:
            self.dl_train = ([t.to(device=device) for t in ds_train], )
            self.dl_train = ([t.to(device=device) for t in ds_test], )
        else:
            self.dl_train = TensorDataLoader([t.to(device=device) for t in ds_train], batch_size=batch_size, shuffle=True)
            self.dl_test = TensorDataLoader([t.to(device=device) for t in ds_test], batch_size=4096)

        self.iter_train = cycle(self.dl_train)
        self.criterion = criterion
        self.to(device)
        self.set_multiobjective_func(torch.mean)

    def _load_movie_lens(self):
        data_path = self.data_dir / "u.data"
        import pandas as pd

        df = pd.read_csv(
            data_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )

        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1

        n_users = df['user_id'].max() + 1
        n_items = df['item_id'].max() + 1

        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=self.rng.seed)

        train_users = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        train_items = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)

        test_users = torch.tensor(test_df['user_id'].values, dtype=torch.long)
        test_items = torch.tensor(test_df['item_id'].values, dtype=torch.long)
        test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float32)

        return (
            (train_users, train_items, train_ratings),
            (test_users, test_items, test_ratings),
            n_users, n_items
        )

    @torch.no_grad
    def _test_epoch(self):
        self.model.eval() # not calling self.eval
        test_loss = 0

        for users, items, ratings in self.dl_test:
            users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
            preds = self.model(users, items)
            test_loss += self.criterion(preds, ratings)

        self.log("test loss", test_loss / len(self.dl_test))
        self.model.train()

    def get_loss(self):
        """Performs one training step and returns the training loss."""
        users, items, ratings = next(self.iter_train)
        users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)

        preds = self.model(users, items)
        loss = maybe_per_sample_loss(self.criterion, (preds,ratings), per_sample=self._multiobjective)

        if _should_run_test_epoch(self, check_dltest=False):
            self._test_epoch()

        return loss

