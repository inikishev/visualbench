
import numpy as np
import torch
from torch.nn import functional as F
from .dataset import DatasetBenchmark
def generate_correlated_logistic_data(
    n_samples=100_000,
    n_features=32,
    n_classes=10,
    n_correlated=768,
    correlation=0.99,
    seed=0
):
    assert n_classes >= 2
    generator = np.random.default_rng(seed)

    X = generator.standard_normal(size=(n_samples, n_features))
    weights = generator.uniform(-2, 2, size=(n_features, n_classes))

    used_pairs = set()
    n_correlated = min(n_correlated, n_features * (n_features - 1) // 2)

    for _ in range(n_correlated):
        idxs = None
        while idxs is None or idxs in used_pairs:
            pair = generator.choice(n_features, size=2, replace=False)
            pair.sort()
            idxs = tuple(pair)

        used_pairs.add(idxs)
        idx1, idx2 = idxs

        noise = generator.standard_normal(n_samples) * np.sqrt(1 - correlation**2)
        X[:, idx2] = correlation * X[:, idx1] + noise

        w = generator.integers(1, 51)
        cls = generator.integers(0, n_classes)
        weights[idx1, cls] = w
        weights[idx2, cls] = -w

    logits = X @ weights

    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    y_one_hot = generator.multinomial(1, pvals=probabilities)
    y = np.argmax(y_one_hot, axis=1)

    X -= X.mean(0, keepdims=True)
    X /= X.std(0, keepdims=True)

    return X, y.astype(np.int64)


class Collinear(DatasetBenchmark):
    """Synthetic dataset with a lot of multicollinearity"""
    def __init__(
        self,
        model,
        batch_size=None,
        test_batch_size=None,
        n_samples=100_000,
        n_features=32,
        n_classes=10,
        n_correlated=768,
        correlation=0.99,
        criterion = F.cross_entropy,
        train_split = 0.8,
        seed=0,
    ):
        X, y = generate_correlated_logistic_data(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_correlated=n_correlated, correlation=correlation, seed=seed)
        super().__init__(
            (X, y),
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_split=train_split,
            dtypes=(torch.float32, torch.long),
        )