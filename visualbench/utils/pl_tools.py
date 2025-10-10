from typing import Any
from collections.abc import Sequence, Callable
import polars  as pl
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_classification_csv(file, target_cols: str | Sequence[str], one_hot_cols: str | Sequence[str] | None = None, scaler: Any = StandardScaler()):
    df = pl.read_csv(file)

    if one_hot_cols is not None:
        df = df.with_columns(df.to_dummies(one_hot_cols)).drop(one_hot_cols)

    X = df.select(pl.exclude(target_cols)).to_numpy()
    y = df.select(target_cols).to_numpy()

    y = np.stack([LabelEncoder().fit_transform(t) for t in y.T], -1)

    X = scaler.fit_transform(np.asarray(X))
    return X, y


def plot_corr(X):
    import seaborn as sns
    sns.heatmap(pl.DataFrame(X).corr(), cmap='coolwarm')
