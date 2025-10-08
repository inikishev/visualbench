from collections.abc import Callable
from os import PathLike
from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ...utils import CUDA_IF_AVAILABLE
from .dataset import DatasetBenchmark

if TYPE_CHECKING:
    import polars as pl

def _load_one_hot_col(df: "pl.DataFrame", col:str, max_length:int):
    import polars as pl
    letters = df[col].str.split("").explode().unique(maintain_order=True).to_list()
    letters_map = {l:i+1 for i,l in enumerate(letters)}
    letters_map[" "] = 0

    df = df.with_columns(
        pl.col(col)
        .str.pad_end(max_length)
        .str.slice(0, max_length)
        .str.split("")
        .list.eval(pl.element().replace(letters_map).cast(int))
    )

    X_int = torch.from_numpy(np.stack(df[col].to_numpy()))# pyright:ignore[reportCallIssue, reportArgumentType]

    X = F.one_hot(X_int, 27).moveaxis(-1,-2) # pylint:disable=not-callable

    return X

class EnglishWords(DatasetBenchmark):
    """please download csv from https://www.kaggle.com/datasets/victorcheng42/english-words-with-stress-position-analyzed?resource=download and specify path to it

    Input/output shapes:
    - ``"word"`` - ``(batch_size, 27, length)```
    - ``"phonetic"`` - ``(batch_size, 87, length)```
    - ``"stress_syllable"`` - ``(batch_size, 16, 2)```
    - ``"stress_pos"`` - ``(batch_size, something)``
    - ``"syllable_len"`` - ``(batch_size, something)``
    """

    def __init__(
        self,
        csv_path: str | PathLike,
        model: torch.nn.Module,
        batch_size: int | None,
        test_batch_size: int | None,
        input: Literal["word", "phonetic"] = "word",
        target: Literal["stress_pos", "syllable_len", "word", "phonetic", "stress_syllable"] = "stress_pos",
        input_length: int = 16,
        target_length: int = 16,
        criterion: Callable = F.mse_loss,
        train_split: float | int = 0.8,
        normalize_X: bool = True,
        normalize_y: bool = True,
        device=CUDA_IF_AVAILABLE,
    ):
        import polars as pl
        if target == "stress_syllable": target_length = 2

        df = pl.read_csv(str(csv_path))

        df = df.filter(pl.col(input).str.len_chars().le(input_length))
        if target in ("word", "phonetic", "stress_syllable"):
            df = df.filter(pl.col(target).str.len_chars().le(target_length))

        df = df.drop_nulls([input, target])

        X = _load_one_hot_col(df, input, input_length)

        if target in ("stress_pos", "syllable_len"):
            y = F.one_hot(df[target].to_torch().long()) # pylint:disable=not-callable
        else:
            if target == "stress_syllable": target_length = 2
            y = _load_one_hot_col(df, target, target_length)

        # print(f'{X.shape = }')
        # print(f'{y.shape = }')
        super().__init__(
            data_train=(X, y),
            model=model,
            criterion=criterion,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            train_split=train_split,
            shuffle_split=True,
            normalize=(normalize_X, normalize_y),
            dtypes = (torch.float32, torch.float32),
            data_device=device,
        )