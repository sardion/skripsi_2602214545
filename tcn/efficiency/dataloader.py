#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for supervised LSTM training.

This module loads sliding-window feature sets from the global `FEATURES_DIR`.
It supports two input modes:

1) **Pre-windowed** tables that already include a `target` column.
   - If the feature matrix is flattened (B, T*F), we reshape to (B, T, F).
   - Otherwise, we assume a proper (B, T, F) structure stored under "_X".

2) **Raw long-format OHLCV** tables (calendar-aligned).
   - We construct sliding windows of length `cfg.window_size` and build a
     one-step-ahead `target`:
       * target_kind == "close": next day's close
       * target_kind == "mean" : next day's arithmetic mean of OHLC

No side effects occur at import. File discovery tries `<stem>.parquet` then `<stem>.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Tuple, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from config import FEATURES_DIR, Config

__all__ = [
    "_read_table",
    "_is_prewindowed",
    "_make_windows_from_raw",
    "WindowedDataset",
    "load_dataset",
    "make_loaders",
]


def _read_table(basepath: Path) -> pd.DataFrame:
    """
    Read a feature table from `<basepath>.parquet` or `<basepath>.csv`.

    Parameters
    ----------
    basepath : Path
        Feature file path **without** extension (stem).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If neither the `.parquet` nor `.csv` file exists.
    """
    if basepath.with_suffix(".parquet").exists():
        return pd.read_parquet(basepath.with_suffix(".parquet"))
    if basepath.with_suffix(".csv").exists():
        return pd.read_csv(basepath.with_suffix(".csv"))
    raise FileNotFoundError(f"Feature file not found as .parquet or .csv: {basepath}")


def _is_prewindowed(df: pd.DataFrame) -> bool:
    """
    Check whether the table is already pre-windowed (i.e., contains a `target` column).

    Parameters
    ----------
    df : pd.DataFrame
        Input table.

    Returns
    -------
    bool
        True if the DataFrame contains a `target` column, False otherwise.
    """
    return "target" in df.columns


def _make_windows_from_raw(df: pd.DataFrame, cfg: Config, target_kind: str) -> pd.DataFrame:
    """
    Construct sliding windows and next-step target from a raw calendar-aligned table.

    For `target_kind == "close"`, the target is `close_price.shift(-1)`.
    For `target_kind == "mean"`, the target is `mean_price.shift(-1)` (REQUIRED).

    Output schema:
      - `_X` : List of (T, F) float32 numpy arrays (windowed features)
      - `target` : float (next-step target)
    """
    # Resolve column names case-insensitively
    cols_lower = {c.lower(): c for c in df.columns}

    if target_kind == "close":
        if "close_price" not in cols_lower:
            raise ValueError(
                "Raw mode requires 'close_price' to compute close target. "
                f"Seen columns: {list(df.columns)}"
            )
        close_col = cols_lower["close_price"]
        target_series = df[close_col].shift(-1).rename("target")

    else:  # target_kind == "mean"
        if "mean_price" not in cols_lower:
            raise ValueError(
                "Raw mode requires 'mean_price' for the mean target (no OHLC fallback). "
                "Please ensure your feature file includes a 'mean_price' column. "
                f"Seen columns: {list(df.columns)}"
            )
        mean_col = cols_lower["mean_price"]
        target_series = df[mean_col].shift(-1).rename("target")

    dfw = df.copy().reset_index(drop=True)
    dfw["target"] = target_series
    dfw = dfw.dropna().reset_index(drop=True)

    feature_cols = [c for c in dfw.columns if c != "target"]
    X_list, y_list = [], []
    T = cfg.window_size
    for i in range(len(dfw) - T):
        x_slice = dfw.loc[i : i + T - 1, feature_cols]
        y_val = dfw.loc[i + T, "target"]
        X_list.append(x_slice.to_numpy(dtype=np.float32))
        y_list.append(float(y_val))

    if not X_list:
        raise ValueError(f"Not enough rows to construct windows with T={T}.")

    return pd.DataFrame({"_X": X_list, "target": y_list})


class WindowedDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Torch dataset for windowed features + scalar target.

    The constructor accepts either:
      - a pre-windowed table with column `_X` (each entry is (T, F) float32 array),
      - or a flat pre-windowed table with shape (B, T*F) and a `target` column,
        which will be reshaped into (B, T, F) when `T` divides the feature count.

    Attributes
    ----------
    cfg : Config
        Training/evaluation configuration (immutable).
    X : NDArray[np.float32]
        Feature tensor as numpy array with shape (B, T, F).
    y : NDArray[np.float32]
        Target array with shape (B,).
    """

    def __init__(self, table: pd.DataFrame, cfg: Config) -> None:
        super().__init__()
        self.cfg: Final[Config] = cfg

        if "_X" in table.columns:
            self.X: npt.NDArray[np.float32] = np.stack(
                table["_X"].to_list(), axis=0
            ).astype(np.float32)  # (B, T, F)
            self.y: npt.NDArray[np.float32] = table["target"].to_numpy(np.float32)
        else:
            if "target" not in table.columns:
                raise ValueError("Pre-windowed mode requires a 'target' column.")
            feat_cols: List[str] = [c for c in table.columns if c != "target"]
            X_arr = table[feat_cols].to_numpy(np.float32)  # (B, T*F) or (B, F)
            T: int = self.cfg.window_size
            B: int = X_arr.shape[0]
            if X_arr.shape[1] % T == 0:
                F: int = X_arr.shape[1] // T
                self.X = X_arr.reshape(B, T, F)
            else:
                # Fallback: treat as a single timestep with F features.
                self.X = X_arr.reshape(B, 1, X_arr.shape[1])
            self.y = table["target"].to_numpy(np.float32)

    def __len__(self) -> int:
        """Return the number of windows (B)."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the (X, y) pair at a given index.

        Parameters
        ----------
        idx : int
            Row index in [0, B).

        Returns
        -------
        (Tensor, Tensor)
            X: float32 tensor with shape (T, F)
            y: float32 tensor with shape (,) — a scalar target
        """
        x_np: npt.NDArray[np.float32] = self.X[idx]
        y_np: npt.NDArray[np.float32] = np.array(self.y[idx], dtype=np.float32)
        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def load_dataset(ticker: str, target_kind: str, cfg: Config) -> WindowedDataset:
    """
    Load a dataset for the specified ticker and target kind.

    This function resolves the canonical feature filename stem using
    `config.feature_filename`, then loads a parquet/csv feature table,
    and returns a `WindowedDataset` (pre-windowed or raw-mode derived).

    Parameters
    ----------
    ticker : str
        Stock code (e.g., "BBCA").
    target_kind : str
        Either "close" or "mean".
    cfg : Config
        Configuration with `window_size` for raw-mode windowing.

    Returns
    -------
    WindowedDataset
        Torch dataset yielding (X, y) pairs where X has shape (T, F).
    """
    from config import feature_filename  # local import to avoid circulars at module import time

    stem: str = feature_filename(ticker, target_kind)  # filename without extension
    basepath: Path = FEATURES_DIR / stem
    df: pd.DataFrame = _read_table(basepath)

    if _is_prewindowed(df):
        table = df
    else:
        table = _make_windows_from_raw(df, cfg, target_kind)

    return WindowedDataset(table, cfg)


def make_loaders(
    ds: WindowedDataset,
    batch_size: int,
    splits: Tuple[tuple, tuple, tuple],
    num_workers: int = 0,
) -> Tuple[
    DataLoader[Tuple[Tensor, Tensor]],
    DataLoader[Tuple[Tensor, Tensor]],
    DataLoader[Tuple[Tensor, Tensor]],
    int,
]:
    """
    Construct train/valid/test dataloaders from index ranges and return input_dim.

    Parameters
    ----------
    ds : WindowedDataset
        The dataset to be split.
    batch_size : int
        Batch size for all loaders.
    splits : Tuple[tuple, tuple, tuple]
        Three index pairs: ((tr_s, tr_e), (va_s, va_e), (te_s, te_e)). Each range
        is clamped to [0, len(ds)].
    num_workers : int, optional
        Number of background worker processes for each DataLoader. Default is 0
        (single-threaded, safest and most reproducible).

    Returns
    -------
    (DataLoader, DataLoader, DataLoader, int)
        - train loader
        - validation loader
        - test loader
        - input_dim (F), derived from ds.X.shape[-1]

    Notes
    -----
    - `shuffle=True` is applied to the train loader only.
    - `drop_last=False` for all loaders to retain all samples.
    """
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = splits
    N: int = len(ds)

    tr_s = max(0, min(tr_s, N)); tr_e = max(0, min(tr_e, N))
    va_s = max(0, min(va_s, N)); va_e = max(0, min(va_e, N))
    te_s = max(0, min(te_s, N)); te_e = max(0, min(te_e, N))

    train: Subset[Tuple[Tensor, Tensor]] = Subset(ds, list(range(tr_s, tr_e)))
    valid: Subset[Tuple[Tensor, Tensor]] = Subset(ds, list(range(va_s, va_e)))
    test:  Subset[Tuple[Tensor, Tensor]] = Subset(ds, list(range(te_s, te_e)))

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=num_workers),
        DataLoader(valid, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        DataLoader(test,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers),
        int(ds.X.shape[-1]),  # input_dim (F)
    )
