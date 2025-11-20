#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading utilities for supervised time-series training.

This module loads sliding-window feature sets from the global `FEATURES_DIR`.
It provides two parallel pipelines:

1) LSTM/TCN-compatible (legacy):
   - Returns a single (T, F) tensor per sample with a scalar target.
   - Entry points: WindowedDataset, load_dataset, make_loaders
   - Raw-mode windowing via _make_windows_from_raw()

2) TFT-compatible (new for TFTLite):
   - Returns a dict per sample: {"past": (T, Dp), "future": (1, Df), "static": (Ds,)} with scalar target.
   - FUTURE features are aligned to the prediction index (1-step ahead).
   - STATIC defaults to a 1-dim constant if you have no true static columns.
   - Entry points: TFTWindowedDataset, load_dataset_tft
   - Raw-mode windowing via _make_tft_windows_from_raw()

No side effects occur at import. File discovery tries `<stem>.parquet` then `<stem>.csv`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Tuple, List, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

from config import FEATURES_DIR, Config

__all__ = [
    # Legacy (LSTM/TCN)
    "_read_table",
    "_is_prewindowed",
    "_make_windows_from_raw",
    "WindowedDataset",
    "load_dataset",
    "make_loaders",
    # TFT-specific
    "TFTWindowedDataset",
    "load_dataset_tft",
]

# ========================= Feature Group Definitions =========================
# IMPORTANT:
# - is_trading_day is time-varying → belongs to PAST (not STATIC).
# - next_is_trading_day is known deterministically for the next day → FUTURE.
# - STATIC should only contain true per-ticker constants; left empty by default.

PAST_COLS: Sequence[str] = [
    # === Price & Volume (Historical) ===
    "open_price", "high_price", "low_price", "close_price", "volume",
    "bid_volume", "offer_volume", "foreign_sell", "foreign_buy",

    # === Trading-day marker (realized) ===
    "is_trading_day",

    # === Macroeconomic (Lagged, Known from Past) ===
    "us_fed_fund_rate", "days_since_us_fed_fund_rate",
    "us_gdp_qoq", "days_since_us_gdp_qoq",
    "us_core_cpi_mom", "days_since_us_core_cpi_mom",
    "id_bi_rate", "days_since_id_bi_rate",
    "id_core_inflation_yoy", "days_since_id_core_inflation_yoy",
    "id_gdp_qoq", "days_since_id_gdp_qoq",
    "id_inflation_mom", "days_since_id_inflation_mom",
    "id_retail_sales_yoy", "days_since_id_retail_sales_yoy",
    "usd_idr", "usd_index_dxy", "crude_oil_wti", "sp500_vix", "gold_futures",

    # === Market & Technical Indicators (Computed from Past) ===
    "ihsg_return_lag_1", "ihsg_std_dev_7", "ihsg_std_dev_14", "ihsg_std_dev_21",
    "rsi_7", "rsi_14",
    "sma_7", "sma_14", "sma_21", "sma_100",
    "std_dev_7", "std_dev_14", "std_dev_21", "std_dev_100",
]

FUTURE_COLS: Sequence[str] = [
    # === Calendar (Known in Advance) ===
    "is_sunday", "is_monday", "is_tuesday", "is_wednesday",
    "is_thursday", "is_friday", "is_saturday",
    "is_month_start", "is_month_end",
    "next_is_trading_day",
]

STATIC_COLS: Sequence[str] = [
    # Keep empty unless you have true per-ticker constants
]

# ============================== Common Utilities ==============================

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

# ===================== Legacy (LSTM/TCN) Window Construction =====================

def _make_windows_from_raw(df: pd.DataFrame, cfg: Config, target_kind: str) -> pd.DataFrame:
    """
    Construct sliding windows and next-step target from a raw calendar-aligned table.

    For `target_kind == "close"`, the target is `close_price.shift(-1)`.
    For `target_kind == "mean"`, the target is `mean_price.shift(-1)` (REQUIRED here).

    Output schema:
      - `_X` : List of (T, F) float32 numpy arrays (windowed features)
      - `target` : float (next-step target)
    """
    cols_lower = {c.lower(): c for c in df.columns}

    if target_kind == "close":
        if "close_price" not in cols_lower:
            raise ValueError(
                "Raw mode requires 'close_price' to compute close target. "
                f"Seen columns: {list(df.columns)}"
            )
        close_col = cols_lower["close_price"]
        target_series = pd.to_numeric(df[close_col], errors="coerce").shift(-1)

    else:  # target_kind == "mean"
        if "mean_price" not in cols_lower:
            raise ValueError(
                "Raw mode requires 'mean_price' for the mean target (no OHLC fallback). "
                "Please ensure your feature file includes a 'mean_price' column. "
                f"Seen columns: {list(df.columns)}"
            )
        mean_col = cols_lower["mean_price"]
        target_series = pd.to_numeric(df[mean_col], errors="coerce").shift(-1)

    dfw = df.copy().reset_index(drop=True)
    dfw["target"] = target_series
    dfw = dfw.dropna().reset_index(drop=True)

    feature_cols = [c for c in dfw.columns if c != "target"]
    X_list, y_list = [], []
    T = cfg.window_size
    for i in range(len(dfw) - T):
        x_slice = dfw.loc[i: i + T - 1, feature_cols]
        y_val = dfw.loc[i + T, "target"]
        X_list.append(x_slice.to_numpy(dtype=np.float32))
        y_list.append(float(y_val))

    if not X_list:
        raise ValueError(f"Not enough rows to construct windows with T={T}.")

    return pd.DataFrame({"_X": X_list, "target": y_list})

# ============================ Legacy Dataset/Loader ============================

class WindowedDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Torch dataset for legacy windowed features + scalar target.

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
    Legacy loader for LSTM/TCN.

    Resolves canonical feature filename stem using `config.feature_filename`,
    loads a parquet/csv feature table, and returns a WindowedDataset (pre-windowed or raw-mode).
    """
    from config import feature_filename  # local import to avoid circulars

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

    Returns
    -------
    (DataLoader, DataLoader, DataLoader, int)
        - train loader
        - validation loader
        - test loader
        - input_dim (F), derived from ds.X.shape[-1]
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
        int(ds.X.shape[-1]),
    )

# ============================ TFT-Specific Pipeline ============================

def _compute_mean_from_ohlc(df: pd.DataFrame) -> pd.Series:
    """Compute arithmetic-mean price from any common OHLC naming."""
    cols = {c.lower(): c for c in df.columns}
    for need in (["open_price", "high_price", "low_price", "close_price"],
                 ["open", "high", "low", "close"]):
        if all(k in cols for k in need):
            o = pd.to_numeric(df[cols[need[0]]], errors="coerce")
            h = pd.to_numeric(df[cols[need[1]]], errors="coerce")
            l = pd.to_numeric(df[cols[need[2]]], errors="coerce")
            c = pd.to_numeric(df[cols[need[3]]], errors="coerce")
            return (o + h + l + c) / 4.0
    raise KeyError("Cannot compute mean_price: missing OHLC columns.")


def _make_tft_windows_from_raw(df: pd.DataFrame, cfg: Config, target_kind: str) -> pd.DataFrame:
    """
    Build TFT-style windows with three streams:

      _Xp: (T, D_past) over [i .. i+T-1]
      _Xf: (1, D_fut)  known-future at prediction index i+T (1-step ahead)
      _Xs: (D_static,) static vector (constant fallback = [1.0] if none)
      target: scalar at i+T

    FUTURE is aligned to the prediction index; PAST uses only historical rows.
    """
    dfw = df.copy().reset_index(drop=True)
    cols = {c.lower(): c for c in dfw.columns}

    # Target aligned to prediction index (i+T)
    if target_kind == "close":
        if "close_price" not in cols:
            raise ValueError("Raw mode requires 'close_price' for close target.")
        target_series = pd.to_numeric(dfw[cols["close_price"]], errors="coerce").shift(-1)
    else:  # "mean"
        if "mean_price" in cols:
            target_series = pd.to_numeric(dfw[cols["mean_price"]], errors="coerce").shift(-1)
        else:
            target_series = _compute_mean_from_ohlc(dfw).shift(-1)

    dfw["target"] = target_series
    dfw = dfw.dropna().reset_index(drop=True)

    present = set(dfw.columns)
    past_cols   = [c for c in PAST_COLS   if c in present]
    future_cols = [c for c in FUTURE_COLS if c in present]
    static_cols = [c for c in STATIC_COLS if c in present]
    use_constant_static = len(static_cols) == 0  # placeholder if no true static

    T = cfg.window_size
    Xp_list, Xf_list, Xs_list, y_list = [], [], [], []

    for i in range(len(dfw) - T):
        # PAST: window [i .. i+T-1]
        if past_cols:
            past_slice = dfw.loc[i:i+T-1, past_cols].to_numpy(dtype=np.float32)
        else:
            past_slice = np.zeros((T, 0), dtype=np.float32)

        # FUTURE: row at prediction index i+T, shape (1, D_fut)
        pred_idx = i + T
        if future_cols:
            fut_row = dfw.loc[pred_idx, future_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        else:
            fut_row = np.zeros((1, 0), dtype=np.float32)

        # STATIC: constant vector (or real static if provided)
        if use_constant_static:
            static_vec = np.array([1.0], dtype=np.float32)  # bias/context
        else:
            static_vec = dfw.loc[pred_idx, static_cols].to_numpy(dtype=np.float32)

        y_val = float(dfw.loc[pred_idx, "target"])

        Xp_list.append(past_slice)
        Xf_list.append(fut_row)
        Xs_list.append(static_vec)
        y_list.append(y_val)

    if not Xp_list:
        raise ValueError(f"Not enough rows to construct TFT windows with T={T}.")

    return pd.DataFrame({"_Xp": Xp_list, "_Xf": Xf_list, "_Xs": Xs_list, "target": y_list})


class TFTWindowedDataset(Dataset[Tuple[dict, Tensor]]):
    """
    Dataset for TFTLite dict-batches.

    Yields:
      ({'past': (T,Dp), 'future': (1,Df), 'static': (Ds,)}, target)

    Exposes:
      d_past, d_fut, d_static (for model initialization)
    """

    def __init__(self, table: pd.DataFrame, cfg: Config) -> None:
        super().__init__()
        self.cfg: Final[Config] = cfg

        required = {"_Xp", "_Xf", "_Xs", "target"}
        if not required.issubset(set(table.columns)):
            raise ValueError("TFTWindowedDataset requires columns: _Xp, _Xf, _Xs, target")

        self.Xp: npt.NDArray[np.float32] = np.stack(table["_Xp"].to_list(), axis=0).astype(np.float32)  # (B,T,Dp)
        self.Xf: npt.NDArray[np.float32] = np.stack(table["_Xf"].to_list(), axis=0).astype(np.float32)  # (B,1,Df)
        self.Xs: npt.NDArray[np.float32] = np.stack(table["_Xs"].to_list(), axis=0).astype(np.float32)  # (B,Ds)
        self.y:  npt.NDArray[np.float32] = table["target"].to_numpy(np.float32)

        self.d_past:   int = int(self.Xp.shape[-1])
        self.d_fut:    int = int(self.Xf.shape[-1])
        self.d_static: int = int(self.Xs.shape[-1])

    def __len__(self) -> int:
        return self.Xp.shape[0]

    def __getitem__(self, idx: int):
        batch = {
            "past":   torch.from_numpy(self.Xp[idx]),
            "future": torch.from_numpy(self.Xf[idx]),
            "static": torch.from_numpy(self.Xs[idx]),
        }
        y = torch.from_numpy(np.array(self.y[idx], dtype=np.float32))
        return batch, y


def load_dataset_tft(ticker: str, target_kind: str, cfg: Config) -> TFTWindowedDataset:
    """
    Build TFTWindowedDataset using raw calendar-aligned feature tables.
    We DO NOT rely on config.feature_filename() here to avoid picking legacy
    sliding-window stems. Instead, we probe common raw/normalized stems and
    open the first one that exists, preferring Parquet then CSV.

    Candidate stems (without extension) searched under FEATURES_DIR:
      - {T}_integrated_features_all_normalized
      - {T}_integrated_features_all
      - {T}_integrated_features_all_raw
      - fallback: use config.feature_filename() if it happens to be raw
    """
    from config import feature_filename  # local import; used only as last resort

    def _read_first_existing(dir_path: Path, stems: list[str]) -> pd.DataFrame:
        for stem in stems:
            parq = dir_path / f"{stem}.parquet"
            csv  = dir_path / f"{stem}.csv"
            if parq.exists():
                return pd.read_parquet(parq)
            if csv.exists():
                return pd.read_csv(csv)
        raise FileNotFoundError(
            "None of the expected raw/normalized feature files were found in "
            f"{dir_path}. Tried stems: {stems}"
        )

    # 1) Preferred raw/normalized stems
    preferred_stems = [
        f"{ticker}_integrated_features_all_normalized",
        f"{ticker}_integrated_features_all",
        f"{ticker}_integrated_features_all_raw",
    ]

    # 2) Try to read one of the preferred stems in FEATURES_DIR
    try:
        df = _read_first_existing(FEATURES_DIR, preferred_stems)
    except FileNotFoundError:
        # 3) Last resort: whatever config.feature_filename() points to (only if raw)
        #    Note: If this still points to a sliding-window file, we'll detect and rebuild
        #    from that table if it contains raw columns; otherwise we re-raise a clear error.
        stem = feature_filename(ticker, target_kind)
        basepath: Path = FEATURES_DIR / stem
        try:
            df = _read_table(basepath)
        except FileNotFoundError as e:
            # Re-raise with a clearer message naming the raw candidates we want.
            tried = ", ".join(preferred_stems + [stem])
            raise FileNotFoundError(
                f"Feature file not found. Expected one of: {tried} (as .parquet or .csv) "
                f"in {FEATURES_DIR}"
            ) from e

    # If the file is already a TFT pre-windowed table, use it directly.
    if {"_Xp","_Xf","_Xs","target"}.issubset(set(df.columns)):
        table = df
    # If it's an LSTM pre-windowed table, or any raw calendar-aligned table,
    # (re)build TFT windows from it.
    else:
        table = _make_tft_windows_from_raw(df, cfg, target_kind)

    return TFTWindowedDataset(table, cfg)

