#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform normalized daily features into sliding-window supervised datasets (CSV only),
emitting TWO lookbacks (w30, w120) and TWO targets (close_price, mean_price) per ticker.

Inputs
  data/raw_features/{KODE}_integrated_features_all_normalized.csv

Outputs
  data/features/{KODE}_sliding_window_close_price_w30.csv
  data/features/{KODE}_sliding_window_close_price_w120.csv
  data/features/{KODE}_sliding_window_mean_price_w30.csv
  data/features/{KODE}_sliding_window_mean_price_w120.csv

Notes
- For each window_size (e.g., 30, 120), each row contains lags t-window..t-1 as inputs, and target at t.
- Targets:
    1) close_price
    2) mean_price  = arithmetic mean of (open_price + high_price + low_price + close_price)/4
       If a 'mean_price' column already exists in the input CSV, it will be used as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, List

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_DIR: Final[Path] = Path("data/raw_features")
OUTPUT_DIR: Final[Path] = Path("data/features")

# Emit BOTH lookbacks to align with Chapter 3 scenarios:
# - 120 for accuracy scenarios
# - 30 for efficiency scenario
WINDOW_SIZES: Final[List[int]] = [30, 120]

# Five tickers as per thesis scope
TICKERS: Final[List[str]] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

# Two targets to emit; 'mean_price' is AMP per naming.
TARGET_COLUMNS: Final[List[str]] = ["close_price", "mean_price"]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_mean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'mean_price' column exists.
    If absent, compute it as the arithmetic mean of open/high/low/close.
    Operates on whatever scale the input columns are on (normalized in your pipeline).
    """
    if "mean_price" in df.columns:
        return df

    required = ["open_price", "high_price", "low_price", "close_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Cannot compute 'mean_price' because required columns are missing: "
            + ", ".join(missing)
        )

    out = df.copy()
    out["mean_price"] = out[required].mean(axis=1)
    return out


def _base_features_to_lag(df: pd.DataFrame) -> list[str]:
    """
    Return the list of base feature columns to be lagged.
    Excludes 'calendar_date' and any target columns to avoid leakage.
    Alphabetically sorted for deterministic column order.
    """
    exclude = {"calendar_date", *TARGET_COLUMNS}
    features = [c for c in df.columns if c not in exclude]
    if not features:
        raise ValueError("No feature columns found to lag after exclusions.")
    features.sort()
    return features


def _create_sliding_window(
    df: pd.DataFrame, window_size: int, target_name: str
) -> pd.DataFrame:
    """
    Create sliding-window supervised dataset for a given target and lookback.

    Inputs:
        df           : calendar-indexed (sorted) DataFrame
        window_size  : number of lookback days (e.g., 30 or 120)
        target_name  : 'close_price' or 'mean_price'

    Output:
        DataFrame with columns:
            {feature}_lag_{window} ... {feature}_lag_1, target_name
        First `window_size` rows are dropped to ensure complete lags.
    """
    if target_name not in df.columns:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame.")

    features = _base_features_to_lag(df)

    # Build lag blocks oldest→newest along the sequence axis
    # We concatenate along columns to form a wide table
    lag_blocks = []
    for lag in range(window_size, 0, -1):
        block = df[features].shift(lag)
        block.columns = [f"{col}_lag_{lag}" for col in features]
        lag_blocks.append(block)

    out = pd.concat(lag_blocks, axis=1)

    # Append target at time t
    out[target_name] = df[target_name]

    # Drop incomplete rows (top 'window_size' rows)
    out = out.iloc[window_size:].reset_index(drop=True)

    # Optional: enforce float32 for features to keep file size reasonable
    # (targets kept as float64 to avoid precision loss if desired)
    feature_cols = [c for c in out.columns if c != target_name]
    out[feature_cols] = out[feature_cols].astype(np.float32)
    out = out.copy()   # defragment DataFrame in memory
    return out


def _read_and_prepare_input(input_path: Path) -> pd.DataFrame:
    """Read input CSV, parse/sort calendar_date, ensure mean_price exists."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    if "calendar_date" not in df.columns:
        raise KeyError("Input CSV must contain 'calendar_date' column.")

    # Sort by calendar_date for deterministic chronological order
    try:
        df["calendar_date"] = pd.to_datetime(df["calendar_date"])
        df = df.sort_values("calendar_date").reset_index(drop=True)
    except Exception:
        logger.warning("calendar_date not parsed as datetime; keeping original order.")

    # Ensure mean_price exists
    df = _ensure_mean_price(df)

    return df


def process_ticker(ticker: str) -> None:
    """Process one ticker for BOTH targets and BOTH window sizes."""
    input_path = INPUT_DIR / f"{ticker}_integrated_features_all_normalized.csv"
    df = _read_and_prepare_input(input_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        for win in WINDOW_SIZES:
            try:
                sw = _create_sliding_window(df, win, target_name=target)
                suffix_target = "close_price" if target == "close_price" else "mean_price"
                output_path = OUTPUT_DIR / f"{ticker}_sliding_window_{suffix_target}_w{win}.csv"
                sw.to_csv(output_path, index=False)
                logger.info(
                    f"{ticker} | target={target} | w={win} → {output_path.name} "
                    f"({sw.shape[0]} rows, {sw.shape[1]} cols)"
                )
            except Exception as exc:
                logger.error(f"{ticker} | target={target} | w={win} failed: {exc}")


def main() -> None:
    for code in TICKERS:
        try:
            process_ticker(code)
        except Exception as exc:
            logger.error(f"Processing {code} failed: {exc}")


if __name__ == "__main__":
    main()
