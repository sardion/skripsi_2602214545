            #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_build_stock_ta_indicators.py

Build per-stock technical indicator features:
- date
- close_price
- rsi_7, rsi_14
- sma_7, sma_14, sma_21, sma_100
- std_dev_7, std_dev_14, std_dev_21, std_dev_100

Input:
- data/raw/yf_ticker_summary_csv/yf_{kode_emiten}.csv
  Columns: Trading Date, Open, High, Low, Close, Adj Close, Volume

Output:
- data/raw/{kode_emiten}_ta_indicators_features_raw.csv

Notes:
- Default tickers: ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]
- For each ticker, compute RSI, SMA, and StdDev.
- Initial periods without sufficient history are filled with 0.
- All numeric outputs rounded to 3 decimal places.
"""

from __future__ import annotations
from pathlib import Path
from typing import Final, List
import pandas as pd
import numpy as np


# =========================
# Config (relative paths)
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
YF_DIR: Final[Path] = RAW_DIR / "yf_ticker_summary_csv"

DEFAULT_TICKERS: Final[List[str]] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]


# =========================
# Technical Indicator Helpers
# =========================
def _clean_numeric(s: pd.Series) -> pd.Series:
    """Remove thousand separators/spaces and cast to float."""
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .replace({"": None})
         .astype(float)
    )


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    """Compute Relative Strength Index (RSI) for given window."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi).fillna(0.0)


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean().fillna(0.0)


def compute_std(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling standard deviation."""
    return series.rolling(window=window, min_periods=window).std().fillna(0.0)


# =========================
# Core Logic
# =========================
def process_ticker(ticker: str) -> None:
    """
    Process one ticker to generate technical indicator features and save output CSV.
    """
    input_csv = YF_DIR / f"yf_{ticker}_raw.csv"
    output_csv = RAW_DIR / f"{ticker}_ta_indicators_features_raw.csv"

    if not input_csv.exists():
        print(f"[WARN] Missing input file for {ticker}: {input_csv}")
        return

    # Read and clean data
    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    if "Trading Date" not in df.columns or "Close" not in df.columns:
        print(f"[ERROR] Missing required columns in {ticker} CSV")
        return

    # Parse and clean
    df["date"] = pd.to_datetime(df["Trading Date"], errors="raise").dt.normalize()
    df["close_price"] = _clean_numeric(df["Close"])
    df = df[["date", "close_price"]].sort_values("date").reset_index(drop=True)

    # Compute features
    df["rsi_7"] = compute_rsi(df["close_price"], 7)
    df["rsi_14"] = compute_rsi(df["close_price"], 14)

    df["sma_7"] = compute_sma(df["close_price"], 7)
    df["sma_14"] = compute_sma(df["close_price"], 14)
    df["sma_21"] = compute_sma(df["close_price"], 21)
    df["sma_100"] = compute_sma(df["close_price"], 100)

    df["std_dev_7"] = compute_std(df["close_price"], 7)
    df["std_dev_14"] = compute_std(df["close_price"], 14)
    df["std_dev_21"] = compute_std(df["close_price"], 21)
    df["std_dev_100"] = compute_std(df["close_price"], 100)

    # Round to 3 decimals
    df = df.round(3)

    # Save output
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {output_csv.name} with {len(df)} rows")


# =========================
# Main
# =========================
def main() -> None:
    print("[INFO] Starting feature_build_stock_ta_indicators.py")
    for ticker in DEFAULT_TICKERS:
        process_ticker(ticker)
    print("[DONE] All technical indicator features generated successfully.")


if __name__ == "__main__":
    main()
