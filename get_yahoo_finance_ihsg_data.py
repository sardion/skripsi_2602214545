#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download historical IHSG (^JKSE) data from Yahoo Finance.

Behavior:
1. Download daily historical data for IHSG (^JKSE) covering 1 January 2019 – 31 August 2025.
2. Yahoo Finance returns columns:
     Date, Open, High, Low, Close, Adj Close, Volume
3. Round all price columns to integer (matching IDX price format).
4. Save the result as:
     ./data/raw/ihsg_raw.csv
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import pandas as pd
from pandas import DataFrame
import yfinance as yf

# =====================================================================================
# CONFIGURATION
# =====================================================================================

SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE_NAME: str = "ihsg_raw.csv"
YAHOO_SYMBOL: str = "^JKSE"  # Yahoo Finance ticker for IHSG (Jakarta Composite Index)

START_DATE_STR: str = "2019-01-01"
END_DATE_STR_INCLUSIVE: str = "2025-08-31"


# =====================================================================================
# FUNCTIONS
# =====================================================================================

def _inclusive_end_for_yf(end_date_inclusive: str) -> str:
    """Add +1 day to include end date since Yahoo's 'end' parameter is exclusive."""
    end_dt = datetime.strptime(end_date_inclusive, "%Y-%m-%d")
    end_plus_one = end_dt + timedelta(days=1)
    return end_plus_one.strftime("%Y-%m-%d")


def _download_ihsg_history_yf(
    yahoo_symbol: str,
    start_date_str: str,
    end_date_inclusive_str: str
) -> DataFrame:
    """Download IHSG (^JKSE) OHLCV data from Yahoo Finance."""
    end_exclusive_str = _inclusive_end_for_yf(end_date_inclusive_str)
    ticker_obj = yf.Ticker(yahoo_symbol)
    df_hist: DataFrame = ticker_obj.history(
        start=start_date_str,
        end=end_exclusive_str,
        auto_adjust=False,
        actions=False
    )
    df_hist.index = pd.to_datetime(df_hist.index, errors="coerce")
    df_hist = df_hist[~df_hist.index.isna()].sort_index()
    return df_hist


def _round_price_columns(df: DataFrame) -> DataFrame:
    """Round price columns (Open, High, Low, Close, Adj Close) to nearest integer."""
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")
    return df


def _finalize_and_save(df_hist: DataFrame, output_dir: str, file_name: str) -> None:
    """Finalize, round, and save IHSG data to CSV."""
    df_out: DataFrame = _round_price_columns(df_hist.copy())
    df_out.index.name = "Trading Date"

    out_path: str = os.path.join(output_dir, file_name)
    try:
        df_out.to_csv(out_path, index=True)
        print(f"[OK] Saved IHSG data to {out_path} with {len(df_out)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to save IHSG data: {e}")


# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main() -> None:
    print("[INFO] Downloading IHSG (^JKSE) data from Yahoo Finance...")
    df_hist = _download_ihsg_history_yf(
        YAHOO_SYMBOL, START_DATE_STR, END_DATE_STR_INCLUSIVE
    )

    if df_hist.empty:
        print("[WARN] No data returned for IHSG (^JKSE).")
        return

    _finalize_and_save(df_hist, OUTPUT_DIR, OUTPUT_FILE_NAME)
    print("[DONE] IHSG data download complete.")


if __name__ == "__main__":
    main()
