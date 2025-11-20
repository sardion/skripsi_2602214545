#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download split-adjusted historical stock data from Yahoo Finance for selected IDX tickers.

Behavior:
1. Define a manual list of base tickers (e.g. ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"])
2. Convert each to Yahoo Finance symbol by appending ".JK" (e.g. "BBCA.JK")
3. For each ticker:
   - Download historical OHLCV data for the date range start_date -> end_date
   - Yahoo Finance returns:
        Date (index)
        Open, High, Low, Close, Adj Close, Volume
     These are adjusted for splits/dividends (Adj Close).
   - Round all price columns to integer to match IDX price quoting conventions.
4. Save each ticker's data to:
     ./data/raw/yf_ticker_summary_csv/yf_<TICKER>_raw.csv
   where <TICKER> is the base ticker (e.g. "BBCA").
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List
import pandas as pd
from pandas import DataFrame
import yfinance as yf


# =====================================================================================
# HARD-CODED CONFIG
# =====================================================================================

SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "yf_ticker_summary_csv")
_make_out_dir_none: None = os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TICKERS_BASE: List[str] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]
START_DATE_STR: str = "2019-01-01"
END_DATE_STR: str = "2025-08-31"


# =====================================================================================
# FUNCTIONS
# =====================================================================================

def _yahoo_symbol_from_base(base_ticker: str) -> str:
    """Convert base ticker (e.g. 'BBCA') into Yahoo Finance symbol (e.g. 'BBCA.JK')."""
    return f"{base_ticker}.JK"


def _download_ticker_history_yf(
    yahoo_symbol: str,
    start_date_str: str,
    end_date_str: str,
) -> DataFrame:
    """Download OHLCV data for a given Yahoo symbol using yfinance."""
    ticker_obj = yf.Ticker(yahoo_symbol)
    df_hist: DataFrame = ticker_obj.history(
        start=start_date_str,
        end=end_date_str,
        auto_adjust=False,
        actions=False
    )
    df_hist.index = pd.to_datetime(df_hist.index, errors="coerce")
    df_hist = df_hist[~df_hist.index.isna()].sort_index()
    return df_hist


def _round_price_columns(df: DataFrame) -> DataFrame:
    """Round all price columns (Open, High, Low, Close, Adj Close) to nearest integer."""
    price_cols: List[str] = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")
    return df


def _finalize_and_save_history(base_ticker: str, df_hist: DataFrame, output_dir: str) -> None:
    """Finalize and save yf_<TICKER>.csv with integer-rounded prices."""
    df_out: DataFrame = _round_price_columns(df_hist.copy())
    df_out.index.name = "Trading Date"

    out_name: str = f"yf_{base_ticker}_raw.csv"
    out_path: str = os.path.join(output_dir, out_name)

    try:
        df_out.to_csv(out_path, index=True)
        print(f"[OK] Saved {out_name} with {len(df_out)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")


def process_single_ticker(base_ticker: str, start_date_str: str, end_date_str: str, output_dir: str) -> None:
    """Download and save one ticker’s adjusted Yahoo Finance data."""
    print(f"[TICKER] Processing {base_ticker} ...")
    yahoo_symbol: str = _yahoo_symbol_from_base(base_ticker)
    df_hist: DataFrame = _download_ticker_history_yf(yahoo_symbol, start_date_str, end_date_str)

    if df_hist.empty:
        print(f"[WARN] No data returned for {base_ticker} ({yahoo_symbol})")
        return

    _finalize_and_save_history(base_ticker, df_hist, output_dir)


# =====================================================================================
# MAIN
# =====================================================================================

def main() -> None:
    if not os.path.isdir(OUTPUT_DIR):
        print(f"[INFO] OUTPUT_DIR does not exist, creating: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for base_ticker in TARGET_TICKERS_BASE:
        process_single_ticker(base_ticker, START_DATE_STR, END_DATE_STR, OUTPUT_DIR)

    print("[DONE] Yahoo Finance download complete.")


if __name__ == "__main__":
    main()
