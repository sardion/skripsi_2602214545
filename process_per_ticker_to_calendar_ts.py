#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process integrated per-ticker data into full calendar-based daily time series.

Prosedur 4 behavior:
1. Input tickers (default ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"])
2. For each ticker <T>:
   - Read:
        ./data/raw/integrated_ticker_summary_csv/integrated_<T>.csv
     This file is expected to have:
        index or column "Trading Date" (YYYY-MM-DD as string or date)
        Open, High, Low, Close from Yahoo Finance
        Volume, Bid Volume, Offer Volume, Foreign Sell, Foreign Buy from IDX
   - Build a continuous calendar date range from
        2020-01-02 to 2025-08-31 (inclusive),
     call this column "calendar_date".
   - Map the integrated trading data onto this daily calendar.
   - Forward fill missing values in between trading days.
   - Save final calendar-indexed result to:
        ./data/processed/<T>_summary_ts_cal.csv
     with "calendar_date" as the index, and columns:
        open_price
        high_price
        low_price
        close_price
        volume
        bid_volume
        offer_volume
        foreign_sell
        foreign_buy

This script is intentionally standalone (no shared modules), uses hardcoded paths,
and applies explicit date logic for transparency and reproducibility.
"""

from __future__ import annotations

import os
import sys
from typing import List
import pandas as pd
from pandas import DataFrame, Series


# =====================================================================================
# HARD-CODED CONFIG
# =====================================================================================

SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

INTEGRATED_INPUT_DIR: str = os.path.join(
    SCRIPT_DIR, "data", "raw", "integrated_ticker_summary_csv"
)
OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "processed")
_make_out_dir_none: None = os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TICKERS: List[str] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

# Calendar date range (starts from first actual trading day)
CAL_START_STR: str = "2020-01-02"
CAL_END_STR: str = "2025-08-31"

# Integrated column names
COL_TRADING_DATE_INTEGRATED: str = "Trading Date"
COL_OPEN_INTEGRATED: str = "Open"
COL_HIGH_INTEGRATED: str = "High"
COL_LOW_INTEGRATED: str = "Low"
COL_CLOSE_INTEGRATED: str = "Close"
COL_VOLUME_INTEGRATED: str = "Volume"
COL_BIDVOL_INTEGRATED: str = "Bid Volume"
COL_OFFERVOL_INTEGRATED: str = "Offer Volume"
COL_FORSELL_INTEGRATED: str = "Foreign Sell"
COL_FORBUY_INTEGRATED: str = "Foreign Buy"

# Final output schema
COL_CAL_DATE_FINAL: str = "calendar_date"
COL_OPEN_FINAL: str = "open_price"
COL_HIGH_FINAL: str = "high_price"
COL_LOW_FINAL: str = "low_price"
COL_CLOSE_FINAL: str = "close_price"
COL_VOLUME_FINAL: str = "volume"
COL_BIDVOL_FINAL: str = "bid_volume"
COL_OFFERVOL_FINAL: str = "offer_volume"
COL_FORSELL_FINAL: str = "foreign_sell"
COL_FORBUY_FINAL: str = "foreign_buy"

FINAL_COL_ORDER: List[str] = [
    COL_CAL_DATE_FINAL,
    COL_OPEN_FINAL,
    COL_HIGH_FINAL,
    COL_LOW_FINAL,
    COL_CLOSE_FINAL,
    COL_VOLUME_FINAL,
    COL_BIDVOL_FINAL,
    COL_OFFERVOL_FINAL,
    COL_FORSELL_FINAL,
    COL_FORBUY_FINAL,
]


# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================

def _build_calendar_df(start_date_str: str, end_date_str: str) -> DataFrame:
    """Build a DataFrame with a full daily calendar range."""
    cal_range: Series = pd.date_range(start=start_date_str, end=end_date_str, freq="D")
    return pd.DataFrame({COL_CAL_DATE_FINAL: cal_range})


def _load_integrated_ticker(ticker: str) -> DataFrame:
    """Load integrated_<ticker>.csv and normalize Trading Date."""
    in_path: str = os.path.join(INTEGRATED_INPUT_DIR, f"integrated_{ticker}.csv")
    if not os.path.isfile(in_path):
        print(f"[WARN] Integrated file not found for {ticker}: {in_path}")
        return pd.DataFrame()

    df_raw: DataFrame = pd.read_csv(in_path)

    if COL_TRADING_DATE_INTEGRATED not in df_raw.columns and "Unnamed: 0" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Unnamed: 0": COL_TRADING_DATE_INTEGRATED})
    if COL_TRADING_DATE_INTEGRATED not in df_raw.columns and "Date" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Date": COL_TRADING_DATE_INTEGRATED})

    if COL_TRADING_DATE_INTEGRATED not in df_raw.columns:
        print(f"[WARN] '{COL_TRADING_DATE_INTEGRATED}' not found in {in_path}")
        return pd.DataFrame()

    df_raw[COL_TRADING_DATE_INTEGRATED] = pd.to_datetime(
        df_raw[COL_TRADING_DATE_INTEGRATED], errors="coerce"
    ).dt.normalize()
    df_raw = df_raw.dropna(subset=[COL_TRADING_DATE_INTEGRATED])
    df_raw = df_raw.sort_values(by=[COL_TRADING_DATE_INTEGRATED], ascending=True)
    return df_raw


def _map_integrated_to_model_schema(df_integrated: DataFrame) -> DataFrame:
    """Map integrated columns to final calendar schema."""
    df_mapped: DataFrame = pd.DataFrame({
        COL_CAL_DATE_FINAL: df_integrated[COL_TRADING_DATE_INTEGRATED],
        COL_OPEN_FINAL:     df_integrated.get(COL_OPEN_INTEGRATED, pd.NA),
        COL_HIGH_FINAL:     df_integrated.get(COL_HIGH_INTEGRATED, pd.NA),
        COL_LOW_FINAL:      df_integrated.get(COL_LOW_INTEGRATED, pd.NA),
        COL_CLOSE_FINAL:    df_integrated.get(COL_CLOSE_INTEGRATED, pd.NA),
        COL_VOLUME_FINAL:   df_integrated.get(COL_VOLUME_INTEGRATED, pd.NA),
        COL_BIDVOL_FINAL:   df_integrated.get(COL_BIDVOL_INTEGRATED, pd.NA),
        COL_OFFERVOL_FINAL: df_integrated.get(COL_OFFERVOL_INTEGRATED, pd.NA),
        COL_FORSELL_FINAL:  df_integrated.get(COL_FORSELL_INTEGRATED, pd.NA),
        COL_FORBUY_FINAL:   df_integrated.get(COL_FORBUY_INTEGRATED, pd.NA),
    })
    return df_mapped.sort_values(by=[COL_CAL_DATE_FINAL], ascending=True)


def _merge_calendar_with_data(df_calendar: DataFrame, df_ticker: DataFrame) -> DataFrame:
    """Left-join ticker data onto full calendar by calendar_date."""
    df_merged: DataFrame = pd.merge(
        left=df_calendar,
        right=df_ticker,
        on=COL_CAL_DATE_FINAL,
        how="left",
        sort=True,
    )
    return df_merged.sort_values(by=[COL_CAL_DATE_FINAL], ascending=True)


def _apply_forward_fill(df_full: DataFrame) -> DataFrame:
    """Forward fill missing values between trading days only."""
    df_out: DataFrame = df_full.copy()
    cols_to_ffill: List[str] = [
        COL_OPEN_FINAL,
        COL_HIGH_FINAL,
        COL_LOW_FINAL,
        COL_CLOSE_FINAL,
        COL_VOLUME_FINAL,
        COL_BIDVOL_FINAL,
        COL_OFFERVOL_FINAL,
        COL_FORSELL_FINAL,
        COL_FORBUY_FINAL,
    ]
    df_out[cols_to_ffill] = df_out[cols_to_ffill].ffill()
    return df_out


def _save_calendar_ts(ticker: str, df_filled: DataFrame) -> None:
    """Save final calendar-indexed time series for this ticker."""
    df_out: DataFrame = df_filled.set_index(COL_CAL_DATE_FINAL)
    out_name: str = f"{ticker}_summary_ts_cal.csv"
    out_path: str = os.path.join(OUTPUT_DIR, out_name)
    try:
        df_out.to_csv(out_path, index=True)
        print(f"[OK] Saved {out_name} with {len(df_out)} rows")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")


# =====================================================================================
# MAIN PIPELINE PER TICKER
# =====================================================================================

def process_single_ticker(ticker: str) -> None:
    """Process one ticker end-to-end."""
    print(f"[TICKER] Processing {ticker} ...")
    df_integrated: DataFrame = _load_integrated_ticker(ticker)
    if df_integrated.empty:
        print(f"[WARN] Skipping {ticker}: integrated data empty.")
        return

    df_ticker_schema: DataFrame = _map_integrated_to_model_schema(df_integrated)
    df_calendar: DataFrame = _build_calendar_df(CAL_START_STR, CAL_END_STR)
    df_merged: DataFrame = _merge_calendar_with_data(df_calendar, df_ticker_schema)
    df_filled: DataFrame = _apply_forward_fill(df_merged)
    _save_calendar_ts(ticker, df_filled)


# =====================================================================================
# SCRIPT MAIN
# =====================================================================================

def main() -> None:
    if not os.path.isdir(INTEGRATED_INPUT_DIR):
        print(f"[FATAL] INTEGRATED_INPUT_DIR does not exist: {INTEGRATED_INPUT_DIR}")
        sys.exit(1)
    if not os.path.isdir(OUTPUT_DIR):
        print(f"[INFO] OUTPUT_DIR does not exist, creating: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker in TARGET_TICKERS:
        process_single_ticker(ticker)

    print("[DONE] Calendar time series generation complete.")


if __name__ == "__main__":
    main()
