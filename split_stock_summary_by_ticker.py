#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incrementally build and maintain per-ticker historical CSVs from daily IDX summaries.

Behavior:
1. Input:
   - Daily CSVs in ./data/raw/idx_stock_summary_csv
     (These are converted from XLSX in Prosedur 1 and contain a 'Trading Date' column)

2. Output:
   - For each target ticker (default: BBCA, ANTM, ICBP, TLKM, ASII):
       ./data/raw/idx_ticker_summary_csv/idx_<TICKER>.csv
     which:
       - contains all known rows for that ticker across time
       - is sorted by Trading Date
       - uses Trading Date as the index in the saved file

3. Sidecar:
   - ./data/raw/idx_ticker_summary_csv/meta/<TICKER>.meta.json
   - Stores "last_trading_date": the most recent Trading Date we have saved
   - If sidecar exists:
       Only new rows AFTER last_trading_date are collected from daily files
   - If sidecar does NOT exist:
       We do a one-time full build for that ticker

4. This avoids full rebuild on every run while keeping reproducibility explicit.

This script is intentionally self-contained (no external helpers, no shared config),
to maximize transparency in an academic setting.
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame, Series


# =====================================================================================
# HARD-CODED CONFIG
# =====================================================================================

# Resolve script directory. In notebooks/Colab, __file__ won't exist.
SCRIPT_DIR: str
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# Input directory:
# Contains 'Stock Summary-YYYYMMDD.csv' with Trading Date column added in Prosedur 1.
INPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "idx_stock_summary_csv")

# Output directory:
# We save per-ticker histories as idx_<TICKER>.csv
OUTPUT_DIR: str = os.path.join(SCRIPT_DIR, "data", "raw", "idx_ticker_summary_csv")

# Sidecar directory for per-ticker metadata
SIDECAR_DIR: str = os.path.join(OUTPUT_DIR, "meta")

# Ensure output dirs exist
_make_out_dir_none: None = os.makedirs(OUTPUT_DIR, exist_ok=True)
_make_meta_dir_none: None = os.makedirs(SIDECAR_DIR, exist_ok=True)

# List of tickers to process by default
TARGET_TICKERS: List[str] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

# Column names we rely on in the source CSVs
COL_TICKER: str = "Stock Code"
COL_DATE: str = "Trading Date"

# Sidecar JSON filename suffix
SIDECAR_SUFFIX: str = ".meta.json"


# =====================================================================================
# INLINE UTILITY FUNCTIONS
# =====================================================================================

def _sidecar_path_for_ticker(ticker: str) -> str:
    """
    Get the path to meta/<TICKER>.meta.json
    """
    sidecar_path: str = os.path.join(SIDECAR_DIR, f"{ticker}{SIDECAR_SUFFIX}")
    return sidecar_path


def _read_sidecar_last_date(ticker: str) -> Optional[str]:
    """
    Read the latest Trading Date we have already stored for this ticker,
    from its sidecar JSON. Return as "YYYY-MM-DD" or None if not found.
    """
    sidecar_path: str = _sidecar_path_for_ticker(ticker)
    if not os.path.isfile(sidecar_path):
        return None

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            meta: Dict[str, Any] = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read sidecar for {ticker}: {e}")
        return None

    last_date_str: Optional[str] = meta.get("last_trading_date")
    if last_date_str is None:
        return None

    return last_date_str


def _write_sidecar_last_date(ticker: str, last_date_str: str) -> None:
    """
    Persist the latest Trading Date for this ticker into the sidecar JSON.
    """
    sidecar_path: str = _sidecar_path_for_ticker(ticker)
    meta: Dict[str, str] = {
        "ticker": ticker,
        "last_trading_date": last_date_str,
    }
    try:
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write sidecar for {ticker}: {e}")


def _load_existing_ticker_csv(ticker: str) -> DataFrame:
    """
    Load the already-saved idx_<TICKER>.csv if it exists.
    If it doesn't exist, return empty DataFrame.
    """
    out_path: str = os.path.join(OUTPUT_DIR, f"idx_{ticker}.csv")
    if not os.path.isfile(out_path):
        return pd.DataFrame()

    try:
        df_existing: DataFrame = pd.read_csv(out_path)
    except Exception as e:
        print(f"[WARN] Failed to read existing idx_{ticker}.csv: {e}")
        df_existing = pd.DataFrame()

    # Re-normalize the Trading Date if it was saved as index
    # During save we use .to_csv(index=True) with Trading Date as index,
    # so pandas.load will create a column out of that index.
    if COL_DATE not in df_existing.columns and "Unnamed: 0" in df_existing.columns:
        df_existing.rename(columns={"Unnamed: 0": COL_DATE}, inplace=True)

    # Make sure date column is datetime
    if COL_DATE in df_existing.columns:
        df_existing[COL_DATE] = pd.to_datetime(df_existing[COL_DATE], errors="coerce")
        df_existing = df_existing.dropna(subset=[COL_DATE])
        df_existing = df_existing.sort_values(by=[COL_DATE], ascending=True)

    return df_existing


def _collect_rows_for_ticker_from_file(
    csv_path: str,
    ticker: str,
    date_col: str,
    ticker_col: str,
) -> DataFrame:
    """
    Read a single daily 'Stock Summary-YYYYMMDD.csv',
    filter to rows for the given ticker,
    normalize Trading Date,
    return that subset.
    """
    try:
        df_daily: DataFrame = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Cannot read {csv_path}: {e}")
        return pd.DataFrame()

    if (ticker_col not in df_daily.columns) or (date_col not in df_daily.columns):
        print(f"[WARN] Missing '{ticker_col}' or '{date_col}' in {csv_path}, skipping")
        return pd.DataFrame()

    df_ticker_rows: DataFrame = df_daily[df_daily[ticker_col] == ticker].copy()
    if df_ticker_rows.empty:
        return df_ticker_rows

    # Normalize Trading Date
    df_ticker_rows[date_col] = pd.to_datetime(df_ticker_rows[date_col], errors="coerce")
    df_ticker_rows = df_ticker_rows.dropna(subset=[date_col])

    return df_ticker_rows


def _filter_rows_newer_than(
    df_rows: DataFrame,
    last_saved_date_str: Optional[str],
    date_col: str,
) -> DataFrame:
    """
    Keep only rows where Trading Date is strictly AFTER last_saved_date_str.
    If last_saved_date_str is None, return all rows.
    """
    if last_saved_date_str is None:
        return df_rows.copy()

    # Parse "YYYY-MM-DD"
    try:
        last_dt: datetime = datetime.strptime(last_saved_date_str, "%Y-%m-%d")
    except Exception:
        # If can't parse, keep everything (fail-open)
        return df_rows.copy()

    df_tmp: DataFrame = df_rows.copy()
    df_tmp[date_col] = pd.to_datetime(df_tmp[date_col], errors="coerce")

    mask_new: Series = df_tmp[date_col] > last_dt
    df_filtered: DataFrame = df_tmp[mask_new].copy()
    return df_filtered


def _append_and_dedupe(
    df_existing: DataFrame,
    df_new: DataFrame,
    date_col: str,
    ticker_col: str,
) -> DataFrame:
    """
    Append new rows to existing rows, drop duplicates,
    and sort ascending by Trading Date.
    """
    # Ensure datetime
    if date_col in df_existing.columns:
        df_existing[date_col] = pd.to_datetime(df_existing[date_col], errors="coerce")
    if date_col in df_new.columns:
        df_new[date_col] = pd.to_datetime(df_new[date_col], errors="coerce")

    df_all: DataFrame = pd.concat([df_existing, df_new], ignore_index=True)

    # Drop dups using Trading Date + Stock Code
    if (date_col in df_all.columns) and (ticker_col in df_all.columns):
        df_all = df_all.drop_duplicates(
            subset=[date_col, ticker_col],
            keep="first"
        )
    elif date_col in df_all.columns:
        df_all = df_all.drop_duplicates(subset=[date_col], keep="first")

    # Sort
    if date_col in df_all.columns:
        df_all = df_all.sort_values(by=[date_col], ascending=True)

    return df_all


def _save_ticker_csv_and_update_sidecar(
    ticker: str,
    df_final: DataFrame,
    date_col: str,
) -> None:
    """
    Save final merged history to idx_<TICKER>.csv with Trading Date as index,
    then update sidecar last_trading_date.
    """
    # Normalize date column again to be safe
    df_final[date_col] = pd.to_datetime(df_final[date_col], errors="coerce")
    df_final = df_final.dropna(subset=[date_col])
    df_final = df_final.sort_values(by=[date_col], ascending=True)

    # Set Trading Date as index
    df_out: DataFrame = df_final.set_index(date_col)

    out_csv_path: str = os.path.join(OUTPUT_DIR, f"idx_{ticker}.csv")

    # Write (overwrite) idx_<TICKER>.csv
    try:
        _none_val: None = df_out.to_csv(out_csv_path, index=True)
    except Exception as e:
        print(f"[ERROR] Failed to write {out_csv_path}: {e}")
        return

    # Update sidecar based on last Trading Date in index
    if not df_out.index.empty:
        last_dt: datetime = pd.to_datetime(df_out.index.max()).to_pydatetime()
        last_date_str: str = last_dt.strftime("%Y-%m-%d")
        _write_sidecar_last_date(ticker, last_date_str)

    print(f"[OK] Updated idx_{ticker}.csv with {len(df_out)} rows")


# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main() -> None:
    # Sanity checks
    if not os.path.isdir(INPUT_DIR):
        print(f"[FATAL] INPUT_DIR does not exist: {INPUT_DIR}")
        sys.exit(1)

    if not os.path.isdir(OUTPUT_DIR):
        print(f"[INFO] OUTPUT_DIR does not exist, creating: {OUTPUT_DIR}")
        _mk_out: None = os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(SIDECAR_DIR):
        print(f"[INFO] SIDECAR_DIR does not exist, creating: {SIDECAR_DIR}")
        _mk_meta: None = os.makedirs(SIDECAR_DIR, exist_ok=True)

    # Get list of daily market-wide CSVs
    all_files: List[str] = sorted(os.listdir(INPUT_DIR))
    daily_csv_files: List[str] = [f for f in all_files if f.lower().endswith(".csv")]

    if not daily_csv_files:
        print(f"[WARN] No .csv files found in {INPUT_DIR}")
        sys.exit(0)

    # Process ticker by ticker
    for ticker in TARGET_TICKERS:
        print(f"[TICKER] Processing {ticker} ...")

        # Load existing ticker history if present
        df_existing: DataFrame = _load_existing_ticker_csv(ticker)

        # Read last saved Trading Date from sidecar
        last_saved_date_str: Optional[str] = _read_sidecar_last_date(ticker)

        collected_new_rows: List[DataFrame] = []

        # Walk each daily summary file, pull rows for this ticker
        for fname in daily_csv_files:
            full_path: str = os.path.join(INPUT_DIR, fname)

            df_rows_for_ticker: DataFrame = _collect_rows_for_ticker_from_file(
                csv_path=full_path,
                ticker=ticker,
                date_col=COL_DATE,
                ticker_col=COL_TICKER,
            )

            if df_rows_for_ticker.empty:
                continue

            # Only keep rows strictly after last_saved_date_str
            df_new_only: DataFrame = _filter_rows_newer_than(
                df_rows=df_rows_for_ticker,
                last_saved_date_str=last_saved_date_str,
                date_col=COL_DATE,
            )

            if not df_new_only.empty:
                collected_new_rows.append(df_new_only)

        # Concatenate newly discovered rows (if any)
        if collected_new_rows:
            df_concat_new: DataFrame = pd.concat(collected_new_rows, ignore_index=True)
        else:
            df_concat_new = pd.DataFrame()

        # Merge old + new, drop duplicates, sort
        df_final: DataFrame = _append_and_dedupe(
            df_existing=df_existing,
            df_new=df_concat_new,
            date_col=COL_DATE,
            ticker_col=COL_TICKER,
        )

        # Save back to idx_<TICKER>.csv and update sidecar
        _save_ticker_csv_and_update_sidecar(
            ticker=ticker,
            df_final=df_final,
            date_col=COL_DATE,
        )

    print("[DONE] Incremental per-ticker split complete.")


if __name__ == "__main__":
    main()
