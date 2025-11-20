#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_all_features_ts_cal.py

Integrate all calendar-aligned (_ts_cal.csv) feature sources into a single
wide, calendar-day-based table per ticker, preserving original values
(including % signs and negative prefixes).

For each ticker:
- Start from {TICKER}_summary_ts_cal.csv as the base (calendar_date is the index).
- Left-join macro, FX/risk/commodity, calendar flags, TA indicators, and IHSG features.
- Do NOT cast/clean values; keep strings as-is to preserve % and signs.
- Save to data/raw_features/{TICKER}_integrated_features_all_raw.csv

Assumptions:
- All _ts_cal.csv live in data/processed/
- Each CSV has calendar_date as the first/index column
- calendar_date alignment is tz-naive and ascending

Author: (your name)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Dict
import pandas as pd


# =========================
# Config
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RAW_FEATURES_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw_features"

TICKERS: Final[List[str]] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

# Base summary columns to copy as the starting frame
SUMMARY_COLS: Final[List[str]] = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "bid_volume",
    "offer_volume",
    "foreign_sell",
    "foreign_buy",
]

# Join plan: file_name -> columns to add
JOIN_SOURCES: Final[Dict[str, List[str]]] = {
    # US Macro
    "us_fed_fund_rate_ts_cal.csv": ["us_fed_fund_rate", "days_since_us_fed_fund_rate"],
    "us_gdp_qoq_ts_cal.csv": ["us_gdp_qoq", "days_since_us_gdp_qoq"],
    "us_core_cpi_mom_ts_cal.csv": ["us_core_cpi_mom", "days_since_us_core_cpi_mom"],
    # ID Macro
    "id_bi_rate_ts_cal.csv": ["id_bi_rate", "days_since_id_bi_rate"],
    "id_core_inflation_yoy_ts_cal.csv": ["id_core_inflation_yoy", "days_since_id_core_inflation_yoy"],
    "id_gdp_qoq_ts_cal.csv": ["id_gdp_qoq", "days_since_id_gdp_qoq"],
    "id_inflation_mom_ts_cal.csv": ["id_inflation_mom", "days_since_id_inflation_mom"],
    "id_retail_sales_yoy_ts_cal.csv": ["id_retail_sales_yoy", "days_since_id_retail_sales_yoy"],
    # FX / Risk / Commodity
    "usd_idr_ts_cal.csv": ["usd_idr"],
    "usd_index_dxy_ts_cal.csv": ["usd_index_dxy"],
    "crude_oil_wti_ts_cal.csv": ["crude_oil_wti"],
    "sp500_vix_ts_cal.csv": ["sp500_vix"],
    "gold_futures_ts_cal.csv": ["gold_futures"],
    # Calendar flags
    "calendar_features_ts_cal.csv": [
        "is_sunday","is_monday","is_tuesday","is_wednesday",
        "is_thursday","is_friday","is_saturday",
        "is_trading_day","next_is_trading_day","is_month_start","is_month_end",
    ],
    # IHSG features
    "ihsg_features_ts_cal.csv": [
        "ihsg_return_lag_1","ihsg_std_dev_7","ihsg_std_dev_14","ihsg_std_dev_21"
    ],
}

# Per-ticker TA filename pattern and columns
TA_COLS: Final[List[str]] = [
    "rsi_7","rsi_14",
    "sma_7","sma_14","sma_21","sma_100",
    "std_dev_7","std_dev_14","std_dev_21","std_dev_100",
]


# =========================
# Helpers
# =========================
def read_ts_with_calendar(path: Path) -> pd.DataFrame:
    """
    Read a _ts_cal.csv where 'calendar_date' is stored as the first column (index).
    Return a DataFrame with an explicit 'calendar_date' column (string preserved for values).
    """
    # Read as strings to preserve % and signs in data columns
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    # If 'calendar_date' is an index column, ensure it is a column
    # Many of our writers used index=True when saving, which yields a first column named 'calendar_date'
    if "calendar_date" not in df.columns:
        # if first column is unnamed index, rename it
        first_col = df.columns[0]
        if first_col.lower() == "unnamed: 0":
            df = df.rename(columns={first_col: "calendar_date"})
        else:
            # assume first column is actually calendar_date
            df = df.rename(columns={first_col: "calendar_date"})
    return df


def left_join_on_calendar(base: pd.DataFrame, add: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Left-join selected columns from 'add' onto 'base' using 'calendar_date'.
    Missing requested columns are ignored (robustness).
    """
    cols_available = [c for c in cols if c in add.columns]
    if not cols_available:
        return base
    subset = add[["calendar_date"] + cols_available].copy()
    merged = base.merge(subset, on="calendar_date", how="left")
    return merged


def process_ticker(ticker: str) -> None:
    """
    Build {TICKER}_integrated_features_all_raw.csv by starting from
    {TICKER}_summary_ts_cal.csv then left-joining all sources.
    """
    RAW_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Base summary file
    base_file = PROCESSED_DIR / f"{ticker}_summary_ts_cal.csv"
    base_df = read_ts_with_calendar(base_file)

    # keep only desired base columns (plus calendar_date)
    keep_cols = ["calendar_date"] + [c for c in SUMMARY_COLS if c in base_df.columns]
    base_df = base_df[keep_cols].copy()

    # 2) Generic sources
    for fname, cols in JOIN_SOURCES.items():
        fpath = PROCESSED_DIR / fname
        if not fpath.exists():
            print(f"[WARN] Missing source: {fname} — skipping.")
            continue
        src_df = read_ts_with_calendar(fpath)
        base_df = left_join_on_calendar(base_df, src_df, cols)

    # 3) Per-ticker TA indicators
    ta_file = PROCESSED_DIR / f"{ticker}_ta_indicators_ts_cal.csv"
    if ta_file.exists():
        ta_df = read_ts_with_calendar(ta_file)
        base_df = left_join_on_calendar(base_df, ta_df, TA_COLS)
    else:
        print(f"[WARN] Missing TA file for {ticker}: {ta_file.name} — skipping TA merge.")

    # Final ordering: calendar_date first; others in current order
    out = base_df.copy()

    # Write output
    out_file = RAW_FEATURES_DIR / f"{ticker}_integrated_features_all_raw.csv"
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {out_file.name} with {len(out)} rows")


def main() -> None:
    for ticker in TICKERS:
        print(f"Integrating features for {ticker} ...")
        process_ticker(ticker)


if __name__ == "__main__":
    main()
