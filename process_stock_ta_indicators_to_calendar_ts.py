#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_stock_ta_indicators_to_calendar_ts.py

Build daily calendar time series for each selected ticker’s technical indicators:
- calendar_date (index)
- rsi_7, rsi_14, sma_7, sma_14, sma_21, sma_100,
  std_dev_7, std_dev_14, std_dev_21, std_dev_100

Input:
- data/raw/{TICKER}_ta_indicators_features_raw.csv
  Columns: date, close_price, rsi_7, rsi_14, sma_7, sma_14,
           sma_21, sma_100, std_dev_7, std_dev_14,
           std_dev_21, std_dev_100

Output:
- data/processed/{TICKER}_ta_indicators_ts_cal.csv

Calendar window:
- 2020-01-02 .. 2025-08-31

Rules:
- Ignore close_price.
- If date column contains tz-aware datetimes (+07:00), convert to tz-naive.
- Forward-fill between missing calendar days (weekends/holidays).
- If earliest Jan 2020 record is after Jan 2, but there’s a Dec 2019 record,
  use that last Dec-2019 value for Jan 2–4 (initial forward-fill).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Final, List
import pandas as pd


# =========================
# Config
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

TICKERS: Final[List[str]] = ["BBCA", "ANTM", "ICBP", "TLKM", "ASII"]

CAL_START: Final[date] = date(2020, 1, 2)
CAL_END: Final[date] = date(2025, 8, 31)


# =========================
# Helpers
# =========================
def build_calendar(start: date, end: date) -> pd.DataFrame:
    """Generate continuous calendar from start to end."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return pd.DataFrame({"calendar_date": pd.to_datetime(days)})


def to_tznaive_normalized(s: pd.Series) -> pd.Series:
    """Convert tz-aware datetimes to tz-naive and normalize to midnight."""
    dt = pd.to_datetime(s, errors="raise")
    if getattr(dt.dtype, "tz", None) is not None:
        dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()


def read_raw_indicators(path: Path) -> pd.DataFrame:
    """Read raw technical indicator file and clean columns."""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "date",
        "rsi_7",
        "rsi_14",
        "sma_7",
        "sma_14",
        "sma_21",
        "sma_100",
        "std_dev_7",
        "std_dev_14",
        "std_dev_21",
        "std_dev_100",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = to_tznaive_normalized(df["date"])
    df = df[required_cols].sort_values("date").reset_index(drop=True)
    return df


def merge_and_fill(calendar_df: pd.DataFrame, ta_df: pd.DataFrame) -> pd.DataFrame:
    """Merge onto calendar, fill missing days, and handle early-period forward fill."""
    pre_start = ta_df.loc[ta_df["date"] < pd.Timestamp(CAL_START)]
    if not pre_start.empty:
        latest_pre = pre_start.iloc[-1]
        first_in_2020 = ta_df.loc[ta_df["date"] >= pd.Timestamp(CAL_START)].head(1)
        if not first_in_2020.empty and first_in_2020.iloc[0]["date"].date() > CAL_START:
            pseudo_row = pd.DataFrame(
                {c: [latest_pre[c]] for c in ta_df.columns}
            )
            pseudo_row["date"] = pd.Timestamp(CAL_START)
            ta_df = pd.concat([ta_df, pseudo_row], ignore_index=True)

    merged = pd.merge(calendar_df, ta_df, left_on="calendar_date", right_on="date", how="left")
    merged = merged.drop(columns=["date"], errors="ignore").sort_values("calendar_date")

    cols_to_fill = [
        "rsi_7",
        "rsi_14",
        "sma_7",
        "sma_14",
        "sma_21",
        "sma_100",
        "std_dev_7",
        "std_dev_14",
        "std_dev_21",
        "std_dev_100",
    ]
    for col in cols_to_fill:
        merged[col] = merged[col].ffill()

    return merged


# =========================
# Main
# =========================
def process_ticker(ticker: str) -> None:
    """Process a single ticker's TA indicator file."""
    input_csv = RAW_DIR / f"{ticker}_ta_indicators_features_raw.csv"
    output_csv = PROCESSED_DIR / f"{ticker}_ta_indicators_ts_cal.csv"

    ta_df = read_raw_indicators(input_csv)
    cal_df = build_calendar(CAL_START, CAL_END)
    filled_df = merge_and_fill(cal_df, ta_df)

    out = filled_df.set_index("calendar_date")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, encoding="utf-8-sig", index=True)
    print(f"[OK] Saved {output_csv.name} with {len(out)} rows")


def main() -> None:
    for ticker in TICKERS:
        print(f"Processing {ticker} ...")
        process_ticker(ticker)


if __name__ == "__main__":
    main()
