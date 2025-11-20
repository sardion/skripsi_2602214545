#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_usd_idr_to_calendar_ts.py

Build a daily calendar time series for the USD/IDR exchange rate with:
- calendar_date (index)
- usd_idr (float)

Input:
- data/raw/usd_idr_historical_data_raw.csv
  Columns: Date, Price, Open, High, Low, Vol., Change %

Output:
- data/processed/usd_idr_ts_cal.csv

Calendar window:
- 2020-01-02 .. 2025-08-31

Rules:
- Convert the 'Date' column to datetime.
- Input data is descending (latest to oldest), so it must be sorted ascending.
- Use 'Price' column as usd_idr (float).
- No retrospective fill is needed since data starts before calendar start.
- Forward-fill missing dates between two consecutive rows (weekends/holidays).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, List

import pandas as pd


# =========================
# Config (relative paths)
# =========================
SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = SCRIPT_DIR
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

INPUT_FILENAME: Final[str] = "usd_idr_historical_data_raw.csv"
OUTPUT_FILENAME: Final[str] = "usd_idr_ts_cal.csv"

INPUT_CSV: Final[Path] = RAW_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = PROCESSED_DIR / OUTPUT_FILENAME

CAL_START: Final[date] = date(2020, 1, 2)
CAL_END: Final[date] = date(2025, 8, 31)


# =========================
# Core functions
# =========================
def build_calendar(start: date, end: date) -> pd.DataFrame:
    """
    Build daily calendar DataFrame between start and end (inclusive).
    """
    days: List[date] = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return pd.DataFrame({"calendar_date": pd.to_datetime(days)})


def read_usd_idr_raw(path: Path) -> pd.DataFrame:
    """
    Read USD/IDR historical data CSV, normalize date and sort ascending.
    The 'Date' column is in descending order in the raw data.
    """
    df: pd.DataFrame = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("Input file must contain 'Date' and 'Price' columns.")

    # Parse Date and normalize to midnight
    parsed_dates = pd.to_datetime(df["Date"], errors="raise", dayfirst=False)
    df["Date"] = parsed_dates.dt.normalize()

    # Sort ascending by Date
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    # Keep only required columns
    df = df[["Date", "Price"]].copy()
    df = df.rename(columns={"Price": "usd_idr"})

    # Ensure numeric price (remove commas or other symbols)
    df["usd_idr"] = (
        df["usd_idr"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    return df


def merge_and_forward_fill(calendar_df: pd.DataFrame, rate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge calendar with USD/IDR data and forward-fill missing dates.
    """
    merged = pd.merge(
        calendar_df,
        rate_df,
        left_on="calendar_date",
        right_on="Date",
        how="left"
    ).drop(columns=["Date"], errors="ignore")

    merged = merged.sort_values(by="calendar_date", ascending=True)
    merged["usd_idr"] = merged["usd_idr"].ffill()

    return merged


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read raw USD/IDR historical data
    df_raw = read_usd_idr_raw(INPUT_CSV)

    # 2. Build full daily calendar (2020-01-02 to 2025-08-31)
    df_calendar = build_calendar(CAL_START, CAL_END)

    # 3. Merge and forward-fill missing dates
    df_filled = merge_and_forward_fill(df_calendar, df_raw)

    # 4. Save output
    df_out = df_filled.set_index("calendar_date")
    df_out.to_csv(OUTPUT_CSV, index=True, encoding="utf-8-sig")

    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(df_out)} rows")


if __name__ == "__main__":
    main()
