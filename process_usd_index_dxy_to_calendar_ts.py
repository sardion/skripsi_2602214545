#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_usd_index_dxy_to_calendar_ts.py

Build a daily calendar time series for the US Dollar Index (DXY) with:
- calendar_date (index)
- usd_index_dxy (float)

Input:
- data/raw/usd_index_dxy_historical_data_raw.csv
  Columns: Date, Price, Open, High, Low, Vol., Change %

Output:
- data/processed/usd_index_dxy_ts_cal.csv

Calendar window:
- 2020-01-02 .. 2025-08-31

Rules:
- Source data is descending (latest first); must be sorted ascending.
- Use 'Price' as usd_index_dxy (float), stripping thousand separators.
- Forward-fill values for weekends/holidays.
- No retrospective fill (data starts before 2020-01-02).
"""

from __future__ import annotations

from datetime import date, timedelta
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

INPUT_FILENAME: Final[str] = "usd_index_dxy_historical_data_raw.csv"
OUTPUT_FILENAME: Final[str] = "usd_index_dxy_ts_cal.csv"

INPUT_CSV: Final[Path] = RAW_DIR / INPUT_FILENAME
OUTPUT_CSV: Final[Path] = PROCESSED_DIR / OUTPUT_FILENAME

CAL_START: Final[date] = date(2020, 1, 2)
CAL_END: Final[date] = date(2025, 8, 31)


# =========================
# Helpers
# =========================
def build_calendar(start: date, end: date) -> pd.DataFrame:
    """Create a continuous daily calendar DataFrame with datetime64[ns] dates."""
    days: List[date] = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return pd.DataFrame({"calendar_date": pd.to_datetime(days)})


def read_and_prepare_raw_data(path: Path) -> pd.DataFrame:
    """
    Read the raw Investing.com CSV file, normalize date to datetime64[ns],
    sort ascending, and keep only (calendar_date, usd_index_dxy).
    """
    df: pd.DataFrame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError("Input file must contain 'Date' and 'Price' columns.")

    # Parse Date robustly (Investing.com formats like 'Nov 03, 2025' or '08/29/2025')
    dt = pd.to_datetime(df["Date"], errors="raise", dayfirst=False).dt.normalize()
    df = df.assign(calendar_date=dt)

    # Sort oldest->newest
    df = df.sort_values("calendar_date").reset_index(drop=True)

    # Clean and cast Price
    price = (
        df["Price"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .astype(float)
    )

    return pd.DataFrame({
        "calendar_date": df["calendar_date"],
        "usd_index_dxy": price
    })


def merge_and_forward_fill(calendar_df: pd.DataFrame, rate_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join onto the full calendar and forward-fill usd_index_dxy."""
    merged = pd.merge(calendar_df, rate_df, on="calendar_date", how="left")
    merged = merged.sort_values("calendar_date")
    merged["usd_index_dxy"] = merged["usd_index_dxy"].ffill()
    return merged


# =========================
# Main
# =========================
def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = read_and_prepare_raw_data(INPUT_CSV)
    cal_df = build_calendar(CAL_START, CAL_END)
    filled_df = merge_and_forward_fill(cal_df, raw_df)

    # Save with calendar_date as index (per your convention)
    out = filled_df.set_index("calendar_date")
    out.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    print(f"[OK] Saved {OUTPUT_FILENAME} with {len(out)} rows")


if __name__ == "__main__":
    main()
